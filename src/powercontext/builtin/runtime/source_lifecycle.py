# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scope-local, restart-safe propagation of explicit evidence availability."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory import InvalidMemoryCitationError, MemoryEntryNotFoundError
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.source_lifecycle import LifecycleNode, SourceLifecycleRepository
from powercontext.builtin.records import BaseAccessError
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.errors import ArtifactNotFoundError
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """Invalid dependency binding, lifecycle event, or generation resolution."""


class SourceLifecycleEvent(BaseModel):
    """One exact, replayable Source lifecycle revision and sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str = Field(min_length=1)
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool
    reason: str = Field(min_length=1)

    @field_validator("node_id", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lifecycle text cannot be blank")  # noqa: TRY003
        return value


def _node_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("node_id must be nonblank text")  # noqa: TRY003
    return value


def _propagate(nodes: dict[str, LifecycleNode]) -> dict[str, LifecycleNode]:
    try:
        order = tuple(TopologicalSorter({key: node.parents for key, node in nodes.items()}).static_order())
    except CycleError as error:
        raise SourceLifecycleError("dependency cycle") from error  # noqa: TRY003
    updated: dict[str, LifecycleNode] = {}
    for node_id in order:
        if node_id not in nodes:
            raise SourceLifecycleError("missing dependency")  # noqa: TRY003
        node = nodes[node_id]
        if node.kind != "source":
            support = [updated[parent].available for parent in node.parents]
            available = all(support) if node.policy == "all" else any(support)
            node = node.model_copy(update={"available": available})
        updated[node_id] = node
    return updated


class ScopedSourceLifecycle:
    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self.scope_id = validate_scope_id(scope_id)
        self._repository = SourceLifecycleRepository()

    def _storage(self) -> AsyncDatabase:
        if self._database is None:
            raise SourceLifecycleError("lifecycle requires relational persistence")  # noqa: TRY003
        return self._database

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        if type(revision) is not int or revision < 1 or not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("invalid Source binding or lifecycle revision")  # noqa: TRY003
        await self._register(
            LifecycleNode(node_id=_node_id(node_id), kind="source", reference=source_ref, revision=revision)
        )

    async def register_memory(
        self, node_id: str, citation: MemoryCitation, *, parents: Sequence[str], policy: Literal["all", "any"]
    ) -> None:
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("invalid Memory citation")  # noqa: TRY003
        await self._register_derived(node_id, citation, "memory", parents, policy)

    async def register_topic(
        self, node_id: str, artifact_ref: ArtifactRef, *, parents: Sequence[str], policy: Literal["all", "any"]
    ) -> None:
        if not isinstance(artifact_ref, ArtifactRef) or artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("invalid Topic reference")  # noqa: TRY003
        await self._register_derived(node_id, artifact_ref, "topic", parents, policy)

    async def _register_derived(
        self,
        node_id: str,
        reference: MemoryCitation | ArtifactRef,
        kind: Literal["memory", "topic"],
        parents: Sequence[str],
        policy: Literal["all", "any"],
    ) -> None:
        node_id = _node_id(node_id)
        if isinstance(parents, str) or not parents or policy not in {"all", "any"}:
            raise SourceLifecycleError("derived nodes need parents and an all/any policy")  # noqa: TRY003
        parent_ids = tuple(_node_id(parent) for parent in parents)
        if len(set(parent_ids)) != len(parent_ids) or node_id in parent_ids:
            raise SourceLifecycleError("duplicate dependency or self-cycle")  # noqa: TRY003
        await self._register(
            LifecycleNode(
                node_id=node_id, kind=kind, reference=reference, parents=tuple(sorted(parent_ids)), policy=policy
            )
        )

    async def _register(self, node: LifecycleNode) -> None:
        database = self._storage()
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            await self._read_original(node)
            async with database.transaction() as connection:
                nodes = await self._repository.load(connection, self.scope_id)
                existing = nodes.get(node.node_id)
                if existing is not None:
                    if existing.model_copy(update={"available": True, "last_event": None}) != node:
                        raise SourceLifecycleError("node_id is already bound differently")  # noqa: TRY003
                else:
                    if any(parent not in nodes for parent in node.parents):
                        raise SourceLifecycleError("parents must be registered first")  # noqa: TRY003
                    nodes[node.node_id] = node
                    node = _propagate(nodes)[node.node_id]
                    nodes[node.node_id] = node
                    await self._repository.save(connection, self.scope_id, node, create=True)
            # A repeated registration can repair an interrupted initial state write.
            await self._sync_memory(nodes, reason="source lifecycle registration")

    async def _read_original(self, node: LifecycleNode) -> None:
        try:
            reference = node.reference
            if isinstance(reference, SourceRef):
                await self._runtime.records.for_scope(self.scope_id).get_source(
                    reference.source_type, reference.source_id
                )
            elif isinstance(reference, MemoryCitation):
                await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=reference))
            else:
                await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                    reference.family, reference.artifact_id, reference.revision
                )
        except (BaseAccessError, ArtifactNotFoundError, MemoryEntryNotFoundError, InvalidMemoryCitationError) as error:
            raise SourceLifecycleError("registration reference is not readable") from error  # noqa: TRY003

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("apply requires a SourceLifecycleEvent")  # noqa: TRY003
        database = self._storage()
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            async with database.transaction() as connection:
                nodes = await self._repository.load(connection, self.scope_id)
                source = nodes.get(event.node_id)
                if source is None or source.kind != "source" or source.revision != event.revision:
                    raise SourceLifecycleError("unknown Source or lifecycle revision mismatch")  # noqa: TRY003
                previous = source.last_event
                payload = event.model_dump(mode="json")
                if previous is not None:
                    sequence = SourceLifecycleEvent.model_validate(previous).sequence
                    if event.sequence < sequence or (event.sequence == sequence and payload != previous):
                        raise SourceLifecycleError("stale or conflicting Source event")  # noqa: TRY003
                nodes[event.node_id] = source.model_copy(update={"available": event.available, "last_event": payload})
                updated = _propagate(nodes)
                # Desired state is durable before domain writes. Replaying this
                # exact event always reconciles actual native state, including
                # after a partial write failure or process restart.
                for node_id, node in updated.items():
                    if node != nodes[node_id] or node_id == event.node_id:
                        await self._repository.save(connection, self.scope_id, node)
            await self._sync_memory(updated, reason=event.reason)
            return sorted(node_id for node_id, node in updated.items() if node.available)

    async def _sync_memory(self, nodes: dict[str, LifecycleNode], *, reason: str) -> None:
        # Preserve the complete reason in the durable event; native Memory
        # change notes have a separate 512-character bound.
        reason = unicodedata.normalize("NFC", reason).strip()[:512]
        memory_nodes = [node for node in nodes.values() if node.kind == "memory"]
        if not memory_nodes:
            return
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            entries = {entry.entry_id: entry for entry in await service.entries(current)}
            states = {item.entry_id: item.state for item in current.content.manifest.entries}
            activate, deactivate = [], []
            for node in memory_nodes:
                reference = node.reference
                if not isinstance(reference, MemoryCitation):
                    raise SourceLifecycleError("invalid persisted Memory binding")  # noqa: TRY003
                entry = entries.get(reference.entry_id)
                if entry is None or entry.entry_version_id != reference.entry_version_id:
                    raise SourceLifecycleError("registered Memory body version changed")  # noqa: TRY003
                expected = "active" if node.available else "inactive"
                if states[entry.entry_id] != expected:
                    (activate if node.available else deactivate).append(entry)
            if deactivate:
                current = await service.forget(current, entries=deactivate, reason=reason)
            if activate:
                await service.reactivate(current, entries=activate, reason=reason)

    async def resolve(
        self, node_id: str, *, purpose: Literal["generation", "audit"]
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        node_id = _node_id(node_id)
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("unknown reference purpose")  # noqa: TRY003
        async with self._runtime._scope_operation(self.scope_id), self._storage().transaction() as connection:
            node = (await self._repository.load(connection, self.scope_id)).get(node_id)
            if node is None or (purpose == "generation" and not node.available):
                raise SourceLifecycleError("node is unknown or unavailable for generation")  # noqa: TRY003
            return node.reference


class SourceLifecycleApplication:
    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, self._database, scope_id)


__all__ = ["ScopedSourceLifecycle", "SourceLifecycleApplication", "SourceLifecycleError", "SourceLifecycleEvent"]
