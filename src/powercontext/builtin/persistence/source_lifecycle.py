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

"""Persistent evidence dependency bindings and Topic retrieval qualification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql import ColumnElement

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.codec import dump_model, load_model, stored_bytes
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.sources import SourceRef


class LifecycleNode(BaseModel):
    """Original reference, immutable dependency binding, and last accepted event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    kind: Literal["source", "memory", "topic"]
    reference: SourceRef | MemoryCitation | ArtifactRef
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] = "all"
    revision: int | None = None
    available: bool = True
    last_event: dict[str, JsonValue] | None = None


class SourceLifecycleRepository:
    async def load(self, connection: AsyncConnection, scope_id: str) -> dict[str, LifecycleNode]:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        rows = (await connection.execute(select(table).where(table.c.scope_id == scope_id))).mappings()
        return {
            str(row["node_id"]): load_model(
                LifecycleNode,
                stored_bytes(row["payload"], column="payload"),
                kind="source-lifecycle",
                name=str(row["node_id"]),
            )
            for row in rows
        }

    async def save(
        self, connection: AsyncConnection, scope_id: str, node: LifecycleNode, *, create: bool = False
    ) -> None:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        reference = node.reference
        values = {
            "kind": node.kind,
            "available": node.available,
            "artifact_id": reference.artifact_id if isinstance(reference, ArtifactRef) else None,
            "artifact_revision": reference.revision if isinstance(reference, ArtifactRef) else None,
            "payload": dump_model(node, kind="source-lifecycle", name=node.node_id),
        }
        if create:
            await connection.execute(insert(table).values(scope_id=scope_id, node_id=node.node_id, **values))
        else:
            await connection.execute(
                update(table).where(table.c.scope_id == scope_id, table.c.node_id == node.node_id).values(**values)
            )


def topic_lifecycle_predicate(scope_id: Any, artifact_id: Any, revision: Any) -> ColumnElement[bool]:
    """Exclude unsupported Topic identities before candidate limits are applied."""

    table = SOURCE_LIFECYCLE_NODES_TABLE
    return (
        ~select(table.c.node_id)
        .where(
            table.c.scope_id == scope_id,
            table.c.kind == "topic",
            table.c.artifact_id == artifact_id,
            table.c.artifact_revision == revision,
            table.c.available.is_(False),
        )
        .correlate_except(table)
        .exists()
    )


def topic_lifecycle_sql(alias: str) -> str:
    """Equivalent predicate for index SQL with a caller-owned schema alias."""

    return f"""NOT EXISTS (
        SELECT 1 FROM pc_source_lifecycle_nodes AS lifecycle
        WHERE lifecycle.scope_id = {alias}.scope_id
          AND lifecycle.kind = 'topic'
          AND lifecycle.artifact_id = {alias}.artifact_id
          AND lifecycle.artifact_revision = {alias}.revision
          AND lifecycle.available = 0
    )"""  # noqa: S608 - Aliases are static schema identifiers in adapter SQL.
