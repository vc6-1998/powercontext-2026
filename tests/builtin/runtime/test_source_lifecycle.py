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

"""Lifecycle persistence failures, isolation, and pre-limit retrieval filtering."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import EmbeddingProfile, MemoryEntryInput
from powercontext.builtin.inference import EmbeddingResult
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.records import ArtifactWrite
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    SearchTopicMemoryRequest,
    SourceLifecycleError,
    SourceLifecycleEvent,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import SourceRef


async def _graph(runtime, key="graph"):
    assert runtime.scopes is not None
    scope = await runtime.scopes.create(ScopeDraft(title=key, summary="Lifecycle test", idempotency_key=key))
    sid = scope.scope_id
    service = runtime.lifecycle.for_scope(sid)
    for node_id in ("left", "right"):
        receipt = await runtime.sources.for_scope(sid).capture(
            CaptureSource(source_id=node_id, content=node_id, metadata={})
        )
        await service.register_source(node_id, receipt.source_ref, revision=1)
    citations = {}
    for policy in ("all", "any"):
        created = await runtime.memory.for_scope(sid).remember(
            RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=f"{policy} supported evidence"),))
        )
        assert created.entry is not None
        citations[policy] = created.entry.citation
        await service.register_memory(policy, created.entry.citation, parents=("left", "right"), policy=policy)
    return sid, service, citations


def _event(*, sequence=1, available=False, reason="Withdrawal"):
    return SourceLifecycleEvent(node_id="left", revision=1, sequence=sequence, available=available, reason=reason)


def test_real_memory_write_failure_replays_after_restart_without_extra_revisions(tmp_path) -> None:
    async def scenario() -> None:
        path = tmp_path / "lifecycle-failure.db"
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{path}"))
        withdrawal = _event(reason="海洋" * 400)
        async with open_builtin_runtime(config) as runtime:
            sid, lifecycle, citations = await _graph(runtime)
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "CREATE TRIGGER stop_memory BEFORE INSERT ON pc_artifacts "
                    "WHEN NEW.family = 'memory' BEGIN SELECT RAISE(ABORT, 'controlled lifecycle failure'); END"
                )
            with pytest.raises(IntegrityError):
                await lifecycle.apply(withdrawal)
        with sqlite3.connect(path) as connection:
            connection.execute("DROP TRIGGER stop_memory")
        async with open_builtin_runtime(config) as runtime:
            lifecycle = runtime.lifecycle.for_scope(sid)
            assert await lifecycle.apply(withdrawal) == ["any", "right"]
            memory = runtime.memory.for_scope(sid)
            page = await memory.list(include_inactive=True)
            assert {entry.entry.text: entry.state for entry in page.entries} == {
                "all supported evidence": "inactive",
                "any supported evidence": "active",
            }
            assert await lifecycle.apply(withdrawal) == ["any", "right"]
            assert (await memory.list(include_inactive=True)).memory_ref == page.memory_ref
            for node_id, citation in citations.items():
                assert await lifecycle.resolve(node_id, purpose="audit") == citation
                assert (
                    await memory.get(GetMemoryEntryRequest(citation=citation))
                ).entry.entry_version_id == citation.entry_version_id
            await lifecycle.apply(_event(sequence=2, available=True))
            restored = await memory.list()
            assert len(restored.entries) == 2
            assert {entry.entry.entry_version_id for entry in restored.entries} == {
                ref.entry_version_id for ref in citations.values()
            }

    asyncio.run(scenario())


def test_scope_isolation_registration_replay_and_invalid_references() -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig())) as runtime:
            left_sid, left, citations = await _graph(runtime, "left-scope")
            right_sid, right, right_citations = await _graph(runtime, "right-scope")
            await left.apply(_event())
            assert len((await runtime.memory.for_scope(right_sid).list()).entries) == 2
            assert await right.resolve("left", purpose="generation") == SourceRef(
                source_type="content", source_id="left"
            )
            assert await right.resolve("all", purpose="generation") == right_citations["all"]
            before = (await runtime.memory.for_scope(left_sid).list()).memory_ref
            await left.register_memory("all", citations["all"], parents=("right", "left"), policy="all")
            assert (await runtime.memory.for_scope(left_sid).list()).memory_ref == before
            with pytest.raises(SourceLifecycleError):
                await left.register_source("missing", SourceRef(source_type="content", source_id="missing"), revision=1)
            for parents in (("left", "left"), ("unknown",)):
                with pytest.raises(SourceLifecycleError):
                    await left.register_memory("invalid", citations["all"], parents=parents, policy="all")
            with pytest.raises(SourceLifecycleError):
                await left.register_topic(
                    "missing-topic",
                    ArtifactRef(family="topic-memory", artifact_id="missing", revision=1),
                    parents=("left",),
                    policy="all",
                )
            with pytest.raises(SourceLifecycleError):
                await left.apply(
                    SourceLifecycleEvent(node_id="all", revision=1, sequence=1, available=False, reason="not a Source")
                )
            with pytest.raises(SourceLifecycleError):
                await left.resolve("all", purpose="unknown")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"node_id": " "},
        {"node_id": 1},
        {"reason": "\t"},
        {"revision": True},
        {"revision": "1"},
        {"sequence": 1.0},
        {"sequence": -1},
        {"available": 1},
        {"extra": True},
    ],
)
def test_event_types_and_identity_are_strict(changes) -> None:
    payload = {**_event().model_dump(), **changes}
    with pytest.raises(ValidationError):
        SourceLifecycleEvent.model_validate(payload)


def test_withdrawn_topics_do_not_fill_fts_or_vector_candidate_budgets() -> None:
    class Embedding:
        profile = EmbeddingProfile(profile_id="lifecycle-topic-vector", model="offline", dimension=2)

        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            return EmbeddingResult(vectors=tuple((1.0, 0.0) for _ in texts))

    async def scenario() -> None:
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig()), embedding_model=Embedding()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Topics", summary="Candidate budgets", idempotency_key="topic-budget")
            )
            sid = scope.scope_id
            lifecycle = runtime.lifecycle.for_scope(sid)
            receipt = await runtime.sources.for_scope(sid).capture(
                CaptureSource(source_id="left", content="origin", metadata={})
            )
            await lifecycle.register_source("left", receipt.source_ref, revision=1)
            references = []
            for index in range(36):
                created = await runtime.records.for_scope(sid).create_artifact(
                    "topic-memory",
                    ArtifactWrite(
                        content={"title": "needle", "summary": "needle evidence", "detail": "needle evidence"}
                    ),
                )
                ref = ArtifactRef(family="topic-memory", artifact_id=created.artifact_id, revision=created.revision)
                references.append(ref)
                if index < 35:
                    await lifecycle.register_topic(f"blocked-{index}", ref, parents=("left",), policy="all")
            await lifecycle.apply(_event())
            for allow_embedding in (False, True):
                result = await runtime.topic_memory.for_scope(sid).search(
                    SearchTopicMemoryRequest(query="needle", limit=1), allow_embedding=allow_embedding
                )
                assert [hit.artifact_ref for hit in result.hits] == references[-1:]
            visible = await runtime.topic_memory.for_scope(sid).browse(limit=100)
            assert [row.artifact_ref for row in visible] == references[-1:]
            historical = await runtime.records.for_scope(sid).get_artifact_revision(
                "topic-memory", references[0].artifact_id, references[0].revision
            )
            assert historical.content["detail"] == "needle evidence"

    asyncio.run(scenario())
