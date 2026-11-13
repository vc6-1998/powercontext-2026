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

"""Opt-in native comparison with the frozen two-query Memory index path."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from time import perf_counter
from types import MethodType

import pytest
from sqlalchemy import event

from powercontext.builtin.artifacts.memory import EmbeddingProfile, MemoryEntryInput
from powercontext.builtin.inference import EmbeddingResult
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex
from powercontext.builtin.persistence.seekdb import SeekDBConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts
from powercontext.builtin.tags import MemoryEntryTagTarget, TagFilter

pytestmark = pytest.mark.skipif(os.environ.get("POWERCONTEXT_TEST_SEEKDB") != "1", reason="requires native seekdb")


class _Embedding:
    profile = EmbeddingProfile(profile_id="native-hybrid-comparison", model="offline", dimension=384)

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        vectors = []
        for value in texts:
            raw = hashlib.shake_256(value.encode()).digest(768)
            vector = tuple(int.from_bytes(raw[start : start + 2], "little") / 32768 - 1 for start in range(0, 768, 2))
            norm = math.sqrt(sum(item * item for item in vector))
            vectors.append(tuple(item / norm for item in vector))
        return EmbeddingResult(vectors=tuple(vectors))


def _identities(result):
    return tuple((hit.memory_ref, hit.entry_id, hit.entry_version_id, hit.text, hit.matched_by) for hit in result.hits)


def test_native_hybrid_matches_frozen_results_and_candidate_statement_budget(tmp_path, monkeypatch) -> None:  # noqa: C901 - one native fixture lifecycle
    async def scenario() -> None:  # noqa: C901 - alternating paths share one native database
        config = BuiltinConfig(database=SeekDBConfig(path=tmp_path / "seekdb"))
        async with open_builtin_contexts(config, embedding_model=_Embedding()) as contexts:
            services = []
            for index in range(3):
                service = (await contexts.get(f"native-scope-{index}")).artifacts.memory
                head = await service.remember(
                    memory=None,
                    entries=tuple(
                        MemoryEntryInput(kind="fact", text=f"Native topic {number % 20} record {number} evidence.")
                        for number in range(172)
                    ),
                    mode="append",
                )
                assert head is not None
                current = await service.remember(
                    memory=head,
                    entries=(MemoryEntryInput(kind="fact", text="Newest topic 19 record 172 evidence."),),
                    mode="append",
                )
                assert current is not None
                services.append((service, current, f"native-scope-{index}"))
            native_search = contexts.index.search
            frozen_search = MethodType(CompositeMemoryIndex.search, contexts.index)
            statements = []

            def record(_connection, _cursor, statement, parameters, _context, _executemany):
                if "AGAINST" in statement.upper() or "l2_distance" in statement.lower():
                    statements.append((statement, parameters))

            async def search(service, head, query, *, frozen=False, tag_filter=None):
                start = len(statements)
                clock = perf_counter()
                with monkeypatch.context() as patch:
                    patch.setattr(contexts.index, "search", frozen_search if frozen else native_search)
                    result = await service.search(
                        query, memories=(head,), mode="hybrid", limit=10, tag_filter=tag_filter
                    )
                elapsed = (perf_counter() - clock) * 1000
                assert len(statements) - start == (2 if frozen else 1)
                return result, elapsed

            event.listen(contexts.database.engine.sync_engine, "before_cursor_execute", record)
            try:
                await search(services[0][0], services[0][1], "topic 0")
                await search(services[0][0], services[0][1], "topic 0", frozen=True)
                native_times, frozen_times = [], []
                compared = 0
                for index, (service, head, _scope) in enumerate(services):
                    for number in range(20):
                        query = f"topic {number}"
                        if (index + number) % 2:
                            baseline, old_time = await search(service, head, query, frozen=True)
                            actual, new_time = await search(service, head, query)
                        else:
                            actual, new_time = await search(service, head, query)
                            baseline, old_time = await search(service, head, query, frozen=True)
                        assert _identities(actual) == _identities(baseline)
                        assert len({hit.entry_id for hit in actual.hits}) == len(actual.hits)
                        assert all(hit.memory_ref == head.as_ref() for hit in actual.hits)
                        native_times.append(new_time)
                        frozen_times.append(old_time)
                        compared += 1
                service, head, scope = services[0]
                entries = await service.entries(head)
                for index, entry in enumerate(entries[:12]):
                    target = MemoryEntryTagTarget(artifact_id=head.artifact_id, entry_id=entry.entry_id)
                    existing = await contexts.records.get_tags(scope, target)
                    tags = ("selected", "even") if index % 2 == 0 else ("selected",)
                    await contexts.records.replace_tags(scope, target, tags, expected_etag=existing.etag)
                for match in ("all", "any"):
                    tags = TagFilter(tags=("selected", "even"), match=match)
                    actual, _ = await search(service, head, "topic", tag_filter=tags)
                    baseline, _ = await search(service, head, "topic", frozen=True, tag_filter=tags)
                    assert _identities(actual) == _identities(baseline)
                statement, parameters = next(item for item in statements if "UNION ALL" in item[0])
                async with contexts.database.transaction() as connection:
                    plan = "\n".join(
                        str(row[0])
                        for row in (await connection.exec_driver_sql(f"EXPLAIN {statement}", parameters)).all()
                    )
                assert "TEXT RETRIEVAL SCAN" in plan and "VECTOR INDEX" in plan
                report = {
                    "measurement": "local synthetic native comparison; no official semantic quality annotations",
                    "scopes": 3,
                    "entries_per_scope": 173,
                    "dimension": 384,
                    "queries": compared,
                    "ordered_identity_text_and_channel_agreement": compared,
                    "native_candidate_statements_per_search": 1,
                    "frozen_candidate_statements_per_search": 2,
                    "native_p50_ms": statistics.median(native_times),
                    "frozen_p50_ms": statistics.median(frozen_times),
                    "tag_filter_all_and_any": "matched frozen results",
                    "explain": plan,
                }
                directory = os.environ.get("POWERCONTEXT_HYBRID_REPORT")
                if directory:
                    await asyncio.to_thread(Path(directory).write_text, json.dumps(report, indent=2))
                assert report["native_p50_ms"] <= 500
            finally:
                event.remove(contexts.database.engine.sync_engine, "before_cursor_execute", record)

    asyncio.run(scenario())
