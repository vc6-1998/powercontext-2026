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

"""Count admission must preserve durable invocation and transaction boundaries."""

from __future__ import annotations

import asyncio

from sqlalchemy import update

from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.processing_intents import ArtifactProcessingIntentRepository
from powercontext.builtin.persistence.sources import SourceRepository
from powercontext.builtin.persistence.tables import SOURCES_TABLE
from powercontext.builtin.runtime import artifact_processing as processing
from powercontext.builtin.runtime.artifact_processing import ArtifactProcessingBinding, ArtifactProcessingSupervisor
from powercontext.builtin.runtime.processing_discovery import source_count_ready_scopes
from powercontext.builtin.sources import BUILTIN_SOURCE_REGISTRY, SourceCursor
from powercontext.builtin.sources.content import ContentSource
from powercontext.sources import SourceMaterialization
from tests.builtin.runtime.test_processing_scheduler import _intent, _Launcher, _profile, _request, _wait


class _AdmissionInterrupted(Exception):
    pass


class _InterruptedIntents(ArtifactProcessingIntentRepository):
    interrupted = False

    async def request(self, connection, scope_id, binding_name, /):
        row = await super().request(connection, scope_id, binding_name)
        if scope_id == "d" and not self.interrupted:
            self.interrupted = True
            raise _AdmissionInterrupted
        return row


def test_count_admission_resumes_after_a_rolled_back_page(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(processing, "_DISCOVERY_PAGE_SIZE", 2)

    async def scenario() -> None:
        async with _profile(tmp_path, "count-rollback") as profile:
            for scope in ("a", "b", "c", "d", "e"):
                await _request(profile.database, "memory-binding", scope, requested=False)

            async def eligible(_connection, scopes):
                return frozenset(scopes) & {"c", "d", "e"}

            launcher = _Launcher(profile.database)
            interrupted = _InterruptedIntents()
            async with ArtifactProcessingSupervisor(
                database=profile.database,
                bindings=(
                    ArtifactProcessingBinding("memory-binding", "memory", launcher, automatic_count_filter=eligible),
                ),
                lease_mode="single-process",
                intents=interrupted,
                oceanbase_tick_seconds=0.01,
            ) as supervisor:
                await _wait(lambda: supervisor.family_status["memory"]["completed"] == 3)
                assert interrupted.interrupted
                assert sorted(work.scope_id for work in launcher.assignments) == ["c", "d", "e"]
                for scope in ("a", "b"):
                    row = await _intent(profile.database, "memory-binding", scope)
                    assert row.requested_generation == row.handled_generation == 0
                for scope in ("c", "d", "e"):
                    row = await _intent(profile.database, "memory-binding", scope)
                    assert row.requested_generation == row.handled_generation == 1

    asyncio.run(scenario())


def test_accepted_retry_does_not_require_another_positive_count_check(tmp_path) -> None:
    async def scenario() -> None:
        async with _profile(tmp_path, "count-accepted-retry") as profile:
            await _request(profile.database, "memory-binding", "scope", requested=False)
            eligible = True
            attempts = 0

            async def qualify(_connection, scopes):
                return frozenset(scopes) if eligible else frozenset()

            async def work(_assignment):
                nonlocal attempts, eligible
                attempts += 1
                if attempts == 1:
                    eligible = False
                    raise _AdmissionInterrupted

            launcher = _Launcher(profile.database, on_work=work)
            async with ArtifactProcessingSupervisor(
                database=profile.database,
                bindings=(
                    ArtifactProcessingBinding("memory-binding", "memory", launcher, automatic_count_filter=qualify),
                ),
                lease_mode="single-process",
                retry_base_seconds=0.02,
                retry_cap_seconds=0.02,
                retry_jitter=lambda: 1,
            ) as supervisor:
                await _wait(lambda: supervisor.family_status["memory"]["completed"] == 1)
                assert supervisor.family_status["memory"]["failed"] == 1
                assert attempts == 2
                assert len({work.claimed_request_generation for work in launcher.assignments}) == 1
                row = await _intent(profile.database, "memory-binding", "scope")
                assert row.requested_generation == row.handled_generation == 1

    asyncio.run(scenario())


def test_journal_counts_use_rows_after_each_family_cursor_not_position_distance(tmp_path) -> None:
    async def scenario() -> None:
        async with _profile(tmp_path, "count-rows") as profile:
            repository = SourceRepository(BUILTIN_SOURCE_REGISTRY)
            async with profile.database.transaction() as connection:
                for scope, size in (("left", 3), ("right", 1)):
                    for index in range(size):
                        source = ContentSource(
                            name=f"s{index}",
                            materialization=SourceMaterialization.CAPTURED,
                            content="Captured evidence",
                        )
                        await repository.add(connection, scope, source)
                        await repository.add(connection, scope, source)
                # A sparse position range still contains only three identities.
                await connection.execute(
                    update(SOURCES_TABLE)
                    .where(SOURCES_TABLE.c.scope_id == "left", SOURCES_TABLE.c.source_id == "s2")
                    .values(journal_position=100)
                )
                await SourceCursorRepository().save(
                    connection, "left", "memory-binding", SourceCursor(sequence=2), expected_generation=None
                )
                assert (
                    await source_count_ready_scopes(
                        connection, ("left", "right"), binding_name="memory-binding", threshold=2
                    )
                    == frozenset()
                )
                assert await source_count_ready_scopes(
                    connection, ("left", "right"), binding_name="topic-binding", threshold=3
                ) == {"left"}

    asyncio.run(scenario())
