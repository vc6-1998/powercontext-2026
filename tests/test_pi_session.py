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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from competition.selftest_support.runtime_fixture import RuntimeCheckpoints, RuntimeSink, run_connector_persisted
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts, open_builtin_runtime
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import ConnectorBinding, SourceDefinitionRegistry, SourceRef, manifest_for_definition
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.observations import SourceObservation
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiEventInput,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def archive(records: list[dict[str, Any]], *, session_id: str = "session", sorted_keys: bool = False) -> bytes:
    header = {"type": "session", "version": 3, "id": session_id}
    return (
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=sorted_keys) for row in [header, *records]) + "\n"
    ).encode()


def message(event_id: str, parent_id: str | None, text: str) -> dict[str, Any]:
    return {"type": "message", "id": event_id, "parentId": parent_id, "message": {"role": "user", "content": text}}


@pytest.mark.parametrize(
    "tail",
    [
        {"type": "model_change", "id": "missing-parent"},
        {"type": "model_change", "id": "forward", "parentId": "later"},
        {"type": "model_change", "id": "root", "parentId": None},
        {"type": "model_change", "id": [], "parentId": None},
    ],
)
def test_unselected_invalid_nodes_still_reject_the_complete_archive(tail: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=archive([message("root", None, "safe"), tail]), leaf_id="root"))


def test_header_only_explicit_missing_leaf_and_non_json_numbers() -> None:
    assert parse_pi_session(PiSessionInput(content=archive([]))) == ()
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=archive([]), leaf_id="missing"))
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=archive([message("root", None, "safe")]) + b'{"x":NaN}\n'))


async def test_unicode_line_separators_tool_arguments_and_complete_original_record() -> None:
    text = " before\u2028middle\u2029after "
    record = message("root", None, text)
    record["message"]["content"] = [
        {"type": "text", "text": text},
        {"type": "thinking", "thinking": "retained only in the original record"},
        {"type": "image", "data": "image bytes"},
        {"type": "toolCall", "id": "call", "name": "read", "arguments": {"z": "中文", "a": [1, True]}},
    ]
    (event,) = parse_pi_session(PiSessionInput(content=archive([record])))
    assert event.content == "user: " + text + '\n[tool_call read call] {"a":[1,true],"z":"中文"}'
    assert event.record == record
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    source = await registry.resolve(event)
    assert await registry.read(source) == event
    projection = registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY)
    assert isinstance(projection, dict) and projection["content"] == event.content


async def test_event_identity_is_bounded_and_does_not_depend_on_delimiters_or_payload() -> None:
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    sources = []
    for session_id, event_id in [("a:b", "c"), ("a", "b:c"), ("中文" * 200, "node/" * 200)]:
        (event,) = parse_pi_session(
            PiSessionInput(content=archive([message(event_id, None, "text")], session_id=session_id))
        )
        sources.append(await registry.resolve(event))
    assert len({source.name for source in sources}) == 3
    assert all(len(source.name) <= 256 for source in sources)
    (changed,) = parse_pi_session(PiSessionInput(content=archive([message("c", None, "edited")], session_id="a:b")))
    assert (await registry.resolve(changed)).name == sources[0].name


async def test_off_branch_rewrite_is_rejected_before_new_branch_import(tmp_path: Path) -> None:
    path = tmp_path / "branches.jsonl"
    records = [message("root", None, "shared"), message("left", "root", "left"), message("right", "root", "right")]
    path.write_bytes(archive(records))
    binding = ConnectorBinding(
        scope_id="scope", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )
    first, _ = await run_connector_persisted(
        tmp_path, PI_EVENT_SOURCE_DEFINITION, PiSessionConnector(path, "left"), binding
    )
    records[1]["message"]["content"] = "rewritten off-branch"
    path.write_bytes(archive(records))
    with pytest.raises(ValueError):
        await run_connector_persisted(tmp_path, PI_EVENT_SOURCE_DEFINITION, PiSessionConnector(path, "right"), binding)
    records[1]["message"]["content"] = "left"
    path.write_bytes(archive(records, sorted_keys=True))
    result, entries = await run_connector_persisted(
        tmp_path, PI_EVENT_SOURCE_DEFINITION, PiSessionConnector(path, "right"), binding
    )
    assert result.previous_checkpoint == first.committed_checkpoint
    assert len(result.items) == 1 and len(entries) == 3


async def test_partial_failure_keeps_checkpoint_and_full_large_evidence_then_restart_resumes(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    large_output = "完整工具结果\n" * 10000
    records: list[dict[str, Any]] = [
        message("first", None, "original"),
        {
            "type": "message",
            "id": "second",
            "parentId": "first",
            "message": {"role": "toolResult", "toolName": "read", "toolCallId": "call", "content": large_output},
        },
    ]
    path.write_bytes(archive(records))
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'failure.db'}"))
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))

    class FailSecond:
        def __init__(self, sink: RuntimeSink) -> None:
            self.sink = sink

        async def submit(
            self, binding: ConnectorBinding, item_id: str, definition_name: str, value: object, /
        ) -> SourceRef:
            assert isinstance(value, PiEventInput)
            if value.event_id == "second":
                raise RuntimeError("injected persistence failure")  # noqa: TRY003
            return await self.sink.submit(binding, item_id, definition_name, value)

    async with open_builtin_runtime(config) as runtime:
        assert runtime.scopes is not None
        scope = await runtime.scopes.create(ScopeDraft(title="Pi", summary="Failure recovery", idempotency_key="pi"))
        binding = ConnectorBinding(
            scope_id=scope.scope_id, binding_id="pi", connector_name="pi-session-history", connector_version="1"
        )
        await runtime.ingestion.register(manifest_for_definition(PI_EVENT_SOURCE_DEFINITION))
        checkpoints = RuntimeCheckpoints(runtime)
        failing = ConnectorLifecycle(sink=FailSecond(RuntimeSink(runtime, registry)), checkpoints=checkpoints)
        with pytest.raises(RuntimeError, match="injected"):
            await failing.run(PiSessionConnector(path), binding)
        assert await checkpoints.load(binding) is None
        lifecycle = ConnectorLifecycle(sink=RuntimeSink(runtime, registry), checkpoints=checkpoints)
        records[0]["message"]["content"] = "changed after partial write"
        path.write_bytes(archive(records))
        with pytest.raises(ValueError, match="conflicts"):
            await lifecycle.run(PiSessionConnector(path), binding)
        assert await checkpoints.load(binding) is None
        records[0]["message"]["content"] = "original"
        path.write_bytes(archive(records))
        completed = await lifecycle.run(PiSessionConnector(path), binding)
        assert completed.committed_checkpoint is not None
    async with open_builtin_runtime(config) as runtime:
        replay = await ConnectorLifecycle(
            sink=RuntimeSink(runtime, registry), checkpoints=RuntimeCheckpoints(runtime)
        ).run(PiSessionConnector(path), binding)
        assert replay.items == ()
        assert replay.previous_checkpoint == completed.committed_checkpoint
    async with open_builtin_contexts(config) as contexts:
        entries = await (await contexts.get(binding.scope_id)).sources.journal.entries()
        assert len(entries) == 2
        stored: dict[str, SourceObservation] = {}
        for entry in entries:
            assert isinstance(entry.source, SourceObservation)
            event_id = entry.source.payload["event_id"]
            assert isinstance(event_id, str)
            stored[event_id] = entry.source
        assert stored["first"].payload["record"] == records[0]
        assert stored["second"].payload["record"] == records[1]
        evidence = stored["second"].projection(TEXT_EVIDENCE_PROJECTION_KEY)
        assert isinstance(evidence, dict)
        assert evidence["content"] == "toolResult: [tool_result read call] " + large_output
