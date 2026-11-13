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

"""Independent wire, UTF-8 boundary, and empty-context regression tests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import subprocess
import sys

import pytest

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    ContextAssembly,
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    PrepareContextRequest,
    apply_context_pack_delta,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource


@pytest.mark.parametrize("chunk_bytes", range(1, 10))
def test_multibyte_boundaries_preserve_exact_original_bytes(chunk_bytes: int) -> None:
    for content in ("", "a\r\nb \t", "海洋", "a😀é\n海洋\r\n"):
        if any(len(character.encode()) > chunk_bytes for character in content):
            with pytest.raises(ContextPackError):
                ContextPack.from_content(content, chunk_bytes=chunk_bytes)
            continue
        pack = ContextPack.from_content(content, chunk_bytes=chunk_bytes)
        assert b"".join(pack.items) == content.encode()
        assert all(len(item) <= chunk_bytes and item.decode() for item in pack.items)
        assert ContextPack.from_content(content, chunk_bytes=chunk_bytes) == pack
        assert ContextPack.from_bytes(pack.to_bytes()) == pack


def test_pack_byte_count_at_decimal_boundaries_matches_independent_digest() -> None:
    for size in (0, 1, 597, 598, 599, 600, 7347, 7348, 7349, 7350):
        pack = ContextPack.build((b"a" * size, b""))
        payload = json.loads(pack.to_bytes())
        unsigned = {"schema": payload["schema"], "items": payload["items"]}
        digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert payload["digest"] == "sha256:" + digest
        assert payload["byte_count"] == pack.byte_count == len(pack.to_bytes())


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "future"),
        ("digest", "sha256:" + "0" * 64),
        ("digest", None),
        ("byte_count", True),
        ("byte_count", 200.0),
        ("byte_count", -1),
        ("items", "YWJj"),
        ("items", [None]),
        ("items", ["@@@"]),
        ("items", ["Yh=="]),
        ("items", ["YQ==="]),
        ("items", ["/w=="]),
        ("extra", True),
    ],
)
def test_pack_rejects_invalid_wire_fields(field: str, value: object) -> None:
    payload = json.loads(ContextPack.build((b"abc",)).to_bytes())
    payload[field] = value
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps(payload).encode())


@pytest.mark.parametrize("data", [b"null", b"[]", b"{", b"\xff", b'{"schema":1,"schema":2}'])
def test_wire_json_errors_are_context_pack_errors(data: bytes) -> None:
    for decoder in (ContextPack.from_bytes, ContextPackDelta.from_bytes):
        with pytest.raises(ContextPackError):
            decoder(data)


def test_delta_missing_extra_and_invalid_fields_are_rejected() -> None:
    base = ContextPack.from_content("海洋" * 100)
    delta = ContextPackDelta.between(base, ContextPack.from_content("海洋" * 101))
    payload = json.loads(delta.to_bytes())
    for field in payload:
        with pytest.raises(ContextPackError):
            ContextPackDelta.from_bytes(
                json.dumps({key: value for key, value in payload.items() if key != field}).encode()
            )
    for changes in (
        {"extra": 1},
        {"schema": "future"},
        {"base_digest": None},
        {"target_digest": "bad"},
        {"prefix_items": "1"},
        {"prefix_items": None},
        {"appended": ["/w=="]},
        {"appended": ["%%"]},
    ):
        with pytest.raises(ContextPackError):
            ContextPackDelta.from_bytes(json.dumps({**payload, **changes}).encode())


@pytest.mark.parametrize("chunk_bytes", [0, -1, True, 1.5])
def test_invalid_chunk_limit_is_rejected(chunk_bytes) -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("text", chunk_bytes=chunk_bytes)


def test_serialized_delta_reconstructs_in_a_fresh_python_process() -> None:
    base = ContextPack.from_content("Original 数据库 context\n" * 100)
    target = ContextPack.from_content("Original 数据库 context\n" * 80 + "Changed 😀 tail\n")
    wire = json.dumps([
        base64.b64encode(base.to_bytes()).decode(),
        base64.b64encode(ContextPackDelta.between(base, target).to_bytes()).decode(),
    ]).encode()
    script = """
import base64, json, sys
from powercontext.builtin.runtime.context_pack import ContextPack, ContextPackDelta, apply_context_pack_delta
base, delta = (base64.b64decode(value) for value in json.loads(sys.stdin.buffer.read()))
result = apply_context_pack_delta(ContextPack.from_bytes(base), ContextPackDelta.from_bytes(delta))
sys.stdout.buffer.write(b''.join(result.items))
"""
    result = subprocess.run([sys.executable, "-c", script], input=wire, capture_output=True, check=True)
    assert result.stdout == b"".join(target.items)


class _Pipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="fact", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


def test_native_empty_ready_and_empty_again_preserve_authorization() -> None:
    async def scenario() -> None:
        authorized: list[tuple[str, ...]] = []

        async def authorize(scope_ids: tuple[str, ...]) -> None:
            authorized.append(scope_ids)

        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_Pipeline()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Pack", summary="Pack test", idempotency_key="pack-empty")
            )
            context = runtime.context.for_scope(scope.scope_id)
            request = PrepareContextRequest(query="database")
            first = await context.prepare_incremental(request, authorize_scopes=authorize)
            assert isinstance(first, ContextPack) and first.items == ()
            await runtime.sources.for_scope(scope.scope_id).capture(
                CaptureSource(source_id="database-source", content="The database runs in Ubuntu.", metadata={})
            )
            while (await runtime.memory.for_scope(scope.scope_id).flush()).processed:
                pass
            second = await context.prepare_incremental(request, base_pack=first, authorize_scopes=authorize)
            assert isinstance(second, ContextPackDelta)
            ready = apply_context_pack_delta(first, ContextPackDelta.from_bytes(second.to_bytes()))
            assert b"".join(ready.items).decode() == (await context.prepare(request)).content
            assert ready.items
            empty_request = PrepareContextRequest(query="database", assembly=ContextAssembly(sections=()))
            third = await context.prepare_incremental(empty_request, base_pack=ready)
            assert isinstance(third, ContextPackDelta)
            assert apply_context_pack_delta(ready, third).items == ()
            assert authorized == [(scope.scope_id,), (scope.scope_id,)]

    asyncio.run(scenario())
