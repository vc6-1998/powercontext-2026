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

"""Literal, scoped evidence, complete wire budgets, and ordinary-mode compatibility."""

from __future__ import annotations

import json

import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryHit
from powercontext.builtin.runtime import PrepareContextRequest
from powercontext.builtin.runtime.prepared_context import PreparedContextBuilder, PreparedMemoryCandidates
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY


def _group(scope: str, text: str) -> PreparedMemoryCandidates:
    memory = ArtifactRef(family="memory", artifact_id="shared-memory", revision=3)
    hit = MemoryHit(
        memory_ref=memory,
        entry_id="same-entry",
        entry_version_id="same-version",
        text=text,
        score=1,
        matched_by=("fts",),
    )
    return PreparedMemoryCandidates(scope_id=scope, memory_ref=memory, hits=(hit,))


def _build(query: str, groups, *, max_bytes=1800, selection=True):
    return PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query=query, max_bytes=max_bytes, evidence_selection=selection),
        current_scope_id="left",
        memory_candidates=groups,
    )


def _check(build, originals, max_bytes):
    content = build.context.content or ""
    assert build.context.content_bytes == len(content.encode()) <= max_bytes
    assert len(build.fragments) == len(build.origins)
    if not build.fragments:
        assert not content
        return
    assert content.startswith(TRUST_POLICY)
    assert "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1" in content
    assert content.endswith("END_POWERCONTEXT_PREPARED_CONTEXT_V1")
    payload = json.loads(content.splitlines()[-2])
    assert payload["trust"] == "untrusted_history"
    for fragment, origin, item in zip(build.fragments, build.origins, payload["items"], strict=True):
        text = originals[fragment.address.memory.scope_id]
        assert type(fragment.start) is type(fragment.end) is int
        assert 0 <= fragment.start < fragment.end <= len(text)
        assert fragment.text == text[fragment.start : fragment.end]
        assert origin == fragment.address
        assert item == {
            "citation": {
                "scope_id": fragment.address.memory.scope_id,
                "memory_ref": fragment.address.memory.artifact.model_dump(mode="json"),
                "entry_id": fragment.address.entry_id,
                "entry_version_id": fragment.address.entry_version_id,
            },
            "content": fragment.text,
            "start": fragment.start,
            "end": fragment.end,
        }


def test_equal_text_and_entry_ids_in_different_scopes_keep_distinct_addresses() -> None:
    text = "The cobalt sensor is calibrated at 37 degrees."
    result = _build("cobalt sensor calibration", (_group("left", text), _group("right", text)), max_bytes=2000)
    _check(result, {"left": text, "right": text}, 2000)
    assert {fragment.address.memory.scope_id for fragment in result.fragments} == {"left", "right"}


@pytest.mark.parametrize("max_bytes", [1400, 1800])
def test_separated_complete_evidence_and_adjacent_sentences_retain_literal_intervals(max_bytes: int) -> None:
    first = "The cobalt sensor is calibrated at 37 degrees. "
    second = "The amber valve opens at 19 degrees. "
    text = first + "General administrative notes. " * 100 + second
    result = _build("cobalt sensor calibrated amber valve opens", (_group("left", text),), max_bytes=max_bytes)
    _check(result, {"left": text}, max_bytes)
    assert any(fragment.start == 0 and fragment.end >= len(first) for fragment in result.fragments)
    assert any(fragment.start <= text.index(second) and fragment.end == len(text) for fragment in result.fragments)
    assert len(result.fragments) >= 2


def test_long_unpunctuated_sentence_selects_query_window_instead_of_prefix() -> None:
    text = "generic notes " * 250 + "cobalt sensor calibrated at 37 degrees " + "neutral details " * 150
    result = _build("cobalt sensor calibrated", (_group("left", text),), max_bytes=1200)
    _check(result, {"left": text}, 1200)
    assert result.fragments
    assert any("cobalt sensor calibrated at 37 degrees" in fragment.text for fragment in result.fragments)
    assert result.fragments[0].start > 2000


@pytest.mark.parametrize("max_bytes", [512, 700, 1200, 1800])
def test_unicode_and_json_escaping_are_charged_in_the_complete_context(max_bytes: int) -> None:
    text = '海洋观测温度记录,中文没有空格。 station says "temperature". 😀\t\n' * 20
    group = _group("left", text)
    result = _build("海洋观测 temperature", (group,), max_bytes=max_bytes)
    _check(result, {"left": text}, max_bytes)
    probe = PreparedContextBuilder().probe_budget(
        request=PrepareContextRequest(query="海洋观测 temperature", max_bytes=max_bytes, evidence_selection=True),
        current_scope_id="left",
        memory_candidates=(group,),
    )
    assert probe.delivered_items == len(result.fragments)
    assert probe.unused_bytes == max_bytes - result.context.content_bytes


def test_default_mode_matches_explicit_false_and_has_no_evidence_fragments() -> None:
    group = _group("left", "A station records temperature.")
    builder = PreparedContextBuilder()
    default = builder.build_scopes_result(
        request=PrepareContextRequest(query="temperature", max_bytes=1200),
        current_scope_id="left",
        memory_candidates=(group,),
    )
    explicit = _build("temperature", (group,), max_bytes=1200, selection=False)
    assert default == explicit
    assert default.fragments == ()
