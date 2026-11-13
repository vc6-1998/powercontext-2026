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

"""Query-driven literal evidence selection under the complete wire budget."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from powercontext.builtin.runtime.prepared_text import TRUST_POLICY

if TYPE_CHECKING:
    from powercontext.builtin.runtime.prepared_context import MemoryEntryAddress

_BEGIN = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
_END = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"
_TOKEN = re.compile(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", re.UNICODE)
_BOUNDARY = re.compile(r"[.!?](?=\s|$)|[。！？]|\n+")  # noqa: RUF001 - Fullwidth sentence marks.
_STOP = frozenset([
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "for",
    "to",
    "by",
    "with",
    "without",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "as",
    "from",
    "can",
    "could",
    "may",
    "might",
    "than",
    "which",
    "who",
    "has",
    "have",
    "had",
    "we",
    "our",
    "they",
    "their",
    "both",
    "between",
    "into",
    "using",
    "used",
    "also",
    "not",
    "no",
    "do",
    "does",
    "did",
    "such",
    "through",
    "during",
    "whether",
    "show",
    "shows",
    "study",
    "studies",
])
_RESULTS = re.compile(
    r"\b(?:found|showed|observed|revealed|demonstrated|increased|decreased|reduced|inhibited|induced|significantly|"
    r"results|conclude|suggest|associated|correlated)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceDocument:
    address: MemoryEntryAddress
    text: str


@dataclass(frozen=True)
class EvidenceFragment:
    """One exact half-open Python-character interval in a cited Memory body."""

    address: MemoryEntryAddress
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Sentence:
    document: int
    start: int
    end: int
    terms: frozenset[str]
    score: float


def _stem(word: str) -> str:
    # Connect conservative inflections and common scientific verb/noun forms;
    # this affects ranking only, never the original delivered text or offsets.
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suffix in ("ing", "ed", "ion", "ity"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def _terms(text: str) -> list[str]:
    return [_stem(match.group().casefold()) for match in _TOKEN.finditer(text) if match.group().casefold() not in _STOP]


def _spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for end in [*(match.end() for match in _BOUNDARY.finditer(text)), len(text)]:
        if end <= start:
            continue
        while end < len(text) and text[end].isspace():
            end += 1
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    return spans


def _sentences(query: str, documents: Sequence[EvidenceDocument]) -> tuple[list[_Sentence], dict[str, float]]:
    query_terms = set(_terms(query))
    document_terms = [Counter(_terms(document.text)) for document in documents]
    weights = {}
    for term in query_terms:
        frequency = sum(term in terms for terms in document_terms)
        weights[term] = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
    raw = [
        (index, start, end, Counter(_terms(document.text[start:end])))
        for index, document in enumerate(documents)
        for start, end in _spans(document.text)
    ]
    average = sum(sum(terms.values()) for _index, _start, _end, terms in raw) / max(1, len(raw))
    total_weight = max(sum(weights.values()), 1e-9)
    scored = []
    for index, start, end, terms in raw:
        matches = query_terms & terms.keys()
        coverage = sum(weights[term] for term in matches) / total_weight
        length = sum(terms.values())
        lexical = sum(
            weights[term] * terms[term] * 2.2 / (terms[term] + 1.2 * (0.4 + 0.6 * length / max(1, average)))
            for term in matches
        )
        doc_coverage = sum(weights[term] for term in query_terms & document_terms[index].keys()) / total_weight
        result_bonus = 1.12 if _RESULTS.search(documents[index].text[start:end]) else 1.0
        score = lexical * (0.6 + 0.4 * coverage) * result_bonus + 0.12 * doc_coverage
        scored.append(_Sentence(index, start, end, frozenset(matches), score))
    contextual = []
    for index, sentence in enumerate(scored):
        # Neighboring conclusions may refer to the matched experiment by pronoun.
        adjacent = [
            scored[other].score
            for other in (index - 1, index + 1)
            if 0 <= other < len(scored) and scored[other].document == sentence.document
        ]
        contextual.append(
            _Sentence(
                sentence.document,
                sentence.start,
                sentence.end,
                sentence.terms,
                sentence.score + 0.22 * max(adjacent, default=0),
            )
        )
    return contextual, weights


def _merge(
    selected: Sequence[tuple[int, int, int]], addition: tuple[int, int, int], documents: Sequence[EvidenceDocument]
) -> list[tuple[int, int, int]]:
    merged: list[tuple[int, int, int]] = []
    for document, start, end in sorted((*selected, addition)):
        if merged and merged[-1][0] == document:
            old_document, old_start, old_end = merged[-1]
            if start <= old_end or not documents[document].text[old_end:start].strip():
                merged[-1] = old_document, old_start, max(old_end, end)
                continue
        merged.append((document, start, end))
    return merged


def _fragments(
    intervals: Sequence[tuple[int, int, int]], documents: Sequence[EvidenceDocument]
) -> tuple[EvidenceFragment, ...]:
    return tuple(
        EvidenceFragment(documents[index].address, start, end, documents[index].text[start:end])
        for index, start, end in intervals
    )


def render_evidence(fragments: Sequence[EvidenceFragment]) -> str:
    items = [
        {
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
        for fragment in fragments
    ]
    encoded = json.dumps({"trust": "untrusted_history", "items": items}, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN}\n{encoded}\n{_END}"))


def _window(
    sentence: _Sentence,
    query: str,
    documents: Sequence[EvidenceDocument],
    selected: Sequence[tuple[int, int, int]],
    max_bytes: int,
) -> tuple[int, int, int] | None:
    """Fit a bounded original word window around a query match."""

    text = documents[sentence.document].text
    words = list(_TOKEN.finditer(text, sentence.start, sentence.end))
    if not words:
        return None
    query_terms = set(_terms(query))
    anchors = [index for index, word in enumerate(words) if _stem(word.group().casefold()) in query_terms] or [0]
    # Rank dense anchor neighborhoods to bound work on very repetitive entries.
    anchors.sort(
        key=lambda index: (
            -len(
                query_terms
                & set(_terms(text[words[max(0, index - 12)].start() : words[min(len(words) - 1, index + 12)].end()]))
            )
        )
    )
    best = None
    best_score = (-1, -1)
    for anchor in anchors[:16]:
        for balance in (0.33, 0.67):
            low, high = 1, len(words)
            fitted = None
            while low <= high:
                count = (low + high) // 2
                left = max(0, min(anchor - int(count * balance), len(words) - count))
                right = left + count
                start = sentence.start if left == 0 else words[left].start()
                end = sentence.end if right == len(words) else words[right - 1].end()
                interval = sentence.document, start, end
                size = len(
                    render_evidence(_fragments(_merge(selected, interval, documents), documents)).encode("utf-8")
                )
                if size <= max_bytes:
                    fitted = interval
                    low = count + 1
                else:
                    high = count - 1
            if fitted is not None:
                _, start, end = fitted
                score = (len(query_terms & set(_terms(text[start:end]))), end - start)
                if score > best_score:
                    best, best_score = fitted, score
    return best


def select_evidence(query: str, documents: Sequence[EvidenceDocument], max_bytes: int) -> tuple[EvidenceFragment, ...]:
    """Add coherent sentences by relevance and marginal serialized byte cost."""

    if not documents:
        return ()
    sentences, weights = _sentences(query, documents)
    remaining = list(sentences)
    selected: list[tuple[int, int, int]] = []
    covered: set[str] = set()
    current_bytes = len(render_evidence(()).encode("utf-8"))
    while remaining:
        best = None
        for position, sentence in enumerate(remaining):
            interval = sentence.document, sentence.start, sentence.end
            proposal = _merge(selected, interval, documents)
            size = len(render_evidence(_fragments(proposal, documents)).encode("utf-8"))
            if size > max_bytes:
                continue
            novelty = sum(weights[term] for term in sentence.terms - covered)
            utility = (sentence.score + 0.25 * novelty) / max(1, size - current_bytes) ** 0.35
            choice = utility, -position, position, proposal, size
            if best is None or choice[:2] > best[:2]:
                best = choice
        if best is None:
            break
        _utility, _tie, position, selected, current_bytes = best
        covered.update(remaining.pop(position).terms)
    # Keep complete sentences first; one literal partial window may use the slack.
    for sentence in sorted(remaining, key=lambda item: -item.score):
        interval = _window(sentence, query, documents, selected, max_bytes)
        if interval is not None:
            selected = _merge(selected, interval, documents)
            break
    return _fragments(selected, documents)


__all__ = ["EvidenceDocument", "EvidenceFragment", "render_evidence", "select_evidence"]
