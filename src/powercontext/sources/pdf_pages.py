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

"""Capture complete PDF pages and retrieve real page evidence from scoped Memory."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime.models import (
        GetMemoryEntryRequest,
        MemoryEntriesPage,
        MemoryEntryRecord,
        MemorySearchPage,
        SearchMemoryRequest,
    )


class PDFDocumentInput(BaseModel):
    """Original PDF bytes with a caller-assigned document identity."""

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def require_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("PDF document ID must not be blank")  # noqa: TRY003
        return value


class PDFPageInput(BaseModel):
    """The exact revision and physical location of one complete page."""

    document_id: str
    document_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    page_number: Annotated[int, Field(ge=1, strict=True)]
    text: str

    @field_validator("document_id")
    @classmethod
    def require_document_id(cls, value: str) -> str:
        return PDFDocumentInput.require_document_id(value)


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every physical page, publishing nothing on a later page failure."""
    # Keep this optional dependency local: core users of other Sources need not install PDF support.
    from pypdf import PdfReader

    digest = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a non-empty user password")  # noqa: TRY003, TRY301
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id, document_sha256=digest, page_number=index, text=page.extract_text() or ""
            )
            for index, page in enumerate(reader.pages, start=1)
        )
    except MemoryError:
        raise
    except Exception as error:
        raise ValueError("PDF could not be completely parsed") from error  # noqa: TRY003
    return pages


def _page_id(value: PDFPageInput) -> str:
    framed = json.dumps(
        [value.document_id, value.document_sha256, value.page_number], ensure_ascii=False, separators=(",", ":")
    )
    return "pdf:" + hashlib.sha256(framed.encode("utf-8")).hexdigest()


class PDFPageSource(Source, PDFPageInput):
    """Captured full page whose old revision remains independently addressable."""

    source_id: str
    definition_version: Literal["1"] = "1"
    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.source_id != _page_id(self) or self.name != self.source_id:
            raise ValueError("PDF page Source identity must agree with its document revision and page")  # noqa: TRY003
        return self


class _PDFPageAdapter:
    input_class = PDFPageInput
    name = "pdf-page"
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        identity = _page_id(value)
        return PDFPageSource(name=identity, source_id=identity, **value.model_dump())

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput.model_validate(source.model_dump())


class _PDFTextProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    _PDFPageAdapter(), projections=(_PDFTextProjection(),)
)


class _PDFMemory(Protocol):
    async def list(self) -> MemoryEntriesPage: ...
    async def search(self, request: SearchMemoryRequest, /) -> MemorySearchPage: ...
    async def get(self, request: GetMemoryEntryRequest, /) -> MemoryEntryRecord: ...


_STOP = frozenset([
    "a",
    "an",
    "the",
    "and",
    "or",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "from",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "where",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "i",
    "we",
    "you",
    "it",
    "its",
    "our",
    "your",
    "this",
    "that",
    "these",
    "those",
    "please",
    "has",
    "have",
    "had",
    "any",
    "into",
    "than",
    "then",
    "not",
    "if",
    "all",
    "only",
    "use",
    "using",
    "used",
    "give",
    "answer",
    "respond",
    "response",
    "question",
    "calculate",
    "compute",
    "define",
    "details",
    "shown",
    "within",
    "assuming",
    "perspective",
    "investment",
    "analyst",
    "relying",
    "explicitly",
    "directly",
    "outlined",
    "reported",
    "primarily",
    "important",
    "materially",
    "quantity",
    "unadjusted",
    "usd",
    "million",
    "millions",
    "fy",
    "state",
])
# General financial vocabulary, independent of any issuer, report, question or page label.
_CONCEPTS = (
    (
        frozenset([
            "geography",
            "geographies",
            "geographic",
            "geographical",
            "country",
            "countries",
            "region",
            "regions",
            "operates",
            "operations",
        ]),
        ("geographic segments", "countries markets", "international domestic"),
    ),
    (
        frozenset(["legal", "litigation", "lawsuit", "lawsuits", "battles", "proceedings"]),
        ("legal proceedings", "litigation contingencies"),
    ),
    (
        frozenset(["restructuring", "reorganization", "reorganisation", "severance"]),
        ("restructuring charges", "reorganization severance"),
    ),
    (frozenset(["ebitda", "depreciation", "amortization"]), ("depreciation amortization", "operating income")),
    (frozenset(["capex", "expenditures"]), ("capital expenditures", "property equipment")),
    (frozenset(["cash", "flow", "flows", "cashflow", "ebitda"]), ("cash flows", "cash operating")),
    (
        frozenset(["income", "earnings", "profit", "margin", "revenue", "revenues", "sales"]),
        ("income statement", "revenue sales"),
    ),
    (frozenset(["debt", "borrowings", "leverage"]), ("debt borrowings", "interest obligations")),
    (frozenset(["dividends", "repurchases", "shareholders", "shareholder"]), ("dividends repurchases",)),
    (frozenset(["inventory", "inventories", "turnover"]), ("inventories inventory",)),
)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", value.casefold())


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


@dataclass(frozen=True)
class _FinancialField:
    phrases: tuple[str, ...]
    statement: str


_FINANCIAL_FIELDS = {
    "current_assets": _FinancialField(("current assets",), "balance"),
    "current_liabilities": _FinancialField(("current liabilities",), "balance"),
    "assets": _FinancialField(("total assets",), "balance"),
    "equity": _FinancialField(("total equity", "shareholders equity", "stockholders equity"), "balance"),
    "inventory": _FinancialField(("inventories", "inventory"), "balance"),
    "debt": _FinancialField(("total debt", "long term debt", "borrowings"), "balance"),
    "net_income": _FinancialField(("net income", "net earnings", "net profit", "profit for the year"), "income"),
    "operating_income": _FinancialField(("operating income", "operating profit"), "income"),
    "revenue": _FinancialField(("net revenue", "net sales", "total revenue", "revenue", "sales"), "income"),
    "gross_profit": _FinancialField(("gross profit",), "income"),
    "cost_of_sales": _FinancialField(("cost of sales", "cost of goods sold", "cost of revenue", "cogs"), "income"),
    "depreciation": _FinancialField(("depreciation", "depreciation expense"), "cash"),
    "amortization": _FinancialField(("amortization", "amortisation"), "cash"),
    "operating_cash": _FinancialField(
        ("cash from operating activities", "net cash provided by operating activities", "operating cash flow"), "cash"
    ),
    "capex": _FinancialField(
        ("capital expenditures", "capital spending", "purchases of property plant and equipment"), "cash"
    ),
    "dividends": _FinancialField(("cash dividends paid", "dividends paid"), "cash"),
}
# Metric names express dependencies on financial fields, never report-specific answers.
_METRIC_FIELDS = tuple(
    (re.compile(pattern), fields)
    for pattern, fields in (
        (r"\b(?:working capital|current ratio)\b", ("current_assets", "current_liabilities")),
        (r"\b(?:quick ratio|acid test ratio)\b", ("current_assets", "inventory", "current_liabilities")),
        (r"\b(?:return on (?:total )?assets|roa)\b", ("net_income", "assets")),
        (r"\b(?:return on (?:shareholders |stockholders )?equity|roe)\b", ("net_income", "equity")),
        (r"\binventor(?:y|ies) turnover\b", ("cost_of_sales", "inventory")),
        (r"\b(?:total )?asset(?:s)? turnover\b", ("revenue", "assets")),
        (r"\bgross (?:profit )?margin\b", ("gross_profit", "revenue")),
        (r"\boperating (?:profit )?margin\b", ("operating_income", "revenue")),
        (r"\bnet (?:profit |income )?margin\b", ("net_income", "revenue")),
        (r"\bebitda\b", ("operating_income", "depreciation", "amortization")),
        (r"\bebitda margin\b", ("revenue",)),
        (r"\b(?:free cash flow|fcf)\b", ("operating_cash", "capex")),
        (r"\b(?:dividend )?payout ratio\b", ("dividends", "net_income")),
        (r"\bdebt (?:to )?equity\b", ("debt", "equity")),
    )
)


def _financial_fields(question: str) -> tuple[_FinancialField, ...]:
    normalized = " ".join(_tokens(question))
    if re.search(r"\b(?:policies|policy|risks?|manage|management)\b", normalized) and not re.search(
        r"\b(?:calculate|compute|amount|value|how much|how many)\b", normalized
    ):
        return ()
    names = dict.fromkeys(name for pattern, fields in _METRIC_FIELDS if pattern.search(normalized) for name in fields)
    # Explicit field requests also benefit from complete multi-page coverage.
    for name, field in _FINANCIAL_FIELDS.items():
        if any(len(_tokens(phrase)) > 1 and f" {phrase} " in f" {normalized} " for phrase in field.phrases):
            names.setdefault(name, None)
    return tuple(_FINANCIAL_FIELDS[name] for name in names)


def _queries(question: str, fields: tuple[_FinancialField, ...] = ()) -> tuple[list[str], dict[str, float], set[str]]:
    normalized = re.sub(r"\bd\s*&\s*a\b", "depreciation amortization", question, flags=re.IGNORECASE)
    normalized = re.sub(r"\bp\s*&\s*l\b", "income statement", normalized, flags=re.IGNORECASE)
    terms = list(dict.fromkeys(term for term in _tokens(normalized) if term not in _STOP))
    if not terms:
        return [], {}, set()
    queries = [" ".join(terms)]
    weights = {_stem(term): (0.25 if term.isdigit() else 1.0) for term in terms}
    term_set = set(terms)
    queries.extend(field.phrases[0] for field in fields)
    for field in fields:
        for phrase in field.phrases:
            queries.append(phrase)
            for word in _tokens(phrase):
                term = _stem(word)
                weights[term] = max(weights.get(term, 0.0), 0.75)
    for triggers, alternatives in _CONCEPTS:
        if triggers & term_set:
            for expansion in alternatives:
                queries.append(expansion)
                for word in _tokens(expansion):
                    weights.setdefault(_stem(word), 0.5)
    # Short independent probes also survive Memory's lexical-coverage admission floor.
    queries.extend(" ".join(terms[index : index + 2]) for index in range(0, min(len(terms), 8), 2))
    return list(dict.fromkeys(queries))[:16], weights, term_set


@dataclass
class _Candidate:
    citation: MemoryCitation
    text: str
    rank_score: float = 0.0


def _document_affinity(text: str, query_terms: set[str]) -> tuple[bool, bool]:
    document_words = set(_tokens(text.partition("\n")[0]))
    names = {
        word
        for word in document_words
        if not word.isdigit()
        and word not in {"k", "q", "annual", "report", "form", "financial", "inc", "corp", "company"}
    }
    years = {word for word in query_terms if re.fullmatch(r"(?:19|20)\d{2}", word)}
    return bool(names & query_terms), bool(document_words & years)


def _rank_pages(candidates: list[_Candidate], weights: dict[str, float], query_terms: set[str]) -> list[_Candidate]:
    counts = [Counter(_stem(token) for token in _tokens(candidate.text)) for candidate in candidates]
    document_frequency: Counter[str] = Counter(term for row in counts for term in row)
    average_length = sum(sum(row.values()) for row in counts) / max(len(counts), 1)
    scored: list[tuple[float, _Candidate]] = []
    for candidate, terms in zip(candidates, counts, strict=True):
        length = sum(terms.values())
        score = candidate.rank_score * 6.0
        for term, weight in weights.items():
            frequency = terms[term]
            if frequency:
                inverse_frequency = math.log1p(
                    (len(counts) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
                )
                norm = frequency + 1.2 * (0.35 + 0.65 * length / max(average_length, 1.0))
                score += weight * inverse_frequency * frequency * 2.2 / norm
        issuer_match, year_match = _document_affinity(candidate.text, query_terms)
        score += 2.0 * issuer_match + 0.75 * year_match
        scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1].citation.entry_id, item[1].citation.entry_version_id))
    return [candidate for _, candidate in scored]


_STATEMENT_TITLES = {
    "income": re.compile(r"^(?:statements? of (?:income|operations|earnings)|income statements?|profit and loss)\b"),
    "cash": re.compile(r"^statements? of cash flows?\b"),
    "balance": re.compile(r"^(?:balance sheets?|statements? of financial position)\b"),
    "equity": re.compile(r"^statements? of (?:changes in )?(?:(?:shareholders|stockholders)'? )?equity\b"),
}
_TITLE_MODIFIERS = re.compile(r"^(?:(?:condensed|consolidated|combined|unaudited|audited|group|interim)\s+)+")


def _requested_statements(query_terms: set[str]) -> set[str]:
    requested = set()
    has_statement = bool({"statement", "statements"} & query_terms)
    if has_statement and {"income", "earnings", "operations", "pnl"} & query_terms:
        requested.add("income")
    if "cash" in query_terms and {"flow", "flows"} & query_terms:
        requested.add("cash")
    if "balance" in query_terms and {"sheet", "sheets"} & query_terms:
        requested.add("balance")
    if has_statement and "equity" in query_terms:
        requested.add("equity")
    return requested


def _page_statements(text: str) -> set[str]:
    # Read titles near the page start, rather than incidental mentions in notes.
    # TOC entries ending in a page number are not financial statements themselves.
    headings = [line.strip().casefold() for line in text.partition("\n")[2].splitlines() if line.strip()][:4]
    return {
        kind
        for line in headings
        if len(line) <= 160 and not re.search(r"\d\s*$", line)
        for kind, pattern in _STATEMENT_TITLES.items()
        if pattern.match(_TITLE_MODIFIERS.sub("", line))
    }


def _cover_requested_statements(ranked: list[_Candidate], query_terms: set[str]) -> list[_Candidate]:
    requested = _requested_statements(query_terms)
    if not requested:
        return ranked
    choices: dict[str, list[int]] = {kind: [] for kind in requested}
    for index, candidate in enumerate(ranked):
        for kind in _page_statements(candidate.text) & requested:
            choices[kind].append(index)
    for kind, indices in choices.items():
        affinities = [(_document_affinity(ranked[index].text, query_terms), index) for index in indices]
        best = max((affinity for affinity, _ in affinities), default=(False, False))
        choices[kind] = [index for affinity, index in affinities if affinity == best][:2]
    # Multi-part questions need each named statement, even when related notes
    # repeat more of the question's vocabulary. Prefer matching issuer/year
    # before taking a kind's best pages. Keep global rank within each
    # coverage round, then fill the remaining places with the original ranking.
    selected: list[int] = []
    for depth in range(2):
        for index in sorted({indices[depth] for indices in choices.values() if len(indices) > depth}):
            if index not in selected:
                selected.append(index)
    selected_set = set(selected)
    return [ranked[index] for index in selected] + [
        row for index, row in enumerate(ranked) if index not in selected_set
    ]


def _cover_financial_fields(
    ranked: list[_Candidate], fields: tuple[_FinancialField, ...], query_terms: set[str]
) -> list[_Candidate]:
    if not fields:
        return ranked
    phrases = [
        tuple(" " + " ".join(_stem(word) for word in _tokens(phrase)) + " " for phrase in field.phrases)
        for field in fields
    ]
    coverage: list[set[int]] = []
    numeric: list[set[int]] = []
    titles: list[set[str]] = []
    for candidate in ranked:
        lines = candidate.text.partition("\n")[2].splitlines()
        rows = [
            (" " + " ".join(_stem(word) for word in _tokens(line)) + " ", bool(re.search(r"\d", line)))
            for line in lines
        ]
        body = " " + " ".join(_stem(word) for word in _tokens(candidate.text.partition("\n")[2])) + " "
        coverage.append({i for i, alternatives in enumerate(phrases) if any(p in body for p in alternatives)})
        numeric.append({
            i
            for i, alternatives in enumerate(phrases)
            if any(has_number and any(p in row for p in alternatives) for row, has_number in rows)
        })
        titles.append(_page_statements(candidate.text))
    remaining = set(range(len(fields)))
    selected: list[int] = []
    while remaining:
        eligible = [i for i in range(len(ranked)) if coverage[i] & remaining]
        if not eligible:
            break
        # Prefer the requested issuer/year and numeric evidence. A short explanation
        # of a formula must not displace the statements containing its actual inputs.
        best = max(
            eligible,
            key=lambda i: (
                *_document_affinity(ranked[i].text, query_terms),
                len(numeric[i] & remaining),
                len(coverage[i] & remaining),
                len(titles[i] & {fields[j].statement for j in coverage[i] & remaining}),
                -i,
            ),
        )
        selected.append(best)
        remaining.difference_update(coverage[best])
    chosen = set(selected)
    return [ranked[i] for i in selected] + [candidate for i, candidate in enumerate(ranked) if i not in chosen]


async def _collect_page_candidates(memory: _PDFMemory, queries: list[str], limit: int) -> list[_Candidate]:
    """Read one active scope snapshot, then add lexical evidence to its real citations."""
    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime.models import SearchMemoryRequest

    candidates: dict[tuple[str, int, str, str], _Candidate] = {}
    # Use the public active-head view to retain pages absent from lexical probes.
    # Candidate text and citations both come from Memory, never from report files
    # or annotations. The task corpus is bounded and the ranking is request-local.
    active = await memory.list()
    for record in active.entries:
        page_ids = {source.source_id for source in record.entry.sources if source.source_type == "pdf-page"}
        if len(page_ids) == 1:
            citation = record.citation
            key = (
                citation.memory_ref.artifact_id,
                citation.memory_ref.revision,
                citation.entry_id,
                citation.entry_version_id,
            )
            candidates[key] = _Candidate(citation, record.entry.text)
    for query in queries:
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=max(100, limit * 4)))
        for rank, hit in enumerate(page.hits, start=1):
            key = (hit.memory_ref.artifact_id, hit.memory_ref.revision, hit.entry_id, hit.entry_version_id)
            if key not in candidates:
                candidates[key] = _Candidate(
                    MemoryCitation(
                        memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id
                    ),
                    hit.text,
                )
            candidates[key].rank_score += 1.0 / (60 + rank)
    return list(candidates.values())


async def search_pdf_pages(memory: _PDFMemory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Fuse lexical probes, rank full page text, and return distinct real citations."""
    from powercontext.builtin.runtime.models import GetMemoryEntryRequest

    if limit <= 0 or not question.strip():
        return ()
    fields = _financial_fields(question)
    queries, weights, question_terms = _queries(question, fields)
    candidates = await _collect_page_candidates(memory, queries, limit)
    seen_pages: set[str] = set()
    selected: list[MemoryCitation] = []
    ranked = _cover_requested_statements(_rank_pages(candidates, weights, question_terms), question_terms)
    for candidate in _cover_financial_fields(ranked, fields, question_terms):
        record = (await memory.get(GetMemoryEntryRequest(citation=candidate.citation))).entry
        page_ids = {source.source_id for source in record.sources if source.source_type == "pdf-page"}
        if len(page_ids) != 1:
            continue
        page_id = next(iter(page_ids))
        if page_id in seen_pages:
            continue
        seen_pages.add(page_id)
        selected.append(candidate.citation)
        if len(selected) == limit:
            break
    return tuple(selected)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "search_pdf_pages",
]
