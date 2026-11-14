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

import hashlib
import io
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from competition.selftest_support.source_acceptance import persist
from powercontext.builtin.runtime import GetMemoryEntryRequest, open_builtin_runtime
from powercontext.sources import SourceDefinitionRegistry, project_source_for_transport
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageInput,
    parse_pdf_document,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def make_pdf(texts: list[str], *, password: str | None = None, broken_last_page: bool = False) -> bytes:
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
    )
    for index, text in enumerate(texts):
        page = writer.add_blank_page(width=600, height=800)
        if not text:
            continue
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        data = f"BT /F1 10 Tf 50 750 Td ({escaped}) Tj ET".encode("ascii")
        if broken_last_page and index == len(texts) - 1:
            data = b"BT /F1 10 Tf (unterminated"
        stream.set_data(data)
        page[NameObject("/Contents")] = writer._add_object(stream)
    if password is not None:
        writer.encrypt(user_password=password, owner_password="fixture-owner")  # noqa: S106
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


async def test_complete_physical_order_numbers_blank_page_and_source_roundtrip() -> None:
    content = make_pdf(["Net revenue 12,345 006.70 (2024)", "", "Cash flows 98765 2023"])
    pages = parse_pdf_document(PDFDocumentInput(document_id="A" * 500, content=content))
    assert len(pages) == 3 and [page.page_number for page in pages] == [1, 2, 3]
    assert pages[1].text == ""
    assert "12,345 006.70 (2024)" in pages[0].text
    assert all(page.document_sha256 == hashlib.sha256(content).hexdigest() for page in pages)
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    for page in pages:
        source = await registry.resolve(page)
        assert len(source.name) <= 256
        assert await registry.read(source) == page
        projection = registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY)
        assert isinstance(projection, dict)
        assert projection["content"] == f"{page.document_id}\n{page.text}"
        assert projection["metadata"] == {
            "document_id": page.document_id,
            "document_sha256": page.document_sha256,
            "page_number": page.page_number,
        }


@pytest.mark.parametrize(
    "content",
    [
        make_pdf(["Readable"], password="fixture-password"),  # noqa: S106
        make_pdf(["First page", "Damaged last page"], broken_last_page=True),
    ],
)
def test_password_or_later_page_failure_is_atomic(content: bytes) -> None:
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="broken", content=content))


@pytest.mark.parametrize(
    "updates", [{"document_id": "  "}, {"document_sha256": "ABC"}, {"page_number": 0}, {"page_number": True}]
)
def test_page_input_rejects_invalid_location_identity(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PDFPageInput.model_validate(
            {"document_id": "report", "document_sha256": "a" * 64, "page_number": 1, "text": "page"} | updates
        )


async def test_multi_document_financial_queries_keep_real_distinct_page_citations(tmp_path: Path) -> None:
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    observations = []
    metadata_by_source = {}
    for company in ("ALPHA", "BETA", "GAMMA"):
        texts = [
            "Geographic segments include domestic and international countries and markets.",
            "Consolidated statement of income. Operating income 100. Net revenue 900. Cost of sales 800.",
            "Statement of cash flows. Depreciation and amortization 40. Capital expenditures 30. Property equipment.",
            "Legal proceedings and litigation. Material contingencies and lawsuits remain pending.",
            "Employee training attendance 150 and executive compensation policy.",
        ]
        for page in parse_pdf_document(PDFDocumentInput(document_id=f"{company}_2024_10K", content=make_pdf(texts))):
            source = await registry.resolve(page)
            observations.append(project_source_for_transport(registry, source))
            metadata_by_source[source.name] = (company, page.page_number)
    config, scope_id, _ = await persist(tmp_path, PDF_PAGE_SOURCE_DEFINITION, observations)
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope_id)
        for question, expected in [
            ("What geographies does ALPHA operate in FY2024?", {("ALPHA", 1)}),
            ("What legal battles does BETA have in FY2024?", {("BETA", 4)}),
            (
                "What is GAMMA FY2024 EBITDA less capex using operating income, depreciation and amortization?",
                {("GAMMA", 2), ("GAMMA", 3)},
            ),
        ]:
            citations = await search_pdf_pages(memory, question, limit=2)
            found = set()
            for citation in citations:
                entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
                found.update(metadata_by_source[source.source_id] for source in entry.sources)
            assert expected <= found
            assert len(citations) == len(found) <= 2
        assert await search_pdf_pages(memory, "BETA cash flow", limit=0) == ()
        assert await search_pdf_pages(memory, " ", limit=10) == ()


async def test_long_page_preserves_full_memory_citation_without_truncation(tmp_path: Path) -> None:
    text = "Operating income and revenue reported in complete financial statements. " * 160
    (page,) = parse_pdf_document(PDFDocumentInput(document_id="LARGE_2024_10K", content=make_pdf([text])))
    assert len(page.text.encode("utf-8")) > 8192
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    source = await registry.resolve(page)
    observation = project_source_for_transport(registry, source)
    config, scope_id, _ = await persist(tmp_path, PDF_PAGE_SOURCE_DEFINITION, [observation])
    async with open_builtin_runtime(config) as runtime:
        citations = await search_pdf_pages(runtime.memory.for_scope(scope_id), "LARGE operating income", limit=1)
        assert len(citations) == 1
        entry = (await runtime.memory.for_scope(scope_id).get(GetMemoryEntryRequest(citation=citations[0]))).entry
        assert entry.text == f"{page.document_id}\n{page.text}".strip()


async def test_report_page_without_original_question_words_is_not_lost_before_ranking(tmp_path: Path) -> None:
    texts = ["OPERATEX company overview for fiscal 2024."]
    texts.extend(["Annual filing schedule and employee training policy."] * 110)
    texts.append("Asia Pacific, Europe and North America market distribution.")
    pages = parse_pdf_document(PDFDocumentInput(document_id="OPERATEX_2024_10K", content=make_pdf(texts)))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    observations = [project_source_for_transport(registry, await registry.resolve(page)) for page in pages]
    expected_source = observations[-1].name
    config, scope_id, _ = await persist(tmp_path, PDF_PAGE_SOURCE_DEFINITION, observations)
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope_id)
        citations = await search_pdf_pages(
            memory, "What geographies does OPERATEX primarily operate in FY2024?", limit=20
        )
        found = set()
        for citation in citations:
            entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
            found.update(reference.source_id for reference in entry.sources)
        assert expected_source in found


async def test_multi_statement_question_retains_each_required_financial_statement(tmp_path: Path) -> None:
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    income = "Consolidated Statement of Income\nNet revenue 1000\nOperating profit 120\nNet income 90"
    cash = "Consolidated Statement of Cash Flows\nDepreciation and amortization 35\nCapital spending 200"
    notes = [
        f"Note {index}: segment results and non-GAAP measures\n"
        "Derivative instruments\nFair value and cash flow hedges\nLosses recognized in\nIncome Statement\n"
        "Operating income and adjusted EBITDA include depreciation and amortization. "
        "Capital expenditures, cash flows, and income statement comparisons are discussed below. "
        "Unadjusted operating income and cash flow statement disclosures include capex."
        for index in range(25)
    ]
    pages = parse_pdf_document(PDFDocumentInput(document_id="ACME_2024_10K", content=make_pdf([income, cash, *notes])))
    sources = [await registry.resolve(page) for page in pages]
    observations = [project_source_for_transport(registry, source) for source in sources]
    for document_id in ("OTHER_2024_10K", "ACME_2023_10K"):
        related = "EBITDA operating income depreciation amortization cash flow statement capex " * 5
        foreign_pages = parse_pdf_document(
            PDFDocumentInput(
                document_id=document_id, content=make_pdf([income + "\n" + related, cash + "\n" + related])
            )
        )
        for page in foreign_pages:
            source = await registry.resolve(page)
            observations.append(project_source_for_transport(registry, source))
    config, scope_id, _ = await persist(tmp_path, PDF_PAGE_SOURCE_DEFINITION, observations)
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope_id)
        citations = await search_pdf_pages(
            memory,
            "What is ACME FY2024 EBITDA less capex? Use operating income from the income statement "
            "and depreciation and amortization from the cash flow statement.",
            limit=2,
        )
        found = set()
        for citation in citations:
            entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
            found.update(reference.source_id for reference in entry.sources)
        assert {source.name for source in sources[:2]} <= found


async def test_financial_metrics_retrieve_all_components_without_losing_narrative_policy(tmp_path: Path) -> None:
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    income = "Consolidated Statement of Income\nNet earnings 90\nCost of sales 800\nRevenue 1000\nOperating profit 120"
    balance = (
        "Consolidated Balance Sheet\nCurrent assets 600\nCurrent liabilities 250\n"
        "Total assets 2000\nInventories 180\nShareholders equity 1100"
    )
    cash = (
        "Consolidated Statement of Cash Flows\nCash from operating activities 140\n"
        "Capital spending 60\nDepreciation and amortization 35\nCash dividends paid 40"
    )
    policy = "Working capital policy\nWe manage current assets and current liabilities within group risk limits."
    formula = (
        "Financial metrics reference\nWorking capital is current assets less current liabilities. "
        "Inventory turnover relates cost of sales and average inventories. "
        "Return on assets compares net income and total assets."
    )
    noise = [
        f"Note {i}: performance discussion\nWorking capital, return on assets, return on equity, "
        "inventory turnover and asset turnover are management indicators."
        for i in range(25)
    ]
    observations = []
    target = []
    policy_ids = set()
    for document_id in ("MERIDIAN_2025_ANNUAL", "MERIDIAN_2024_ANNUAL", "ORION_2025_ANNUAL"):
        pages = parse_pdf_document(
            PDFDocumentInput(
                document_id=document_id, content=make_pdf([income, balance, cash, policy, formula, *noise])
            )
        )
        for page in pages:
            source = await registry.resolve(page)
            observations.append(project_source_for_transport(registry, source))
            if document_id.startswith("MERIDIAN_") and page.page_number == 4:
                policy_ids.add(source.name)
            if document_id == "MERIDIAN_2025_ANNUAL":
                target.append(source.name)
    config, scope_id, _ = await persist(tmp_path, PDF_PAGE_SOURCE_DEFINITION, observations)
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope_id)
        for metric, required in [
            ("working capital", [1]),
            ("current ratio", [1]),
            ("acid-test ratio", [1]),
            ("return on assets", [0, 1]),
            ("ROE", [0, 1]),
            ("inventory turnover", [0, 1]),
            ("asset turnover", [0, 1]),
            ("EBITDA", [0, 2]),
            ("free cash flow", [2]),
            ("dividend payout ratio", [0, 2]),
            ("net profit margin", [0]),
        ]:
            citations = await search_pdf_pages(memory, f"What was MERIDIAN FY2025 {metric}?", limit=2)
            found = {
                source.source_id
                for citation in citations
                for source in (await memory.get(GetMemoryEntryRequest(citation=citation))).entry.sources
            }
            assert {target[index] for index in required} <= found, metric
        (citation,) = await search_pdf_pages(memory, "How does MERIDIAN manage working capital policy?", limit=1)
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        assert len(entry.sources) == 1 and entry.sources[0].source_id in policy_ids
        (citation,) = await search_pdf_pages(
            memory, "How did MERIDIAN manage working capital policy in FY2025?", limit=1
        )
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        assert {source.source_id for source in entry.sources} == {target[3]}


@pytest.mark.parametrize(
    "paragraph", ["", "A normal paragraph of longer words covers the majority of this page. " * 20]
)
def test_individually_positioned_letters_preserve_words_columns_and_line_breaks(paragraph: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=400)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Courier"),
            })
        })
    })
    ops = [f"BT /F1 10 Tf 20 250 Td ({paragraph}) Tj ET"] if paragraph else []
    for y, x, words in [(200, 20, "NET INCOME"), (200, 180, "56"), (175, 20, "TOTAL ASSETS"), (175, 180, "92")]:
        for char in words:
            if char != " ":
                ops.append(f"q 1 0 0 1 {x} {y} cm BT /F1 10 Tf ({char}) Tj ET Q")
            x += 6
    stream = DecodedStreamObject()
    stream.set_data(" ".join(ops).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    (captured,) = parse_pdf_document(PDFDocumentInput(document_id="text-runs", content=output.getvalue()))
    assert captured.text.splitlines() == ([paragraph] if paragraph else []) + ["NET INCOME 56", "TOTAL ASSETS 92"]
