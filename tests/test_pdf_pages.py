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
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

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


@pytest.mark.parametrize("blank", [True, False])
def test_null_font_cannot_hide_meaningful_text(blank: bool) -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(["Visible revenue 42"])))
    page = writer.pages[0]
    resources = page["/Resources"]
    assert isinstance(resources, DictionaryObject)
    fonts = resources["/Font"]
    assert isinstance(fonts, DictionaryObject)
    fonts[NameObject("/NullFont")] = IndirectObject(0, 0, writer)
    stream = DecodedStreamObject()
    glyphs = b" " if blank else b"Hidden revenue 99"
    contents = page.get_contents()
    assert contents is not None
    stream.set_data(contents.get_data() + b" BT /NullFont 10 Tf (" + glyphs + b") Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    value = PDFDocumentInput(document_id="font-resource", content=output.getvalue())
    if blank:
        assert parse_pdf_document(value)[0].text.strip() == "Visible revenue 42"
    else:
        with pytest.raises(ValueError):
            parse_pdf_document(value)


def test_two_long_text_fragments_form_one_word_without_a_global_short_word_trigger() -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(["Long paragraph containing ordinary words. " * 20])))
    page = writer.pages[0]
    resources = page["/Resources"]
    assert isinstance(resources, DictionaryObject)
    fonts = resources["/Font"]
    assert isinstance(fonts, DictionaryObject)
    fonts[NameObject("/Mono")] = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Courier"),
    })
    stream = DecodedStreamObject()
    contents = page.get_contents()
    assert contents is not None
    stream.set_data(
        contents.get_data() + b" q 1 0 0 1 20 700 cm BT /Mono 10 Tf (attributab) Tj ET Q"
        b" q 1 0 0 1 80.1 700 cm BT /Mono 10 Tf (le) Tj ET Q"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    text = parse_pdf_document(PDFDocumentInput(document_id="fragmented", content=output.getvalue()))[0].text
    assert text.endswith("attributable")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Net\nincome\u00a0Debt service 42", "Net income Debt service 42"),
        ("ﬁnancial ﬂows oﬃce ﬁled certiﬁcates", "financial flows office filed certificates"),
    ],
)
def test_mapped_separators_and_ligatures_preserve_readable_words(text: str, expected: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    chars = list(dict.fromkeys(text))
    cmap = DecodedStreamObject()
    mappings = "\n".join(f"<{i:04x}> <{ord(char):04x}>" for i, char in enumerate(chars, 1))
    cmap.set_data(
        (
            "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
            "/CMapName /Identity-UCS def /CMapType 2 def\n1 begincodespacerange <0000> <ffff> endcodespacerange\n"
            f"{len(chars)} beginbfchar\n{mappings}\nendbfchar\nendcmap end end"
        ).encode("ascii")
    )
    descendant = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/CIDFontType2"),
        NameObject("/BaseFont"): NameObject("/Courier"),
        NameObject("/DW"): NumberObject(600),
    })
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type0"),
        NameObject("/BaseFont"): NameObject("/Courier"),
        NameObject("/Encoding"): NameObject("/Identity-H"),
        NameObject("/DescendantFonts"): ArrayObject([descendant]),
        NameObject("/ToUnicode"): writer._add_object(cmap),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
    })
    encoded = "".join(f"{chars.index(char) + 1:04x}" for char in text)
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 10 Tf 20 700 Td <{encoded}> Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    captured = parse_pdf_document(PDFDocumentInput(document_id="mapped", content=output.getvalue()))[0].text
    assert captured == expected


async def test_wrapped_titles_issuer_aliases_and_capital_intensity(tmp_path: Path) -> None:
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    observations = []
    identities = {}
    for document, issuer, ticker in [
        ("NORTH_STAR_2025_ANNUAL", "North Star", "NST"),
        ("CEDAR_2025_ANNUAL", "Cedar", "CDR"),
    ]:
        texts = [
            f"{issuer} corporate overview. NYSE: {ticker}",
            f"Table of Contents\n{issuer} Consolidated Statements of\nOperations\nRevenue 1000\nCost of sales 700\nOperating profit 150",
            f"Table\nof\nContents\n{issuer}\nConsolidated Balance Sheet\nProperty plant and equipment 900\nAccounts payable 80\nInventories 60",
            f"{issuer}\nConsolidated Statement of Cash Flows\nPurchases of property plant and equipment 80\nDepreciation 60",
            *[
                f"Note {i}: operating metrics\nCapital intensive business, DPO and operating margin are measures of performance."
                for i in range(25)
            ],
            *[
                "The following table presents amounts in the Consolidated Statement of Income and Consolidated Balance Sheets.\n"
                "Revenue 2000\nProperty plant and equipment 1500\nCapital expenditures 90\nCost of sales 50\nAccounts payable 20"
                for _ in range(10)
            ],
        ]
        for page in parse_pdf_document(PDFDocumentInput(document_id=document, content=make_pdf(texts))):
            source = await registry.resolve(page)
            identities[source.name] = (document, page.page_number)
            observations.append(project_source_for_transport(registry, source))
    config, scope, _ = await persist(tmp_path, PDF_PAGE_SOURCE_DEFINITION, observations)
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope)
        for question, document, pages in [
            ("What was North Star FY2025 operating margin?", "NORTH_STAR_2025_ANNUAL", {2}),
            ("Is NnS capital intensive in FY2025?", "NORTH_STAR_2025_ANNUAL", {2, 3, 4}),
            ("What was CDR FY2025 DPO?", "CEDAR_2025_ANNUAL", {2, 3}),
        ]:
            citations = await search_pdf_pages(memory, question, limit=len(pages))
            found = {
                identities[source.source_id]
                for citation in citations
                for source in (await memory.get(GetMemoryEntryRequest(citation=citation))).entry.sources
            }
            assert {(document, page) for page in pages} == found, question


async def test_specific_query_term_can_outweigh_common_terms_on_short_pages(tmp_path: Path) -> None:
    texts = [
        "Forensic compliance controls. " + "Regional employee training records and annual scheduling details. " * 6,
        *[
            "Compliance controls overview. Regional employee training records and annual scheduling details."
            for _ in range(24)
        ],
        *["Unrelated telecommunications administration. " * 6 for _ in range(24)],
    ]
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    observations = []
    for page in parse_pdf_document(PDFDocumentInput(document_id="CEDAR_2025", content=make_pdf(texts))):
        observations.append(project_source_for_transport(registry, await registry.resolve(page)))
    config, scope, _ = await persist(tmp_path, PDF_PAGE_SOURCE_DEFINITION, observations)
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope)
        (citation,) = await search_pdf_pages(memory, "CEDAR forensic compliance controls in FY2025", limit=1)
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        assert {source.source_id for source in entry.sources} == {observations[0].name}


@pytest.mark.parametrize(
    ("operations", "expected"),
    [
        (
            "BT /F1 10 Tf 20 700 Td (shar) Tj ET BT /F1 10 Tf 48 700 Td (es) Tj ET",
            "shares",
        ),
        (
            "BT /F1 10 Tf 20 700 Td (word) Tj ET BT /F1 10 Tf 48.5 700 Td (boundary) Tj ET",
            "word boundary",
        ),
        (
            "BT /F1 10 Tf 20 700 Td (FORM 10-K) Tj ET "
            + " ".join(f"BT /F1 10 Tf {20 + 6 * i} 700 Td ({char}) Tj ET" for i, char in enumerate("FORM 10-K")),
            "FORM 10-K",
        ),
        (
            "BT /F1 10 Tf 20 700 Td (A) Tj -4 Tc ( ) Tj 0 Tc (B) Tj ET",
            "A B",
        ),
        (
            "q .1 0 0 .1 0 0 cm BT /F1 100 Tf 200 7000 Td (2018) Tj 400 0 Td (2017) Tj ET Q",
            "2018 2017",
        ),
        (
            "BT /F1 10 Tf 20 700 Td [( )0( )0( )] TJ ET BT /F1 10 Tf 25 700 Td [(2)0(0)0(2)0(2)] TJ ET",
            "2022",
        ),
        (
            "BT /F1 10 Tf 20 700 Td (2022) Tj ( BOOKKEEPER) Tj ET",
            "2022 BOOKKEEPER",
        ),
        (
            "BT /F1 10 Tf 20 700 Td [(2) (0) (2) (2) ( ) (B) (O) (O) (K)] TJ ET",
            "2022 BOOK",
        ),
        (
            "BT /F1 10 Tf 20 700 Td [(A) -500 -500 (B)] TJ ET",
            "A B",
        ),
        (
            "BT /F1 10 Tf 20 700 Td (shar) Tj ET BT /F1 10 Tf 45.5 700 Td (es) Tj ET",
            "shares",
        ),
        (
            "BT /F1 10 Tf 20 700 Td (FORM 10-K) Tj ET BT /F1 10 Tf 20 700 Td (FORM 10-K) Tj ET",
            "FORM 10-K",
        ),
        (
            "BT /F1 10 Tf 20 700 Td (REVENUE) Tj ET BT /F1 10 Tf 20 675 Td (REVENUE) Tj ET",
            "REVENUE\nREVENUE",
        ),
    ],
)
def test_positioned_words_preserve_kerning_and_only_remove_identical_overprints(operations: str, expected: str) -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(["placeholder"])))
    page = writer.pages[0]
    page["/Resources"]["/Font"]["/F1"][NameObject("/BaseFont")] = NameObject("/Courier")
    stream = DecodedStreamObject()
    stream.set_data(operations.encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    assert parse_pdf_document(PDFDocumentInput(document_id="positions", content=output.getvalue()))[0].text == expected


def test_heading_drawn_in_reverse_order_preserves_tracking_and_word_boundaries() -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(["placeholder"])))
    page = writer.pages[0]
    page["/Resources"]["/Font"]["/F1"][NameObject("/BaseFont")] = NameObject("/Courier")
    ops = []
    x = 20
    for char in "SUBSIDIARY BRANDS":
        if char != " ":
            ops.append(f"BT /F1 10 Tf {x} 700 Td ({char}) Tj ET")
        x += 12 if char == " " else 8
    stream = DecodedStreamObject()
    stream.set_data(" ".join(reversed(ops)).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    assert (
        parse_pdf_document(PDFDocumentInput(document_id="tracking", content=output.getvalue()))[0].text
        == "SUBSIDIARY BRANDS"
    )


def test_font_variant_name_is_decoded_instead_of_becoming_body_text() -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(["Annual Repor"])))
    page = writer.pages[0]
    font = page["/Resources"]["/Font"]["/F1"]
    font[NameObject("/Encoding")] = DictionaryObject({
        NameObject("/BaseEncoding"): NameObject("/WinAnsiEncoding"),
        NameObject("/Differences"): ArrayObject([NumberObject(20), NameObject("/t.ss04")]),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 10 Tf 20 700 Td (Annual Repor\\024) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    assert (
        parse_pdf_document(PDFDocumentInput(document_id="variant", content=output.getvalue()))[0].text
        == "Annual Report"
    )


def test_form_text_keeps_scoped_fonts_and_excludes_entirely_off_page_runs() -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(["ROOT TEXT"])))
    page = writer.pages[0]
    form = DecodedStreamObject()
    form.update({
        NameObject("/Type"): NameObject("/XObject"),
        NameObject("/Subtype"): NameObject("/Form"),
        NameObject("/BBox"): ArrayObject([NumberObject(n) for n in (0, 0, 600, 800)]),
        NameObject("/Resources"): DictionaryObject({
            NameObject("/Font"): DictionaryObject({
                NameObject("/F1"): DictionaryObject({
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Courier"),
                })
            })
        }),
    })
    form.set_data(b"BT /F1 10 Tf -400 650 Td (HIDDEN ADDRESS) Tj ET BT /F1 10 Tf 20 650 Td (FORM TEXT 42) Tj ET")
    page["/Resources"][NameObject("/XObject")] = DictionaryObject({NameObject("/Form1"): writer._add_object(form)})
    stream = DecodedStreamObject()
    contents = page.get_contents()
    assert contents is not None
    stream.set_data(contents.get_data() + b" q /Form1 Do Q")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    assert (
        parse_pdf_document(PDFDocumentInput(document_id="forms", content=output.getvalue()))[0].text
        == "ROOT TEXT\nFORM TEXT 42"
    )


@pytest.mark.parametrize(
    ("marked", "expected"),
    [
        ("/Span << /ActualText (ti) >> BDC (X) Tj EMC", "certificate"),
        ("/Span << /ActualText <FEFF00740069> >> BDC [(X) (Y)] TJ EMC", "certificate"),
        (
            "/Span << /ActualText (ti) >> BDC /Span << /ActualText (wrong) >> BDC (X) Tj EMC EMC",
            "certificate",
        ),
        ("/Span << /ActualText () >> BDC (X) Tj EMC", "cerficate"),
    ],
)
def test_marked_actual_text_replaces_the_sequence_once(marked: str, expected: str) -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(["placeholder"])))
    page = writer.pages[0]
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 10 Tf 20 700 Td (cer) Tj {marked} (ficate) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    assert (
        parse_pdf_document(PDFDocumentInput(document_id="actual-text", content=output.getvalue()))[0].text == expected
    )


def test_named_actual_text_does_not_inline_unrelated_indirect_metadata() -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(["placeholder"])))
    page = writer.pages[0]
    properties = DictionaryObject({
        NameObject("/ActualText"): TextStringObject("ti"),
        NameObject("/Metadata"): writer._add_object(DictionaryObject()),
    })
    page["/Resources"][NameObject("/Properties")] = DictionaryObject({
        NameObject("/MC0"): writer._add_object(properties)
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 10 Tf 20 700 Td (cer) Tj /Span /MC0 BDC (X) Tj EMC (ficate) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    assert (
        parse_pdf_document(PDFDocumentInput(document_id="named-actual", content=output.getvalue()))[0].text
        == "certificate"
    )


def test_actual_text_on_a_space_is_not_mistaken_for_a_blank_page() -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(["placeholder"])))
    page = writer.pages[0]
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 10 Tf 20 700 Td /Span << /ActualText (Annual Report) >> BDC ( ) Tj EMC ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    assert (
        parse_pdf_document(PDFDocumentInput(document_id="space-actual", content=output.getvalue()))[0].text
        == "Annual Report"
    )


async def test_actual_text_unicode_stays_exact_after_memory_restart(tmp_path: Path) -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(["placeholder"])))
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 10 Tf 20 700 Td /Span << /ActualText <FEFF037E002000630061006600650301> >> BDC (X) Tj EMC ET"
    )
    writer.pages[0][NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    page = parse_pdf_document(PDFDocumentInput(document_id="unicode-pdf", content=output.getvalue()))[0]
    assert page.text == "; café"
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    observation = project_source_for_transport(registry, await registry.resolve(page))
    config, scope, _ = await persist(tmp_path, PDF_PAGE_SOURCE_DEFINITION, [observation])
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope)
        (listed,) = (await memory.list()).entries
        entry = (await memory.get(GetMemoryEntryRequest(citation=listed.citation))).entry
        assert entry.text == "unicode-pdf\n; café"
        assert entry.sources[0].source_id == observation.name
