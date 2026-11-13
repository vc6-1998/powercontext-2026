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

import asyncio
import json
from urllib.parse import quote

import pytest

from powercontext.sources import SourceDefinitionRegistry, SourceMaterialization, TextEvidence
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


@pytest.mark.parametrize(
    "content",
    [
        b"\n",
        b"\xef\xbb\xbf",
        b"\nid,name\n1,one\n",
        b'id,name\n1,valid\n2,"bad"junk\n',
        b"id,name\n1,valid\n2,two,extra\n",
        b"id,name\n1,valid\n   ,blank\n",
        b'id\n""\n',
        b"id,name\n1,valid\n\xff,invalid\n",
    ],
)
def test_invalid_later_records_and_empty_headers_reject_whole_document(content: bytes) -> None:
    with pytest.raises(ValueError):
        parse_csv_document(CSVDocumentInput(document_id="table", primary_key="id", content=content))


def test_unicode_percent_encoding_and_exact_encoded_identity_limit() -> None:
    assert csv_row_source_id("机场:%", " /一 ") == f"csv:{quote('机场:%', safe='')}:{quote(' /一 ', safe='')}"
    identity = csv_row_source_id("d", "x" * 250)
    assert len(identity) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("d", "x" * 251)
    with pytest.raises(ValueError):
        parse_csv_document(CSVDocumentInput(document_id="d", primary_key="id", content=b"id\n" + b"x" * 251))
    assert csv_row_source_id("a:b", "c") != csv_row_source_id("a", "b:c")


def test_exact_spaced_column_names_empty_values_and_unicode_survive_source_readback() -> None:
    rows = parse_csv_document(
        CSVDocumentInput(
            document_id="资料",
            primary_key=" id ",
            content=" id ,id,name,empty\n 001 ,other,机场,\n".encode(),
        )
    )
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(rows[0]))
    assert source.materialization is SourceMaterialization.CAPTURED
    assert asyncio.run(registry.read(source)) == rows[0]
    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.content == '{" id ":" 001 ","id":"other","name":"机场","empty":""}'
    assert list(json.loads(evidence.content)) == [" id ", "id", "name", "empty"]
    assert evidence.metadata["columns"] == [" id ", "id", "name", "empty"]
    assert evidence.source_id == csv_row_source_id("资料", " 001 ")


@pytest.mark.parametrize(
    "changes",
    [
        {"document_id": " "},
        {"primary_key": "missing"},
        {"primary_key_value": "mismatch"},
        {"columns": ("id", "id")},
        {"columns": ("id", " ")},
        {"values": ("1",)},
    ],
)
def test_direct_row_inputs_obey_same_schema_and_identity_contract(changes: dict[str, object]) -> None:
    value = {
        "document_id": "table",
        "primary_key": "id",
        "primary_key_value": "1",
        "columns": ("id", "value"),
        "values": ("1", "raw"),
    }
    with pytest.raises(ValueError):
        CSVRowInput.model_validate(value | changes)


def test_captured_source_json_roundtrip_keeps_its_exact_payload_and_identity() -> None:
    row = CSVRowInput(
        document_id="table", primary_key="id", primary_key_value="1", columns=("id", "value"), values=("1", "01.00")
    )
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(row))
    restored = CSVRowSource.model_validate_json(source.model_dump_json())
    assert asyncio.run(registry.read(restored)) == row
    assert registry.project(restored, TEXT_EVIDENCE_PROJECTION_KEY) == registry.project(
        source, TEXT_EVIDENCE_PROJECTION_KEY
    )
    with pytest.raises(ValueError):
        CSVRowSource.model_validate(source.model_dump() | {"source_id": "wrong"})
    with pytest.raises(ValueError):
        CSVRowSource.model_validate(source.model_dump() | {"materialization": "referenced"})
