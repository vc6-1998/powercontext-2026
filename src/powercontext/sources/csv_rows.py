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

"""Capture CSV rows as exact, independently addressable text evidence."""

from __future__ import annotations

import csv
import io
import json
from typing import Literal, Self
from urllib.parse import quote

from pydantic import BaseModel, JsonValue, field_validator, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


def _validate_document_id(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("document_id must be non-empty without surrounding whitespace")  # noqa: TRY003
    return value


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Identify a logical row without depending on its order or other fields."""
    _validate_document_id(document_id)
    if not primary_key_value.strip():
        raise ValueError("primary key value must not be blank")  # noqa: TRY003
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("encoded CSV row identity exceeds the source ID limit")  # noqa: TRY003
    return source_id


def _validate_columns(columns: tuple[str, ...], primary_key: str) -> None:
    if not columns or any(not column.strip() for column in columns) or len(set(columns)) != len(columns):
        raise ValueError("CSV columns must be non-blank and unique")  # noqa: TRY003
    if not primary_key or primary_key not in columns:
        raise ValueError("primary_key must exactly match a CSV column")  # noqa: TRY003


class CSVDocumentInput(BaseModel):
    """Complete offline CSV bytes and the exact identity column to use."""

    document_id: str
    primary_key: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_document_id(value)


class CSVRowInput(BaseModel):
    """One validated record retaining the original column order and strings."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_row(self) -> Self:
        _validate_columns(self.columns, self.primary_key)
        if len(self.values) != len(self.columns):
            raise ValueError("CSV row width must match its columns")  # noqa: TRY003
        if self.values[self.columns.index(self.primary_key)] != self.primary_key_value:
            raise ValueError("primary_key_value must equal the original row field")  # noqa: TRY003
        csv_row_source_id(self.document_id, self.primary_key_value)
        return self


class CSVRowSource(Source, CSVRowInput):
    """A captured row that remains readable without the original CSV file."""

    source_id: str
    definition_version: Literal["1"] = "1"
    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        identity = csv_row_source_id(self.document_id, self.primary_key_value)
        if self.name != identity or self.source_id != identity:
            raise ValueError("CSV Source name and source_id must match its row identity")  # noqa: TRY003
        return self


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Validate the complete document before exposing any captured rows."""
    # newline="" preserves physical newline bytes inside quoted fields.
    reader = csv.reader(io.StringIO(value.content.decode("utf-8-sig"), newline=""), strict=True)
    try:
        columns = tuple(next(reader, ()))
        _validate_columns(columns, value.primary_key)
        key_index = columns.index(value.primary_key)
        seen: set[str] = set()
        rows: list[CSVRowInput] = []
        for record in reader:
            if not record:
                continue
            if len(record) != len(columns):
                raise ValueError("CSV row width must match its header")  # noqa: TRY003
            primary_key_value = record[key_index]
            if primary_key_value in seen:
                raise ValueError("CSV primary keys must be unique within a document")  # noqa: TRY003
            row = CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
            seen.add(primary_key_value)
            rows.append(row)
    except csv.Error as error:
        raise ValueError(str(error)) from error
    return tuple(rows)


class _CSVRowAdapter:
    input_class = CSVRowInput
    name = "csv-row"
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(name=identity, source_id=identity, **value.model_dump())

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class _CSVRowTextProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row: dict[str, JsonValue] = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    _CSVRowAdapter(), projections=(_CSVRowTextProjection(),)
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
