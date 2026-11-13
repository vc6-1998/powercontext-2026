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

"""Capture complete raw Pi v3 branch history with durable incremental progress."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from powercontext.errors import SourceConflictError
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PiSessionInput(BaseModel):
    """Offline session bytes and the selected branch endpoint."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible raw event, including fields not present in its text view."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @field_validator("session_id", "event_id", "event_type", "role")
    @classmethod
    def require_identity(cls, value: str) -> str:
        if not value:
            raise ValueError("Pi event identity and role must not be empty")  # noqa: TRY003
        return value

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if (
            self.record.get("id") != self.event_id
            or "parentId" not in self.record
            or self.record["parentId"] != self.parent_id
            or self.record.get("type") != self.event_type
        ):
            raise ValueError("Pi event identity must agree with its complete original record")  # noqa: TRY003
        return self


def _source_id(session_id: str, event_id: str) -> str:
    # JSON array framing prevents delimiter ambiguity; hashing also bounds long IDs.
    identity = json.dumps([session_id, event_id], ensure_ascii=False, separators=(",", ":"))
    return "pi:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant in Pi archive: {value}")  # noqa: TRY003


def _load_archive(content: bytes) -> tuple[str, dict[str, dict[str, JsonValue]]]:
    lines = (line for line in io.StringIO(content.decode("utf-8-sig"), newline=None) if line.strip())
    header = json.loads(next(lines, "null"), parse_constant=_reject_constant)
    if (
        not isinstance(header, dict)
        or header.get("type") != "session"
        or header.get("version") != 3
        or not isinstance(header.get("id"), str)
        or not header["id"]
    ):
        raise ValueError("Pi archive must start with a version 3 session header")  # noqa: TRY003
    records: dict[str, dict[str, JsonValue]] = {}
    for line in lines:
        record = json.loads(line, parse_constant=_reject_constant)
        if not isinstance(record, dict):
            raise ValueError("Pi session entries must be JSON objects")  # noqa: TRY003, TRY004
        event_id = record.get("id")
        event_type = record.get("type")
        if not isinstance(event_id, str) or not event_id or event_id in records:
            raise ValueError("Pi session entry IDs must be non-empty and unique")  # noqa: TRY003
        if not isinstance(event_type, str) or not event_type or event_type == "session":
            raise ValueError("Pi session entry type must identify an event or control node")  # noqa: TRY003
        if "parentId" not in record:
            raise ValueError("Pi session entries must declare parentId")  # noqa: TRY003
        parent = record["parentId"]
        if parent is not None and (not isinstance(parent, str) or parent not in records):
            raise ValueError("Pi parent must be null or an earlier session entry")  # noqa: TRY003
        records[event_id] = record
    return header["id"], records


def _required_text(value: JsonValue, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Pi visible event requires a non-empty {field}")  # noqa: TRY003
    return value


def _text(value: JsonValue, *, tool_calls: bool = False) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if not isinstance(value, list):
        raise ValueError("Pi content must be a string or a list of content blocks")  # noqa: TRY003, TRY004
    parts: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            raise ValueError("Pi content blocks must be JSON objects")  # noqa: TRY003, TRY004
        if block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("Pi text block must contain a string")  # noqa: TRY003
            parts.append(text)
        elif tool_calls and block.get("type") == "toolCall":
            name = _required_text(block.get("name"), "tool name")
            call_id = _required_text(block.get("id"), "tool call ID")
            args = json.dumps(
                block.get("arguments", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            parts.append(f"[tool_call {name} {call_id}] {args}")
    return "\n".join(parts)


def _message_body(message: dict[str, JsonValue]) -> tuple[str, str]:
    role = _required_text(message.get("role"), "message role")
    if role == "bashExecution":
        command = message.get("command", "")
        output = message.get("output", "")
        if not isinstance(command, str) or not isinstance(output, str):
            raise ValueError("Pi bash command and output must be strings")  # noqa: TRY003
        if not command.strip() and not output.strip():
            return role, ""
        body = f"$ {command}\n{output}"
    else:
        body = _text(message.get("content"), tool_calls=role != "toolResult")
        if role == "toolResult" and body.strip():
            name = _required_text(message.get("toolName"), "tool name")
            call_id = _required_text(message.get("toolCallId"), "tool call ID")
            body = f"[tool_result {name} {call_id}] {body}"
    return role, body


def _event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_type = record["type"]
    role: str
    body: str
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            raise ValueError("Pi message entry requires a message object")  # noqa: TRY003
        role, body = _message_body(message)
    elif event_type in {"compaction", "branch_summary"}:
        role = str(event_type)
        summary = record.get("summary", "")
        if not isinstance(summary, str):
            raise ValueError("Pi summary must be a string")  # noqa: TRY003
        body = summary
    elif event_type == "custom_message":
        role = "custom"
        body = _text(record.get("content"))
    else:
        return None
    if not body.strip():
        return None
    return PiEventInput.model_validate({
        "session_id": session_id,
        "event_id": record["id"],
        "parent_id": record["parentId"],
        "event_type": event_type,
        "role": role,
        "content": f"{role}: {body}",
        "record": record,
    })


def _branch_events(
    session_id: str, records: dict[str, dict[str, JsonValue]], leaf_id: str | None
) -> tuple[PiEventInput, ...]:
    current = leaf_id if leaf_id is not None else next(reversed(records), None)
    if current is not None and current not in records:
        raise ValueError("selected Pi leaf does not exist")  # noqa: TRY003
    chain: list[dict[str, JsonValue]] = []
    while current is not None:
        record = records[current]
        chain.append(record)
        parent = record["parentId"]
        current = parent if isinstance(parent, str) else None
    events = (_event(session_id, record) for record in reversed(chain))
    return tuple(event for event in events if event is not None)


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse a whole archive, then expose only the selected raw ancestor path."""
    session_id, records = _load_archive(value.content)
    return _branch_events(session_id, records, value.leaf_id)


class PiEventSource(Source, PiEventInput):
    """Captured original event whose identity is independent of its payload."""

    source_id: str
    definition_version: Literal["1"] = "1"
    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED

    @model_validator(mode="after")
    def validate_source_identity(self) -> Self:
        identity = _source_id(self.session_id, self.event_id)
        if self.source_id != identity or self.name != identity:
            raise ValueError("Pi Source identity must match its session and event")  # noqa: TRY003
        return self


class _PiEventAdapter:
    input_class = PiEventInput
    name = "pi-event"
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = _source_id(value.session_id, value.event_id)
        return PiEventSource(name=identity, source_id=identity, **value.model_dump())

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput.model_validate(source.model_dump())


class _PiEventTextProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pi-event",
            source_id=source.name,
            content=source.content,
            metadata={
                key: getattr(source, key) for key in ("session_id", "event_id", "parent_id", "event_type", "role")
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    _PiEventAdapter(), projections=(_PiEventTextProjection(),)
)


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    sessions: dict[str, dict[str, str]] = Field(default_factory=dict)


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class PiSessionConnector:
    """Import new branch events; let the lifecycle commit progress after acceptance."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        session_id, records = _load_archive(self.path.read_bytes())
        events = _branch_events(session_id, records, self.leaf_id)
        state = _Checkpoint() if session.checkpoint is None else _Checkpoint.model_validate(session.checkpoint)
        imported = state.sessions.setdefault(session_id, {})
        # Validate every previously captured event, including events on other branches,
        # before any new submission can be accepted or any checkpoint can advance.
        for event_id, digest in imported.items():
            if event_id not in records or _record_digest(records[event_id]) != digest:
                raise ValueError("previously imported Pi event was removed or rewritten")  # noqa: TRY003
        for event in events:
            if event.event_id in imported:
                continue
            try:
                outcome = await session.submit(_source_id(session_id, event.event_id), "pi-event", event)
            except SourceConflictError as error:
                raise ValueError("Pi event conflicts with previously captured raw evidence") from error  # noqa: TRY003
            if outcome.status is ConnectorSubmissionStatus.ACCEPTED:
                imported[event.event_id] = _record_digest(event.record)
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=state.model_dump(mode="json"))


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
