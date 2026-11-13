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

"""Portable, verified Context Packs and prefix-reusing wire deltas."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, field

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ContextPackError(ValueError):
    """A malformed Context Pack, delta, or UTF-8 chunk boundary."""


def _json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _encoded_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _validate_items(items: tuple[bytes, ...]) -> None:
    if not isinstance(items, tuple) or any(not isinstance(item, bytes) for item in items):
        raise ContextPackError("items must be a tuple of UTF-8 bytes")  # noqa: TRY003
    try:
        for item in items:
            item.decode("utf-8")
    except UnicodeError as error:
        raise ContextPackError("each item must be valid UTF-8") from error  # noqa: TRY003


def _decode_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("wire items must be a Base64 string array")  # noqa: TRY003
    decoded: list[bytes] = []
    for item in value:
        if not isinstance(item, str):
            raise ContextPackError("wire items must be Base64 strings")  # noqa: TRY003
        try:
            raw = base64.b64decode(item, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ContextPackError("invalid Base64 item") from error  # noqa: TRY003
        if base64.b64encode(raw).decode("ascii") != item:
            raise ContextPackError("Base64 items must use standard canonical encoding")  # noqa: TRY003
        decoded.append(raw)
    return tuple(decoded)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContextPackError("duplicate JSON field")  # noqa: TRY003
        result[key] = value
    return result


def _decode_wire(data: bytes, *, schema: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise ContextPackError("wire data must be UTF-8 bytes")  # noqa: TRY003
    try:
        payload = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_fields)
    except (UnicodeError, ValueError) as error:
        raise ContextPackError("invalid Context Pack JSON") from error  # noqa: TRY003
    if not isinstance(payload, dict) or set(payload) != fields or payload["schema"] != schema:
        raise ContextPackError("invalid Context Pack schema or fields")  # noqa: TRY003
    return payload


def _digest(value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContextPackError("invalid SHA-256 digest")  # noqa: TRY003
    return value


def _prefix(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
    return value


@dataclass(frozen=True, slots=True)
class ContextPack:
    """An immutable ordered UTF-8 chunk sequence with self-contained wire checks."""

    items: tuple[bytes, ...]
    digest: str = field(init=False)
    byte_count: int = field(init=False)
    _wire: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_items(self.items)
        unsigned: dict[str, object] = {"schema": _PACK_SCHEMA, "items": _encoded_items(self.items)}
        digest = "sha256:" + hashlib.sha256(_json_bytes(unsigned)).hexdigest()
        payload = {**unsigned, "digest": digest, "byte_count": 0}
        # The decimal count contributes to its own length; converge at digit boundaries.
        while True:
            wire = _json_bytes(payload)
            if payload["byte_count"] == len(wire):
                break
            payload["byte_count"] = len(wire)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "byte_count", len(wire))
        object.__setattr__(self, "_wire", wire)

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        """Build a complete pack without changing chunk contents or boundaries."""

        return cls(items)

    @classmethod
    def from_content(cls, content: str | None, *, chunk_bytes: int = 512) -> ContextPack:
        """Greedily split original UTF-8 bytes without cutting a code point."""

        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        if content is not None and not isinstance(content, str):
            raise ContextPackError("content must be text or None")  # noqa: TRY003
        try:
            raw = (content or "").encode("utf-8")
        except UnicodeError as error:
            raise ContextPackError("content must be valid UTF-8") from error  # noqa: TRY003
        items: list[bytes] = []
        start = 0
        while start < len(raw):
            end = min(start + chunk_bytes, len(raw))
            while end < len(raw) and end > start and raw[end] & 0xC0 == 0x80:
                end -= 1
            if end == start:
                raise ContextPackError("chunk_bytes cannot fit a UTF-8 code point")  # noqa: TRY003
            items.append(raw[start:end])
            start = end
        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON including the exact wire byte count."""

        return self._wire

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and independently verify a complete serialized pack."""

        payload = _decode_wire(data, schema=_PACK_SCHEMA, fields={"schema", "items", "digest", "byte_count"})
        pack = cls.build(_decode_items(payload["items"]))
        if _digest(payload["digest"]) != pack.digest:
            raise ContextPackError("pack digest mismatch")  # noqa: TRY003
        count = payload["byte_count"]
        if type(count) is not int or count != pack.byte_count:
            raise ContextPackError("pack byte_count mismatch")  # noqa: TRY003
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A base digest, retained prefix length, and complete replacement suffix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _digest(self.base_digest)
        _digest(self.target_digest)
        _prefix(self.prefix_items)
        _validate_items(self.appended)

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Reuse the longest identical chunk prefix of two complete packs."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("base and target must be Context Packs")  # noqa: TRY003
        prefix = 0
        for old, new in zip(base.items, target.items, strict=False):
            if old != new:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        """Include all serialized protocol overhead in the transmission cost."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Encode only the interoperable v1 delta fields."""

        return _json_bytes({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encoded_items(self.appended),
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode a delta without needing sender state or the base pack."""

        payload = _decode_wire(
            data,
            schema=_DELTA_SCHEMA,
            fields={"schema", "base_digest", "target_digest", "prefix_items", "appended"},
        )
        return cls(
            _digest(payload["base_digest"]),
            _digest(payload["target_digest"]),
            _prefix(payload["prefix_items"]),
            _decode_items(payload["appended"]),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Reconstruct and verify a complete target using only the base and delta."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("apply requires a Context Pack and delta")  # noqa: TRY003
    if base.digest != delta.base_digest:
        raise ContextPackError("delta base digest mismatch")  # noqa: TRY003
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base item count")  # noqa: TRY003
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest mismatch")  # noqa: TRY003
    return target


__all__ = ["ContextPack", "ContextPackDelta", "ContextPackError", "apply_context_pack_delta"]
