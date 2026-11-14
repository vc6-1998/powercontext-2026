# Copyright (c) 2026 OceanBase.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.

"""Decode page text without treating drawing order or font names as prose."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Iterator
from copy import copy
from itertools import chain, pairwise
from statistics import median
from typing import Any

from pypdf import PageObject
from pypdf._codecs import adobe_glyphs
from pypdf._text_extraction._layout_mode._fixed_width_page import recurse_to_target_op
from pypdf._text_extraction._layout_mode._text_state_manager import TextStateManager
from pypdf._text_extraction._layout_mode._text_state_params import TextStateParams
from pypdf.generic import (
    ArrayObject,
    ByteStringObject,
    ContentStream,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NullObject,
    create_string_object,
)

_LIGATURES = str.maketrans(dict(zip("ﬀﬁﬂﬃﬄﬅﬆ", ("ff", "fi", "fl", "ffi", "ffl", "st", "st"), strict=True)))


def _without_blank_missing_fonts(page: PageObject) -> PageObject:  # noqa: C901
    """Ignore null font resources only when they cannot hide any text glyphs."""
    resources = page.get("/Resources")
    if resources is None:
        return page
    resources = resources.get_object()
    fonts = resources.get("/Font")
    if fonts is None:
        return page
    fonts = fonts.get_object()
    missing = {
        name
        for name, reference in fonts.items()
        if isinstance(reference, NullObject) or (isinstance(reference, IndirectObject) and reference.idnum == 0)
    }
    if not missing:
        return page
    content = page.get_contents()
    operations = []
    current_font = None
    font_stack = []
    for operands, operator in content.operations if content is not None else ():
        if operator == b"q":
            font_stack.append(current_font)
        elif operator == b"Q" and font_stack:
            current_font = font_stack.pop()
        elif operator == b"Tf":
            current_font = operands[0]
            if current_font in missing:
                continue
        if current_font in missing and operator in (b"Tj", b"TJ", b"'", b'"'):
            values = operands[-1] if operator == b"TJ" else operands[-1:]
            if any(isinstance(value, (str, bytes)) and value.strip() for value in values):
                raise ValueError("PDF text uses a missing font")  # noqa: TRY003
            operands = [*operands[:-1], ArrayObject() if operator == b"TJ" else ByteStringObject(b"")]
        operations.append((operands, operator))
    cleaned = copy(page)
    clean_resources = DictionaryObject(resources)
    clean_resources[NameObject("/Font")] = DictionaryObject({k: v for k, v in fonts.items() if k not in missing})
    cleaned[NameObject("/Resources")] = clean_resources
    stream = ContentStream(None, page.pdf)
    stream.operations = operations
    cleaned[NameObject("/Contents")] = stream
    return cleaned


def _font_for_extraction(font: DictionaryObject) -> DictionaryObject:
    """Resolve known glyph variants while keeping original character widths."""
    encoding = font.get("/Encoding")
    if encoding is None:
        return font
    encoding = encoding.get_object()
    if not isinstance(encoding, DictionaryObject):
        return font
    differences = encoding.get("/Differences")
    if differences is None or not isinstance(differences.get_object(), ArrayObject):
        return font
    normalized = ArrayObject()
    for value in differences.get_object():
        # Adobe glyph variants retain the base character (for example /t.ss04).
        # Unknown numeric Type3 glyph names must not be guessed into letters.
        base = str(value).split(".", 1)[0]
        normalized.append(NameObject(base) if isinstance(value, str) and base in adobe_glyphs else value)
    if normalized == differences.get_object():
        return font
    changed = DictionaryObject(font)
    changed_encoding = DictionaryObject(encoding)
    changed_encoding[NameObject("/Differences")] = normalized
    changed[NameObject("/Encoding")] = changed_encoding
    return changed


def _font_aliases(
    resources: DictionaryObject, fonts: DictionaryObject, identities: dict[int, NameObject]
) -> dict[NameObject, NameObject]:
    local_fonts = resources.get("/Font", DictionaryObject()).get_object()
    aliases = {}
    for name, reference in local_fonts.items():
        font = reference.get_object()
        identity = id(font)
        if identity not in identities:
            alias = NameObject(f"/PageFont{len(fonts)}")
            identities[identity] = alias
            fonts[alias] = _font_for_extraction(font)
        aliases[name] = identities[identity]
    return aliases


def _flatten_page(page: PageObject) -> PageObject:  # noqa: C901
    """Expand text-bearing forms with scoped font names and graphics transforms."""
    fonts = DictionaryObject()
    identities: dict[int, NameObject] = {}
    active: set[int] = set()

    def expand(current: PageObject, depth: int) -> list[tuple[list[Any], bytes]]:  # noqa: C901
        if depth > 32:
            raise ValueError("PDF form nesting is too deep")  # noqa: TRY003
        current = _without_blank_missing_fonts(current)
        resources = current.get("/Resources", DictionaryObject()).get_object()
        aliases = _font_aliases(resources, fonts, identities)
        content = current.get_contents()
        if content is None:
            return []
        operations = []
        objects = resources.get("/XObject", DictionaryObject()).get_object()
        for operands, operator in ContentStream(content, page.pdf, "bytes").operations:
            if operator == b"Tf":
                operations.append(([aliases.get(operands[0], operands[0]), *operands[1:]], operator))
            elif operator == b"BDC" and isinstance(operands[1], NameObject):
                properties = resources.get("/Properties", DictionaryObject()).get_object()
                value = properties.get(operands[1])
                if value is not None and "/ActualText" in value.get_object():
                    value = DictionaryObject({NameObject("/ActualText"): value.get_object()["/ActualText"]})
                    operands = [operands[0], value]
                operations.append((operands, operator))
            elif operator == b"Do":
                obj = objects[operands[0]].get_object()
                if obj.get("/Subtype") != "/Form":
                    continue  # Images contribute no extractable text.
                identity = id(obj)
                if identity in active:
                    raise ValueError("PDF contains a cyclic form")  # noqa: TRY003
                active.add(identity)
                nested = copy(current)
                nested[NameObject("/Contents")] = obj
                nested[NameObject("/Resources")] = obj.get("/Resources", resources).get_object()
                operations.append(([], b"q"))
                if "/Matrix" in obj:
                    operations.append((list(obj["/Matrix"]), b"cm"))
                operations.extend(expand(nested, depth + 1))
                operations.append(([], b"Q"))
                active.remove(identity)
            else:
                operations.append((operands, operator))
        return operations

    operations = expand(page, 0)
    flattened = copy(page)
    resources = DictionaryObject(page.get("/Resources", DictionaryObject()).get_object())
    resources[NameObject("/Font")] = fonts
    flattened[NameObject("/Resources")] = resources
    content = ContentStream(None, page.pdf)
    content.operations = operations
    flattened[NameObject("/Contents")] = content
    return flattened


def _positioned_operations(content: ContentStream) -> Iterator[tuple[list[Any], bytes]]:
    """Make every text advance explicit for pypdf's layout interpreter.

    Its TJ interpreter advances only on numeric entries, so adjacent byte
    strings otherwise share an incorrect position. An explicit zero moves
    past each string without changing the PDF's drawing semantics.
    """
    for operands, operator in content.operations:
        if operator in (b"'", b'"'):
            if operator == b'"':
                yield [operands[0]], b"Tw"
                yield [operands[1]], b"Tc"
            yield [], b"T*"
            operands, operator = [operands[-1]], b"Tj"
        if operator == b"Tj":
            yield [[operands[0], 0]], b"TJ"
        elif operator == b"TJ":
            for value in operands[0]:
                yield [[value, 0] if isinstance(value, bytes) else [value]], b"TJ"
        else:
            yield operands, operator


def _actual_text_spans(operations: list[tuple[list[Any], bytes]]) -> list[tuple[int, int, str]]:
    """Locate explicit character replacements in marked-content sequences."""
    spans = []
    stack: list[tuple[int, str | None]] = []
    count = 0
    for operands, operator in operations:
        if operator in (b"BMC", b"BDC"):
            replacement = None
            if operator == b"BDC" and isinstance(operands[1], DictionaryObject):
                value = operands[1].get("/ActualText")
                if isinstance(value, bytes):
                    value = create_string_object(value)
                if isinstance(value, str):
                    replacement = value
            stack.append((count, replacement))
        elif operator == b"EMC" and stack:
            start, replacement = stack.pop()
            if replacement is not None and not any(text is not None for _, text in stack):
                spans.append((start, count, replacement))
        elif operator == b"TJ":
            count += sum(isinstance(value, bytes) for value in operands[0])
    return spans


def _replace_actual_text(runs: list[TextStateParams], spans: list[tuple[int, int, str]]) -> list[TextStateParams]:
    """Replace each marked sequence once, retaining its original painted extent."""
    for start, end, text in sorted(spans, reverse=True):
        if start == end or end > len(runs):
            continue
        original = runs[start:end]
        replacement = copy(original[0])
        replacement.value = text
        replacement.text = text.replace("\u00a0", " ").translate(_LIGATURES)
        replacement.displaced_tx = original[-1].displaced_tx
        # Even an empty replacement retains its advance, so the following
        # letters are not separated by the width of an explicitly deleted glyph.
        runs[start:end] = [replacement]
    return runs


def _run_text(run: TextStateParams) -> str:
    text = run.text
    if any(isinstance(value, str) and "\n" in value for value in run.font.character_map.values()):
        # A mapped separator advances within this drawing, not onto another row.
        # Preserve its word boundary instead of deleting it.
        text = text.replace("\n", " ")
    return text.replace("\u00a0", " ").translate(_LIGATURES)


def _without_fragmented_overprints(runs: list[TextStateParams]) -> list[TextStateParams]:
    """Keep one painted copy when a full span is also drawn character by character."""
    removed = set()
    for span in runs:
        compact = "".join(span.text.split())
        if len(compact) < 2:
            continue
        tolerance = 0.01 * span.font_height
        fragments = sorted(
            (
                run
                for run in runs
                if run is not span
                and len(run.text.strip()) == 1
                and run.font.name == span.font.name
                and abs(run.font_height - span.font_height) <= tolerance
                and abs(run.ty - span.ty) <= tolerance
                and run.tx >= span.tx - tolerance
                and run.displaced_tx <= span.displaced_tx + tolerance
            ),
            key=lambda run: run.tx,
        )
        if (
            len(fragments) > 1
            and "".join(run.text.strip() for run in fragments) == compact
            and abs(fragments[0].tx - span.tx) <= tolerance
            and abs(fragments[-1].displaced_tx - span.displaced_tx) <= tolerance
        ):
            removed.add(id(span))
    return [run for run in runs if id(run) not in removed]


def _render_row(runs: list[TextStateParams]) -> str:
    runs = _without_fragmented_overprints(runs)
    # Explicit spaces establish word boundaries even when their advance is
    # narrower than a nominal space. Ignore padding that overlaps painted text.
    spaces = sorted((run.tx + run.displaced_tx) / 2 for run in runs if run.text.isspace())
    runs = sorted((run for run in runs if not run.text.isspace()), key=lambda run: run.tx)
    gaps = [
        max(0.0, right.tx - left.displaced_tx)
        for left, right in pairwise(runs)
        if len(left.text) == len(right.text) == 1
        and left.text.isalpha()
        and right.text.isalpha()
        and right.tx - left.displaced_tx >= -0.01 * min(left.font_height, right.font_height)
    ]
    # Letter-spaced headings have a regular advance between every character.
    tracking = median(gaps) * 1.6 if len(gaps) >= 3 else 0.0
    pieces = []
    previous = None
    for run in runs:
        text = run.text
        if previous is not None:
            threshold = max(
                0.7 * min(abs(previous.space_tx * previous.transform[0]), abs(run.space_tx * run.transform[0])),
                tracking,
                0.01,
            )
            index = bisect_left(spaces, previous.displaced_tx - 0.001)
            explicit_space = index < len(spaces) and spaces[index] <= run.tx + 0.001
            if (
                (run.tx - previous.displaced_tx > threshold or explicit_space)
                and pieces
                and not pieces[-1][-1:].isspace()
                and not text[:1].isspace()
            ):
                pieces.append(" ")
        pieces.append(text)
        previous = run
    return "".join(pieces)


def _visible_runs(page: PageObject, runs: list[TextStateParams]) -> list[TextStateParams]:
    left, bottom, right, top = map(float, page.cropbox)
    visible = []
    seen = set()
    for run in runs:
        if run.text == "" and isinstance(run.value, bytes):
            continue
        if max(run.tx, run.displaced_tx) < left or min(run.tx, run.displaced_tx) > right:
            continue
        if run.ty + run.font_height < bottom or run.ty > top:
            continue
        key = (run.text, *(round(value, 5) for value in (run.tx, run.ty, run.displaced_tx, run.font_height)))
        if key not in seen:
            seen.add(key)
            visible.append(run)
    return visible


def extract_page_text(page: PageObject) -> str:
    """Preserve decoded words and use painted positions for independent fragments."""
    page = _flatten_page(page)
    plain = (page.extract_text() or "").translate(_LIGATURES)
    content = page.get_contents()
    if content is None:
        return plain
    operations = list(_positioned_operations(ContentStream(content, page.pdf, "bytes")))
    try:
        _, runs = recurse_to_target_op(
            chain(operations, [([], b"Q")]),
            TextStateManager(),
            b"Q",
            page._layout_mode_fonts(),
            False,
        )
    except (ValueError, KeyError, TypeError, ArithmeticError):
        return plain
    for run in runs:
        run.text = _run_text(run)
    if any(run.rotated or not run.font.interpretable for run in runs):
        return plain
    # Before intentional clipping/deduplication, account for all decoded characters.
    # A different content order is allowed; missing content must fall back intact.
    if Counter("".join("".join(run.text.split()) for run in runs)) != Counter("".join(plain.split())):
        return plain
    runs = _replace_actual_text(runs, _actual_text_spans(operations))
    visible = _visible_runs(page, runs)
    # Use effective vertical direction; all fonts on one page normally agree.
    visible.sort(key=lambda run: run.ty if run.flip_vertical else -run.ty)
    rows: list[list[TextStateParams]] = []
    for run in visible:
        if rows and abs(rows[-1][0].ty - run.ty) <= 0.12 * min(rows[-1][0].font_height, run.font_height):
            rows[-1].append(run)
        else:
            rows.append([run])
    return "\n".join(text for row in rows if (text := _render_row(row)).strip()).strip()
