# Taskboard filtering specification — revision 1

## FILTER-001: Deadline boundary
`select_tasks(..., due_on_or_before="YYYY-MM-DD")` returns tasks whose deadline
is on or before the cutoff. The boundary is inclusive. A task due exactly on
the cutoff date must be included; a task due later must be excluded.
Parse calendar dates, not string prefixes. This also applies on leap days.

## FILTER-002: Missing deadlines
Tasks with no deadline are excluded when a cutoff is supplied. Without a cutoff,
undated tasks remain eligible. Invalid date values raise ValueError.

## FILTER-003: Compatibility
`status` is optional and accepts `open` or `done`; any other value raises ValueError.
`limit` is optional and is applied after filtering. Zero means an empty result;
negative values, booleans and non-integers are rejected. Input records must not
be modified, and returned records must be independent shallow copies.

## FILTER-004: Result ordering
Preserve input order; do not sort the result by deadline.
