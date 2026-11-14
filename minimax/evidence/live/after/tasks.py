"""Taskboard filtering library. Dates are ISO calendar dates (YYYY-MM-DD)."""

from datetime import date


def select_tasks(tasks, *, status=None, due_on_or_before=None, limit=None):
    """Select tasks by status and deadline without changing the caller's data."""
    if status not in (None, "open", "done"):
        raise ValueError("status must be open or done")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be a nonnegative integer")
    cutoff = date.fromisoformat(due_on_or_before) if due_on_or_before is not None else None
    selected = []
    for task in tasks:
        if status is not None and task["status"] != status:
            continue
        if cutoff is not None:
            if task.get("due_date") is None:
                continue
            if not date.fromisoformat(task["due_date"]) <= cutoff:
                continue
        selected.append(dict(task))
    return selected if limit is None else selected[:limit]
