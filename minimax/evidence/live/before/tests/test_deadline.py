"""Target tests: the inclusive deadline boundary is the requested behavior."""
import pytest
from tasks import select_tasks


@pytest.mark.parametrize("cutoff", ["2026-11-14", "2024-02-29", "2026-12-31"])
def test_deadline_includes_the_cutoff_date(cutoff):
    tasks = [
        {"id": "earlier", "status": "open", "due_date": "2024-01-01"},
        {"id": "boundary", "status": "open", "due_date": cutoff},
        {"id": "later", "status": "open", "due_date": "2027-01-01"},
        {"id": "undated", "status": "open", "due_date": None},
    ]
    assert [t["id"] for t in select_tasks(tasks, due_on_or_before=cutoff)] == ["earlier", "boundary"]


def test_inclusive_deadline_combines_with_status_and_limit_in_original_order():
    tasks = [
        {"id": "boundary", "status": "open", "due_date": "2026-11-14"},
        {"id": "done", "status": "done", "due_date": "2026-11-13"},
        {"id": "earlier", "status": "open", "due_date": "2026-11-12"},
    ]
    assert [t["id"] for t in select_tasks(tasks, status="open", due_on_or_before="2026-11-14", limit=1)] == ["boundary"]
