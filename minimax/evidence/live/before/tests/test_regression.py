"""Existing status, order, limit, validation and non-mutating behavior."""
from copy import deepcopy
import pytest
from tasks import select_tasks


@pytest.fixture
def tasks():
    return [
        {"id": "b", "status": "open", "due_date": "2026-11-12"},
        {"id": "a", "status": "done", "due_date": "2026-11-10"},
        {"id": "c", "status": "open", "due_date": None},
    ]


def test_status_and_original_order(tasks):
    assert [t["id"] for t in select_tasks(tasks, status="open")] == ["b", "c"]


def test_missing_deadline_with_and_without_cutoff(tasks):
    assert len(select_tasks(tasks)) == 3
    assert [t["id"] for t in select_tasks(tasks, due_on_or_before="2026-11-13")] == ["b", "a"]


def test_limit_applies_after_status_filter(tasks):
    assert select_tasks(tasks, limit=0) == []
    assert [t["id"] for t in select_tasks(tasks, status="done", limit=1)] == ["a"]


def test_selection_does_not_modify_input_or_alias_records(tasks):
    before = deepcopy(tasks)
    selected = select_tasks(tasks)
    selected[0]["id"] = "changed"
    assert tasks == before


@pytest.mark.parametrize("kwargs", [{"limit": -1}, {"limit": True}, {"limit": 1.5}, {"status": "bad"}, {"due_on_or_before": "2026-02-30"}])
def test_invalid_filter_arguments(tasks, kwargs):
    with pytest.raises(ValueError):
        select_tasks(tasks, **kwargs)


def test_empty_collection():
    assert select_tasks([], status="open", due_on_or_before="2026-11-14") == []
