import json
import pytest
from sdd_context.corpus import Corpus, digest
from sdd_context.minimax_client import apply_edits, parse_edits


def test_final_json_can_be_extracted_without_retaining_thinking():
    content = '<think>private provider text</think>\n```json\n{"files":[{"path":"tasks.py","content":"pass\\n"}]}\n```'
    assert parse_edits(content, ["tasks.py"]) == [{"path": "tasks.py", "content": "pass\n"}]


@pytest.mark.parametrize("path", ["tests/test_deadline.py", "../outside.py", "/tmp/outside.py"])
def test_model_cannot_change_tests_or_unrelated_files(path):
    with pytest.raises(ValueError, match="undeclared"):
        parse_edits(json.dumps({"files": [{"path": path, "content": "pass"}]}), ["tasks.py"])


def test_saved_code_integrity_is_checked_before_any_write(tmp_path):
    (tmp_path / "tasks.py").write_text("old")
    record = {"files": [{"path": "tasks.py", "content": "tampered"}], "file_sha256": {"tasks.py": digest("original")}}
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_edits(record, {"editable_files": ["tasks.py"]}, tmp_path)
    assert (tmp_path / "tasks.py").read_text() == "old"


def test_socket_path_limit_is_reported_before_native_startup(tmp_path):
    with pytest.raises(ValueError, match="socket path is too long"):
        with Corpus(tmp_path / ("x" * 110)):
            pytest.fail("Native database must not start for an invalid Unix socket path")
