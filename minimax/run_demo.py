"""Clean, machine-checked six-condition demo. No key is required to replay a genuine saved call.

Exit 0: all six conditions passed. Exit 2: preparation passed, real model step pending.
Exit 1: an executed condition failed. Preparation never claims competition completion.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv

from sdd_context.corpus import Corpus, digest
from sdd_context.minimax_client import apply_edits, generate_edits
from sdd_context.state import task_state

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hashes(root, pattern="*"):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob(pattern)) if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}


def run_tests(root, paths, output, label):
    xml = output / f"{label}.xml"
    command = [sys.executable, "-m", "pytest", "-q", "-c", str(root / "pytest.ini"),
               "--confcutdir", str(root), "-p", "no:cacheprovider", f"--junitxml={xml}", *paths]
    run = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
    (output / f"{label}.log").write_text(run.stdout + run.stderr)
    if not xml.exists():
        raise AssertionError(f"{label}: test collection/environment failure; see log")
    cases = ET.parse(xml).findall(".//testcase")
    result = {"command": command, "exit_code": run.returncode, "source_sha256": file_hashes(root, "*.py"),
              "tests": len(cases), "failures": sum(c.find("failure") is not None for c in cases),
              "errors": len(ET.parse(xml).findall(".//error")),
              "skipped": sum(c.find("skipped") is not None for c in cases),
              "failed_cases": [{"name": c.attrib["name"], "message": c.find("failure").attrib.get("message", "")}
                               for c in cases if c.find("failure") is not None]}
    save(output / f"{label}.json", result)
    assert result["tests"] and not result["errors"] and not result["skipped"], f"{label}: incomplete functional execution"
    return result


def resumed_context(task_path, state_dir, output):
    command = [sys.executable, "-m", "sdd_context", "--state-dir", str(state_dir), "--task", str(task_path), "context"]
    result = subprocess.run(command, cwd=HERE, capture_output=True, text=True, timeout=120)
    (output / "restart-stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stderr[-2000:]
    # Native seekdb can print informational lines before our JSON.
    begin = result.stdout.find('{\n  "task_id"')
    assert begin >= 0, "CLI did not return context JSON"
    context = json.loads(result.stdout[begin:])
    save(output / "context-after-restart.json", context)
    return context


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "requirements": {}, "official_submission_ready": False}
    try:
        task_path = HERE / "sample/task.json"
        task = json.loads(task_path.read_text())
        before = output / "before"
        shutil.copytree(HERE / "sample/project", before, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        test_hashes = file_hashes(before / "tests")
        before_target = run_tests(before, task["target_tests"], output, "before-target")
        before_regression = run_tests(before, task["regression_tests"], output, "before-regression")
        assert before_target["exit_code"] == 1 and before_target["failures"] > 0
        assert all("assert " in case["message"] for case in before_target["failed_cases"]), "Before failure must be a behavior assertion"
        assert before_regression["exit_code"] == 0, "Existing functionality must work before modification"
        # Embedded seekdb creates Unix sockets below this directory. Keep the
        # native path short regardless of how deeply the artifact directory lives.
        state_dir = Path(tempfile.mkdtemp(prefix="sdd-"))
        report["state_directory"] = str(state_dir)
        print("Baseline verified; importing sources into seekdb", flush=True)
        with Corpus(state_dir / "seekdb") as corpus:
            manifest = corpus.ingest(task["project"], before)
            initial_sources = corpus.retrieve(task["project"], task["query"])
        save(output / "initial-import.json", manifest)
        save(output / "initial-sources.json", initial_sources)
        print("Initial seekdb retrieval verified; updating specifications and state", flush=True)
        initial_state = asyncio.run(task_state(state_dir, task["project"], task["id"]))
        assert initial_state["state"] is None, "Acceptance must start with empty task state"
        # A real specification revision is re-imported; context must stop using revision 1.
        specification = before / "specs/filtering.md"
        old_hash = digest(specification.read_text())
        new_rule = "## FILTER-005: Compound filters\nApply the inclusive deadline and status filters before limit; an eligible task on the cutoff consumes a result slot.\n"
        specification.write_text(specification.read_text().replace("revision 1", "revision 2") + "\n" + new_rule)
        with Corpus(state_dir / "seekdb") as corpus:
            refreshed_manifest = corpus.ingest(task["project"], before)
        save(output / "updated-import.json", refreshed_manifest)
        progress = {"done": ["Imported engineering sources", "Updated specifications to revision 2", "Verified baseline functional failure and passing regressions"],
                    "todo": ["Ask MiniMax to fix tasks.py", "Run unchanged target and regression tests"],
                    "validation": {"target_exit_code": before_target["exit_code"], "regression_exit_code": before_regression["exit_code"]}}
        written = asyncio.run(task_state(state_dir, task["project"], task["id"], progress))
        save(output / "state-write.json", written)
        print("Restarting context module in a fresh process", flush=True)
        context = resumed_context(task_path, state_dir, output)
        assert context["process_id"] != os.getpid(), "Restart test must run in a new process"
        assert context["progress"] == written and context["next_actions"] == progress["todo"]
        sources = {s["path"]: s for s in context["sources"]}
        assert set(task["required_sources"]) <= sources.keys(), "Missing required specs/code/tests in context"
        assert sources["specs/filtering.md"]["sha256"] != old_hash
        assert new_rule.strip() in sources["specs/filtering.md"]["content"]
        assert "revision 1" not in sources["specs/filtering.md"]["content"]
        for path, source in sources.items():
            assert (before / path).is_file() and digest((before / path).read_text()) == source["sha256"]
        save(output / "database-queries.json", context["database"])
        report["requirements"] = {"runnable": "preparation_passed", "minimax_actual_use": "pending",
                                  "context_correct_and_updated": "passed", "seekdb_actual_use": "passed",
                                  "state_reused_after_restart": "passed", "code_change_effective": "pending"}
        report["baseline"] = {"target_failures": before_target["failures"], "regression_tests": before_regression["tests"]}
        print("PREPARATION PASSED: real seekdb, updated sources, real PowerContext restart; model step pending", flush=True)
        if args.prepare_only:
            report["status"] = "awaiting_minimax"
            return 2
        model_dir = output / "model"
        if args.replay:
            # Reproduction uses the actual saved final code; no need for the model to reproduce identical text.
            shutil.copytree(args.replay.resolve(), model_dir)
            record = json.loads((model_dir / "record.json").read_text())
            recorded_request = (model_dir / "request.json").read_text().rstrip("\n")
            assert digest(recorded_request) == record["request_sha256"]
            submitted = json.loads(json.loads(recorded_request)["messages"][-1]["content"])["context"]
            assert submitted["task"] == context["task"]
            assert submitted["sources"] == context["sources"], "Replay sources differ from this clean task"
            assert submitted["progress"]["state"] == context["progress"]["state"], "Replay task state differs"
        else:
            record = generate_edits(task, context, model_dir)
        assert record["provider"] == "MiniMax" and record["live_call"] and record["response_id"]
        after = output / "after"
        shutil.copytree(before, after)
        apply_edits(record, task, after)
        assert file_hashes(after / "tests") == test_hashes == file_hashes(before / "tests"), "Tests changed"
        differences = []
        for path in task["editable_files"]:
            differences.extend(difflib.unified_diff((before/path).read_text().splitlines(True), (after/path).read_text().splitlines(True), fromfile=f"before/{path}", tofile=f"after/{path}"))
        assert differences, "MiniMax did not modify the target code"
        (output / "change.patch").write_text("".join(differences))
        after_target = run_tests(after, task["target_tests"], output, "after-target")
        after_regression = run_tests(after, task["regression_tests"], output, "after-regression")
        assert after_target["exit_code"] == after_regression["exit_code"] == 0, "Candidate functionality/regression failed"
        final_state = {"done": progress["done"] + ["MiniMax produced the code change", "All target and regression tests passed"], "todo": [],
                       "validation": {"target_exit_code": 0, "regression_exit_code": 0, "source_sha256": file_hashes(after, "*.py")}}
        asyncio.run(task_state(state_dir, task["project"], task["id"], final_state))
        assert asyncio.run(task_state(state_dir, task["project"], task["id"]))["state"]["validation"] == final_state["validation"]
        report["requirements"].update(runnable="passed", minimax_actual_use="passed", code_change_effective="passed")
        report.update(status="passed", official_submission_ready=True,
                      test_files_sha256=test_hashes, before_source_sha256=file_hashes(before,"*.py"), after_source_sha256=file_hashes(after,"*.py"))
        print("ALL SIX PUBLISHED CONDITIONS PASSED", flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        save(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--replay", type=Path, help="Directory containing genuine model/request.json and record.json")
    parser.add_argument("--output", type=Path, required=True, help="New directory, never an existing run")
    parser.add_argument("--env-file", type=Path, default=HERE / ".env.local", help="Local credentials file; never copied into evidence")
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    if args.generate and not os.environ.get("MINIMAX_API_KEY", "").strip():
        parser.error("Configure MINIMAX_API_KEY locally before --generate; use --prepare-only without credentials")
    sys.exit(run(args))
