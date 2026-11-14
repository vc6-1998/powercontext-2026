"""One real MiniMax code-edit call with a redacted, replayable development record."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .corpus import digest


def request_payload(task, context, model):
    return {"model": model, "max_completion_tokens": 8192, "stream": False,
            "messages": [{"role": "system", "content": (
                "You are fixing a small Python project. Follow its current specifications and preserve existing behavior. "
                "Return only a JSON object {\"files\":[{\"path\":\"...\",\"content\":\"complete replacement file\"}]}. "
                "Do not modify tests, specifications or any file outside editable_files. Make the smallest correct change. "
                "Do not include reasoning or explanations in the response.")},
                {"role": "user", "content": json.dumps({"editable_files": task["editable_files"], "context": context}, ensure_ascii=False)}]}


def parse_edits(content: str, editable_files: list[str]):
    # Do not retain provider thinking text if an older model embeds it in content.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    if "<think>" in content:
        raise ValueError("Incomplete model response; no code was applied")
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    value = json.loads(content)
    edits = value.get("files")
    if not isinstance(edits, list) or not edits:
        raise ValueError("MiniMax must return a nonempty files list")
    paths = []
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"path", "content"}:
            raise ValueError("Malformed model edit")
        if edit["path"] not in editable_files or not isinstance(edit["content"], str) or not edit["content"].strip():
            raise ValueError("Model attempted an undeclared edit")
        paths.append(edit["path"])
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate edit paths")
    return edits


def generate_edits(task, context, output: Path):
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not key:
        raise ValueError("MINIMAX_API_KEY is not configured; no live MiniMax record can be produced")
    base = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.cn/v1").rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme != "https" or parsed.hostname not in {"api.minimax.cn", "api.minimaxi.com", "api.minimax.io"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Configure an official HTTPS MiniMax API endpoint")
    model = os.environ.get("MINIMAX_MODEL", "MiniMax-M3.1-Flash")
    payload = request_payload(task, context, model)
    output.mkdir(parents=True, exist_ok=True)
    request_text = json.dumps(payload, ensure_ascii=False, indent=2)
    (output / "request.json").write_text(request_text + "\n")
    with httpx.Client(timeout=180, follow_redirects=False) as client:
        response = client.post(base + "/chat/completions", json=payload,
                               headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        body = response.json()
    choice = body["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"MiniMax response did not finish normally: {choice.get('finish_reason')}")
    edits = parse_edits(choice["message"]["content"], task["editable_files"])
    record = {"provider": "MiniMax", "live_call": True, "endpoint": base + "/chat/completions",
              "requested_model": model, "returned_model": body.get("model"), "response_id": body.get("id"),
              "timestamp": datetime.now(timezone.utc).isoformat(), "usage": body.get("usage"),
              "finish_reason": choice["finish_reason"], "request_sha256": digest(request_text),
              "files": edits, "file_sha256": {e["path"]: digest(e["content"]) for e in edits}}
    # Only final code and public call metadata are persisted; credentials and thinking are omitted.
    (output / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def apply_edits(record, task, project: Path):
    edits = parse_edits(json.dumps({"files": record["files"]}), task["editable_files"])
    root = project.resolve()
    for edit in edits:
        path = (root / edit["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Model edit must replace an existing declared project file")
        if digest(edit["content"]) != record["file_sha256"][edit["path"]]:
            raise ValueError("Recorded MiniMax output hash mismatch")
    for edit in edits:
        (root / edit["path"]).write_text(edit["content"], encoding="utf-8")
