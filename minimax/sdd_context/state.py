"""Task progress persisted and reused through PowerContext public Memory APIs."""
from __future__ import annotations

import json
from pathlib import Path

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, RememberMemoryRequest, ReviseMemoryEntryRequest, open_builtin_runtime
from powercontext.builtin.scope import ScopeDraft


async def task_state(directory: Path, project: str, task_id: str, update: dict | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{directory.resolve() / 'powercontext.db'}"))
    async with open_builtin_runtime(config) as runtime:
        scope = await runtime.scopes.create(ScopeDraft(title=f"{project}: {task_id}", summary="SDD task progress",
                                                      idempotency_key=f"sdd:{project}:{task_id}"))
        memory = runtime.memory.for_scope(scope.scope_id)
        entries = (await memory.list()).entries
        states = [entry for entry in entries if entry.entry.kind == "sdd-task-state"]
        if len(states) > 1:
            raise ValueError("Ambiguous task state: multiple active state records")
        if update is not None:
            payload = {"project": project, "task_id": task_id, **update}
            if payload["project"] != project or payload["task_id"] != task_id:
                raise ValueError("State cannot change project/task identity")
            text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if states:
                await memory.revise(ReviseMemoryEntryRequest(citation=states[0].citation, kind="sdd-task-state",
                                                            text=text, reason="Task progress updated"))
            else:
                await memory.remember(RememberMemoryRequest(entries=(MemoryEntryInput(kind="sdd-task-state", text=text),)))
            states = [entry for entry in (await memory.list()).entries if entry.entry.kind == "sdd-task-state"]
        if not states:
            return {"scope_id": scope.scope_id, "state": None, "citation": None}
        return {"scope_id": scope.scope_id, "state": json.loads(states[0].entry.text),
                "citation": states[0].citation.model_dump(mode="json")}
