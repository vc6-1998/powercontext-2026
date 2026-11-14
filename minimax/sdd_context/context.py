"""Compose current indexed engineering sources and restored task progress."""
import asyncio
import os
from pathlib import Path

from .corpus import Corpus
from .state import task_state


def build_context(task: dict, state_dir: Path):
    with Corpus(state_dir / "seekdb") as corpus:
        sources = corpus.retrieve(task["project"], task["query"])
        database = {"engine": "seekdb", "version": corpus.version, "queries": corpus.queries}
    progress = asyncio.run(task_state(state_dir, task["project"], task["id"]))
    return {"task_id": task["id"], "project": task["project"], "task": task["description"],
            "sources": sources, "progress": progress,
            "next_actions": progress["state"].get("todo", []) if progress["state"] else ["Implement and verify the requested change"],
            "database": database, "process_id": os.getpid()}
