import argparse
import asyncio
import json
from pathlib import Path

from .context import build_context
from .corpus import Corpus
from .state import task_state


def main():
    parser = argparse.ArgumentParser(description="SDD engineering context backed by seekdb and PowerContext")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest").add_argument("--project-root", type=Path, required=True)
    commands.add_parser("context")
    commands.add_parser("state")
    commands.add_parser("save").add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.task.read_text())
    if args.command == "ingest":
        with Corpus(args.state_dir / "seekdb") as corpus:
            result = {"documents": corpus.ingest(task["project"], args.project_root), "engine": "seekdb", "version": corpus.version}
    elif args.command == "context":
        result = build_context(task, args.state_dir)
    else:
        update = json.loads(args.input.read_text()) if args.command == "save" else None
        result = asyncio.run(task_state(args.state_dir, task["project"], task["id"], update))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
