"""Authoritative engineering documents and task-time SQL retrieval in real seekdb."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pylibseekdb
import pymysql


def digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class Corpus:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.instance = None
        self.connection = None
        self.queries = []

    def __enter__(self):
        if len(os.fsencode(self.directory / "run/sql.sock")) >= 108:
            raise ValueError("seekdb Unix socket path is too long; choose a shorter state directory such as /tmp/sdd-demo")
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self.instance = pylibseekdb.open(str(self.directory))
        try:
            self.connection = pymysql.connect(**self.instance.connection_options(), database="test",
                                              charset="utf8mb4", autocommit=True,
                                              connect_timeout=20, read_timeout=30, write_timeout=30,
                                              cursorclass=pymysql.cursors.DictCursor)
            with self.connection.cursor() as cursor:
                cursor.execute("""CREATE TABLE IF NOT EXISTS sdd_documents (
                    project VARCHAR(128) NOT NULL, path VARCHAR(512) NOT NULL,
                    kind VARCHAR(16) NOT NULL, revision CHAR(64) NOT NULL,
                    content LONGTEXT NOT NULL, PRIMARY KEY(project, path)
                ) DEFAULT CHARSET=utf8mb4""")
                cursor.execute("SELECT VERSION() AS version")
                self.version = cursor.fetchone()["version"]
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        try:
            if self.connection:
                self.connection.close()
        finally:
            if self.instance:
                self.instance.close()

    def ingest(self, project: str, root: Path):
        """Atomically replace this project's index, including removed documents."""
        root = root.resolve()
        rows = []
        for path in sorted(root.rglob("*")):
            if path.suffix not in (".py", ".md") or not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Source leaves project root: {relative}")
            content = path.read_text(encoding="utf-8")
            kind = "test" if relative.parts[0] == "tests" else "spec" if path.suffix == ".md" else "code"
            rows.append((project, relative.as_posix(), kind, digest(content), content))
        if not rows:
            raise ValueError("Project contains no supported engineering documents")
        self.connection.begin()  # Explicit BEGIN avoids embedded driver's autocommit handshake ambiguity.
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM sdd_documents WHERE project=%s", (project,))
                cursor.executemany("INSERT INTO sdd_documents(project,path,kind,revision,content) VALUES(%s,%s,%s,%s,%s)", rows)
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        return [{"path": r[1], "kind": r[2], "sha256": r[3]} for r in rows]

    def retrieve(self, project: str, query: str, per_kind: int = 4):
        """Rank indexed text using parameterized SQL; never read source files here.

        A small bounded lexical ranker is sufficient for this engineering sample.
        Underscores are escaped because SQL LIKE treats them as wildcards.
        """
        terms = list(dict.fromkeys(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", query.lower())))[:24]
        if not terms or not 1 <= per_kind <= 20:
            raise ValueError("Supply search terms and per_kind in 1..20")
        patterns = ["%" + term.replace("_", "!_") + "%" for term in terms]
        score = "+".join("CASE WHEN LOWER(content) LIKE %s ESCAPE '!' THEN 1 ELSE 0 END" for _ in terms)
        statement = f"SELECT path,kind,revision,content,({score}) AS score FROM sdd_documents WHERE project=%s AND kind=%s HAVING score>0 ORDER BY score DESC,path LIMIT %s"
        results = []
        for kind in ("spec", "code", "test"):
            parameters = (*patterns, project, kind, per_kind)
            with self.connection.cursor() as cursor:
                cursor.execute(statement, parameters)
                rows = cursor.fetchall()
            self.queries.append({"sql": statement, "parameters": list(parameters),
                                 "returned": [{"path": row["path"], "sha256": row["revision"]} for row in rows]})
            for row in rows:
                results.append({"path": row["path"], "kind": row["kind"], "sha256": row["revision"],
                                "lines": [1, len(row["content"].splitlines())],
                                "score": row["score"], "content": row["content"]})
        return results
