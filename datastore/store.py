"""The qaari persistent store: one DuckDB file plus large tensors on disk next to it.

    from datastore.store import connect, run
    con = connect()                      # creates ~/qaari-store/qaari.duckdb with the schema if needed
    with run(con, "quran_phonetic:hafs_m4", "loader", {"moshaf": "hafs_m4"}):
        ...

``QAARI_STORE`` overrides the directory. Julia reads the same file with DuckDB.jl; Octave gets TSV
exports. Results that are worth computing once belong here, not in scratch space.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb

STORE_DIR = Path(os.environ.get("QAARI_STORE", Path.home() / "qaari-store"))
DB_PATH = STORE_DIR / "qaari.duckdb"
SCHEMA = Path(__file__).with_name("schema.sql")


def connect(path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    p = Path(path) if path else DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(p), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA.read_text())
    return con


@contextlib.contextmanager
def run(con: duckdb.DuckDBPyConnection, run_id: str, kind: str, params: dict[str, Any] | None = None,
        note: str = "", cost_usd: float | None = None) -> Iterator[str]:
    """Record a run (re-running the same id replaces its row) and stamp its finish time."""
    con.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
    con.execute("INSERT INTO runs (run_id, kind, params, note, cost_usd) VALUES (?, ?, ?, ?, ?)",
                [run_id, kind, json.dumps(params or {}, ensure_ascii=False), note, cost_usd])
    yield run_id
    con.execute("UPDATE runs SET finished_at = current_timestamp WHERE run_id = ?", [run_id])


def tensor_dir(*parts: str) -> Path:
    """Directory under the store for large files (posterior dumps, exports)."""
    d = STORE_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
