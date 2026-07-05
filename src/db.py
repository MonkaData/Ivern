"""Couche d'accès SQLite : models / prompts / runs / errors."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .config import DB_PATH

PathLike = Union[str, Path]

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    hf_id TEXT NOT NULL,
    quantization TEXT,
    loaded_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    role TEXT,
    template_text TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT,
    image_path TEXT,
    true_label TEXT,
    model_id INTEGER REFERENCES models(id),
    prompt_id INTEGER REFERENCES prompts(id),
    prompt_kind TEXT,
    raw_output TEXT,
    parsed_json TEXT,
    pred_class TEXT,
    confidence REAL,
    ground_truth TEXT,
    latency_ms INTEGER,
    json_valid INTEGER,
    reason TEXT,
    warning_text TEXT,
    error_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    error_type TEXT,
    comment TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

RUN_COLUMNS = {
    "case_id": "TEXT",
    "image_path": "TEXT",
    "true_label": "TEXT",
    "prompt_kind": "TEXT",
    "raw_output": "TEXT",
    "parsed_json": "TEXT",
    "pred_class": "TEXT",
    "confidence": "REAL",
    "ground_truth": "TEXT",
    "latency_ms": "INTEGER",
    "json_valid": "INTEGER",
    "reason": "TEXT",
    "warning_text": "TEXT",
    "error_message": "TEXT",
}


@contextmanager
def connect(path: PathLike = DB_PATH):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: PathLike = DB_PATH) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        for name, sql_type in RUN_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {sql_type}")


def register_model(name: str, hf_id: str, quantization: str = "4bit-nf4",
                   path: PathLike = DB_PATH) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO models(name, hf_id, quantization) VALUES (?,?,?)",
            (name, hf_id, quantization),
        )
        if cur.lastrowid and cur.rowcount:
            return int(cur.lastrowid)
        row = conn.execute("SELECT id FROM models WHERE name=?", (name,)).fetchone()
        return int(row["id"])


def register_prompt(name: str, version: str, role: str, template: str,
                    path: PathLike = DB_PATH) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO prompts(name, version, role, template_text) VALUES (?,?,?,?)",
            (name, version, role, template),
        )
        if cur.lastrowid and cur.rowcount:
            return int(cur.lastrowid)
        row = conn.execute(
            "SELECT id FROM prompts WHERE name=? AND version=?", (name, version)
        ).fetchone()
        return int(row["id"])


def insert_run(
    *,
    case_id: Optional[str],
    image_path: Optional[str],
    true_label: Optional[str],
    model_id: int,
    prompt_id: int,
    prompt_kind: Optional[str],
    raw_output: str,
    parsed_json: Optional[Dict[str, Any]],
    pred_class: str,
    confidence: Optional[float],
    ground_truth: Optional[str],
    latency_ms: int,
    json_valid: bool,
    reason: Optional[str] = None,
    warning_text: Optional[str] = None,
    error_message: Optional[str] = None,
    db_path: PathLike = DB_PATH,
) -> int:
    parsed_str = json.dumps(parsed_json, ensure_ascii=False) if parsed_json is not None else None
    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO runs (
               case_id, image_path, true_label, model_id, prompt_id, prompt_kind,
               raw_output, parsed_json, pred_class, confidence, ground_truth,
               latency_ms, json_valid, reason, warning_text, error_message
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                case_id,
                image_path,
                true_label,
                model_id,
                prompt_id,
                prompt_kind,
                raw_output,
                parsed_str,
                pred_class,
                confidence,
                ground_truth,
                latency_ms,
                1 if json_valid else 0,
                reason,
                warning_text,
                error_message,
            ),
        )
        return int(cur.lastrowid)


def insert_error(run_id: int, error_type: str, comment: str = "",
                 db_path: PathLike = DB_PATH) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO errors(run_id, error_type, comment) VALUES (?,?,?)",
            (run_id, error_type, comment),
        )
        return int(cur.lastrowid)


def fetch_runs(model_id: Optional[int] = None,
               db_path: PathLike = DB_PATH) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        if model_id is None:
            cur = conn.execute(
                """SELECT r.*, m.name AS model_name FROM runs r
                   LEFT JOIN models m ON r.model_id = m.id ORDER BY r.id"""
            )
        else:
            cur = conn.execute(
                """SELECT r.*, m.name AS model_name FROM runs r
                   LEFT JOIN models m ON r.model_id = m.id
                   WHERE r.model_id = ? ORDER BY r.id""",
                (model_id,),
            )
        return list(cur.fetchall())


def fetch_models(db_path: PathLike = DB_PATH) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(conn.execute("SELECT * FROM models ORDER BY id").fetchall())


def fetch_errors(db_path: PathLike = DB_PATH) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(conn.execute(
            """SELECT e.*, r.case_id, r.pred_class, r.ground_truth, r.error_message FROM errors e
               LEFT JOIN runs r ON e.run_id = r.id ORDER BY e.id DESC"""
        ).fetchall())


if __name__ == "__main__":  # pragma: no cover
    init_db()
    print(f"DB initialisée: {DB_PATH}")
