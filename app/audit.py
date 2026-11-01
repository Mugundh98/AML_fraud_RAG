"""Append-only, tamper-evident audit log (SQLite).

Every entry stores the SHA-256 of its own content plus the previous entry's hash, a hash chain. Editing or
deleting an old row breaks every hash after it, which verify() reports. That makes tampering visible, not
impossible: someone who can rewrite the whole file can rebuild the chain. (Real systems also ship the
latest hash somewhere else, e.g. to WORM storage or a ticket, so the chain can be checked against it.)

Unlike Langfuse traces, the audit log keeps real names: it's the bank's own record of who did what.
"""

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    event TEXT NOT NULL,
    session_id TEXT,
    trace_id TEXT,
    details TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
)
"""

FIELDS = ("ts", "actor_id", "actor_role", "event", "session_id", "trace_id", "details")


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute(SCHEMA)

    def record(self, *, actor_id: str, actor_role: str, event: str, details: dict[str, Any],
               session_id: str | None = None, trace_id: str | None = None) -> dict:
        with self._lock, self._connect() as db:
            last = db.execute("SELECT hash FROM audit_events ORDER BY id DESC LIMIT 1").fetchone()
            row = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "actor_id": actor_id, "actor_role": actor_role, "event": event,
                "session_id": session_id, "trace_id": trace_id,
                "details": json.dumps(details, sort_keys=True, default=str),
                "prev_hash": last[0] if last else GENESIS,
            }
            row["hash"] = _hash(row)
            cursor = db.execute(
                f"INSERT INTO audit_events ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                tuple(row.values()),
            )
            return {"id": cursor.lastrowid, **row}

    def recent(self, limit: int = 50, event: str | None = None) -> list[dict]:
        query = "SELECT * FROM audit_events" + (" WHERE event = ?" if event else "") + " ORDER BY id DESC LIMIT ?"
        with self._connect() as db:
            rows = db.execute(query, (event, limit) if event else (limit,)).fetchall()
        return [{**dict(r), "details": json.loads(r["details"])} for r in rows]

    def verify(self) -> dict:
        """Recompute the chain. Reports the first entry whose hash or link doesn't match."""
        previous, count = GENESIS, 0
        with self._connect() as db:
            for r in db.execute("SELECT * FROM audit_events ORDER BY id"):
                row = dict(r)
                if row["prev_hash"] != previous:
                    return {"ok": False, "entries": count, "broken_at": row["id"], "reason": "link to previous entry"}
                if _hash(row) != row["hash"]:
                    return {"ok": False, "entries": count, "broken_at": row["id"], "reason": "content changed"}
                previous, count = row["hash"], count + 1
        return {"ok": True, "entries": count, "latest_hash": previous}

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db


def _hash(row: dict) -> str:
    content = json.dumps({k: row[k] for k in (*FIELDS, "prev_hash")}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()
