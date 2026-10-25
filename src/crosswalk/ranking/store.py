"""History of indicator windows and leaderboard snapshots, in SQLite.

Why SQLite and not Postgres: the ranking service is the only writer (one
replica consuming one topic), the data is small (one row per camera per
minute of footage, a few MB per year of video), and every query is a scan of
one table. SQLite on the service's PersistentVolumeClaim gives durable,
transactional storage with no extra Deployment, credentials, backups or
network hop. What it cannot do is accept writes from several pods, so the
ranking Deployment is pinned to one replica with the Recreate strategy (a
second writer would corrupt nothing, since SQLite locks, but would fight for
the lock and duplicate consumption). If the leaderboard ever needs several
writers or replicas, move to Postgres; the store is a small class, so that
is a contained change.

WAL journal mode lets the HTTP threads read while the consumer writes.

Idempotency: windows are upserted by ``window_id``. A Kafka redelivery or a
replayed clip pass with the same stream id overwrites the same row instead
of counting twice.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from crosswalk.stream.events import IndicatorEvent

SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    window_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    space TEXT NOT NULL,
    window_start_s REAL NOT NULL,
    window_end_s REAL NOT NULL,
    observed_s REAL NOT NULL,
    frames INTEGER NOT NULL,
    pedestrians INTEGER NOT NULL,
    vehicles INTEGER NOT NULL,
    pedestrian_seconds REAL NOT NULL,
    vehicle_seconds REAL NOT NULL,
    close_approaches INTEGER NOT NULL,
    min_separation REAL,
    path_crossings INTEGER NOT NULL,
    ttc_events INTEGER NOT NULL,
    min_ttc_s REAL,
    pet_events INTEGER NOT NULL,
    min_pet_s REAL,
    received_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS windows_camera ON windows (camera_id, received_at);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    leaderboard TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

COLUMNS = [
    "window_id", "camera_id", "stream_id", "space", "window_start_s", "window_end_s", "observed_s",
    "frames", "pedestrians", "vehicles", "pedestrian_seconds", "vehicle_seconds", "close_approaches",
    "min_separation", "path_crossings", "ttc_events", "min_ttc_s", "pet_events", "min_pet_s",
]  # fmt: skip


class Store:
    def __init__(self, path: str | Path, clock=time.time):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by threads, serialised by a lock: simpler
        # than a pool, and SQLite serialises writes anyway.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.clock = clock
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)

    def upsert_windows(self, events: list[IndicatorEvent]) -> int:
        if not events:
            return 0
        now = self.clock()
        rows = [(*(getattr(e, c) for c in COLUMNS), now) for e in events]
        placeholders = ",".join("?" * (len(COLUMNS) + 1))
        updates = ",".join(f"{c}=excluded.{c}" for c in COLUMNS[1:])
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(
                    f"INSERT INTO windows ({','.join(COLUMNS)}, received_at) VALUES ({placeholders}) "
                    f"ON CONFLICT(window_id) DO UPDATE SET {updates}, received_at=excluded.received_at",
                    rows,
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return len(rows)

    def windows(self, since: float | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM windows"
        args: tuple = ()
        if since is not None:
            query += " WHERE received_at >= ?"
            args = (since,)
        with self._lock:
            return [
                dict(r)
                for r in self._conn.execute(query + " ORDER BY camera_id, window_start_s", args)
            ]

    def windows_for(self, camera_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(r)
                for r in self._conn.execute(
                    "SELECT * FROM windows WHERE camera_id = ? ORDER BY received_at, window_start_s",
                    (camera_id,),
                )
            ]

    def save_snapshot(self, leaderboard: dict[str, Any], keep: int = 500) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO snapshots (created_at, leaderboard) VALUES (?, ?)",
                (self.clock(), json.dumps(leaderboard, separators=(",", ":"))),
            )
            self._conn.execute(
                "DELETE FROM snapshots WHERE id NOT IN (SELECT id FROM snapshots ORDER BY id DESC LIMIT ?)",
                (keep,),
            )
            return int(cur.lastrowid)

    def snapshot_before(self, timestamp: float) -> dict[str, Any] | None:
        """The newest snapshot taken at or before ``timestamp``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT leaderboard FROM snapshots WHERE created_at <= ? ORDER BY id DESC LIMIT 1",
                (timestamp,),
            ).fetchone()
        return json.loads(row["leaderboard"]) if row else None

    def snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, created_at, leaderboard FROM snapshots ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r["id"], "created_at": r["created_at"], **json.loads(r["leaderboard"])}
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
