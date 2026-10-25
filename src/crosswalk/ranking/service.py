"""The ranking stage: indicator events in, a leaderboard out over HTTP.

It is a consumer like the others (indicator windows are upserted into the
SQLite store) plus a read-only web service:

- ``GET /`` the leaderboard page (one self-contained HTML file, no external
  requests, so it works on an offline homelab and on a phone);
- ``GET /api/leaderboard`` the current leaderboard as JSON;
- ``GET /api/crossings?camera=<id>`` one crossing's stored windows;
- ``GET /api/history?limit=N`` recent leaderboard snapshots;
- ``/healthz``, ``/readyz``, ``/metrics`` as every stage.

A snapshot of the leaderboard is stored every ``snapshot_every_s``, and the
ranking change shown on the page is relative to the newest snapshot at least
``change_window_s`` old (default: one day), so a movement means "since about
yesterday", not "since the last refresh".
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Sequence
from importlib import resources
from typing import Any

from pydantic import Field

from crosswalk.config import StreamConfig, StrictModel
from crosswalk.indicators.camera import CameraConfig, load_cameras
from crosswalk.ranking import leaderboard
from crosswalk.ranking.store import Store
from crosswalk.stream import events as ev
from crosswalk.stream.stage import Output

log = logging.getLogger(__name__)

FRESHNESS = "crosswalk_ranking_last_update_timestamp_seconds"
WINDOWS = "crosswalk_ranking_windows_total"
RANKED = "crosswalk_ranking_crossings"
RANKING_METRIC_NAMES = {FRESHNESS, WINDOWS, RANKED}


class RankingStageConfig(StrictModel):
    kafka: StreamConfig = Field(default_factory=lambda: StreamConfig(group="crosswalk-ranking"))
    database: str = "/data/ranking.sqlite3"
    cameras_dir: str | None = "configs/cameras"
    min_observed_s: float = Field(600.0, ge=0)
    min_windows: int = Field(5, ge=1)
    min_pedestrian_minutes: float = Field(30.0, ge=0)
    confidence: float = Field(0.95, gt=0, lt=1)
    # Only windows received in this many seconds count ("right now"); null
    # keeps all history.
    period_s: float | None = Field(None, gt=0)
    snapshot_every_s: float = Field(300.0, gt=0)
    change_window_s: float = Field(86_400.0, gt=0)

    def rules(self) -> leaderboard.RankingRules:
        return leaderboard.RankingRules(
            min_observed_s=self.min_observed_s,
            min_windows=self.min_windows,
            min_pedestrian_minutes=self.min_pedestrian_minutes,
            confidence=self.confidence,
            period_s=self.period_s,
        )


class RankingHandler:
    name = "ranking"

    def __init__(self, store: Store, metrics=None):
        self.store = store
        self.metrics = metrics

    def event_id(self, event: ev.Event) -> str | None:
        return event.window_id if isinstance(event, ev.IndicatorEvent) else None

    def process(self, events: Sequence[ev.Event]) -> list[list[Output] | BaseException]:
        windows = [e for e in events if isinstance(e, ev.IndicatorEvent)]
        # One transaction per batch; a database error fails the batch and
        # the runner retries it per event, then dead-letters.
        self.store.upsert_windows(windows)
        if self.metrics is not None and windows:
            self.metrics["windows"].inc(len(windows))
            self.metrics["freshness"].set(time.time())
        results: list[list[Output] | BaseException] = []
        for e in events:
            if isinstance(e, ev.IndicatorEvent | ev.EndEvent):
                results.append([])
            else:
                results.append(ValueError(f"ranking does not consume {e.type!r} events"))
        return results


class Leaderboard:
    """Builds, caches and snapshots the leaderboard for the HTTP routes."""

    def __init__(
        self,
        store: Store,
        cameras: dict[str, CameraConfig],
        config: RankingStageConfig,
        clock=time.time,
    ):
        self.store = store
        self.cameras = cameras
        self.config = config
        self.clock = clock
        self._lock = threading.Lock()
        self._cache: tuple[float, dict[str, Any]] | None = None
        self._last_snapshot = 0.0

    def current(self, max_age_s: float = 5.0) -> dict[str, Any]:
        now = self.clock()
        with self._lock:
            if self._cache and now - self._cache[0] < max_age_s:
                return self._cache[1]
            previous = self.store.snapshot_before(now - self.config.change_window_s)
            board = leaderboard.build(
                self.store.windows(), self.cameras, self.config.rules(), previous, now
            )
            self._cache = (now, board)
            if now - self._last_snapshot >= self.config.snapshot_every_s and board["crossings"]:
                self.store.save_snapshot(board)
                self._last_snapshot = now
            return board

    def routes(self, metrics=None) -> dict:
        page = resources.files("crosswalk.ranking").joinpath("leaderboard.html").read_bytes()

        def as_json(data) -> tuple[int, bytes, str]:
            return 200, json.dumps(data, default=str).encode(), "application/json"

        def board(_q):
            data = self.current()
            if metrics is not None:
                for space, table in data["tables"].items():
                    metrics["ranked"].labels(space=space).set(len(table))
            return as_json(data)

        def crossing(q):
            camera = (q.get("camera") or [""])[0]
            if not camera:
                return 400, b'{"error": "camera is required"}', "application/json"
            return as_json({"camera_id": camera, "windows": self.store.windows_for(camera)})

        def history(q):
            try:
                limit = max(1, min(200, int((q.get("limit") or ["50"])[0])))
            except ValueError:
                limit = 50
            return as_json({"snapshots": self.store.snapshots(limit)})

        return {
            "/": lambda _q: (200, page, "text/html; charset=utf-8"),
            "/api/leaderboard": board,
            "/api/crossings": crossing,
            "/api/history": history,
        }


def ranking_metrics(registry):
    from prometheus_client import Counter, Gauge

    return {
        "windows": Counter(
            WINDOWS.removesuffix("_total"), "Indicator windows stored", registry=registry
        ),
        "freshness": Gauge(
            FRESHNESS, "Unix time the newest indicator window was stored", registry=registry
        ),
        "ranked": Gauge(
            RANKED, "Crossings with enough data to be ranked", ["space"], registry=registry
        ),
    }


def build_runner(config: RankingStageConfig, bus, store: Store, metrics=None, extra=None):
    from crosswalk.stream.bus import TopicSpec
    from crosswalk.stream.stage import RunnerConfig, StageRunner

    k = config.kafka
    return StageRunner(
        RankingHandler(store, extra),
        bus,
        RunnerConfig(
            group=k.group or "crosswalk-ranking",
            input_topics=[ev.TOPIC_INDICATORS],
            topics=[TopicSpec(ev.TOPIC_INDICATORS, k.partitions, k.replication_factor)],
            max_records=k.max_records,
            poll_timeout_s=k.poll_timeout_s,
            retries=k.retries,
            dedupe_capacity=k.dedupe_capacity,
        ),
        metrics=metrics,
    )


def serve_ranking(config: RankingStageConfig, port: int) -> dict[str, Any]:
    from prometheus_client import CollectorRegistry

    from crosswalk.stream.service import serve

    registry = CollectorRegistry()
    extra = ranking_metrics(registry)
    store = Store(config.database)
    board = Leaderboard(store, load_cameras(config.cameras_dir), config)
    return serve(
        "ranking",
        lambda bus, m: build_runner(config, bus, store, m, extra),
        port,
        routes=board.routes(extra),
        registry=registry,
    )
