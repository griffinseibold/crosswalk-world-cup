"""Run one Python stage as a long-lived service (one Deployment per stage).

``crosswalk stage tracking --config configs/stages.yaml`` (and ``indicators``,
``ranking``) all go through :func:`serve`: it starts the probe/metrics server,
installs SIGTERM handling, waits for Kafka (a fresh install may sync this app
before the broker is ready; a missing broker is a state, not a crash loop),
then runs the stage's consume loop until stopped.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from typing import Any

from crosswalk.stream.bus import Bus, KafkaBus
from crosswalk.stream.stage import StageMetrics, StageRunner

log = logging.getLogger(__name__)

KAFKA_RETRY_S = 10


def serve(
    name: str,
    make_runner: Callable[[Bus, StageMetrics], StageRunner],
    port: int,
    routes: dict | None = None,
    registry=None,
    bus: Bus | None = None,
    on_start: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from prometheus_client import CollectorRegistry, generate_latest

    from crosswalk.inference.server import StatusServer

    registry = registry or CollectorRegistry()
    stage_metrics = StageMetrics(registry)
    state: dict[str, Any] = {"stage": name, "phase": "starting"}

    class _Metrics:
        def render(self) -> bytes:
            return generate_latest(registry)

    server = StatusServer(port, _Metrics(), lambda: dict(state), routes=routes)
    server.start()
    stop = threading.Event()
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, lambda s, f: stop.set())

    bus = bus or KafkaBus()
    runner = make_runner(bus, stage_metrics)
    try:
        while not stop.is_set():
            try:
                runner.start()
                break
            except Exception as err:
                state.update(phase="waiting for Kafka", error=f"{type(err).__name__}: {err}")
                log.warning(
                    "%s: Kafka not reachable (%s); retrying in %ds", name, err, KAFKA_RETRY_S
                )
                server.ready.set()
                stop.wait(KAFKA_RETRY_S)
        if stop.is_set():
            return state
        state.pop("error", None)
        if on_start:
            on_start()
        state["phase"] = "consuming"
        state["counts"] = runner.counts
        server.ready.set()
        threading.Thread(target=lambda: (stop.wait(), runner.stop.set()), daemon=True).start()
        runner.run()
        state["phase"] = "stopped"
        return state
    finally:
        runner.close()
        server.stop()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
