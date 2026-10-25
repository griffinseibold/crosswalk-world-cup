"""Configuration and entry points for the streaming stages after inference.

One YAML file (``configs/stages.yaml``, rendered into a ConfigMap by the
chart) holds a section per stage, so the camera calibration that the
indicators stage needs and the window length the ranking expects are
reviewed side by side.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from crosswalk.config import StrictModel, TrackingStageConfig, load_config


class StagesConfig(StrictModel):
    tracking: TrackingStageConfig = Field(default_factory=TrackingStageConfig)


def run_stage(name: str, path: str, overrides: list[str], port: int = 8080) -> dict[str, Any]:
    from crosswalk.stream.service import serve

    config = load_config(StagesConfig, path, overrides)
    if name == "tracking":
        from crosswalk.tracking.stage import build_runner

        return serve("tracking", lambda bus, m: build_runner(config.tracking, bus, m), port)
    raise SystemExit(f"unknown stage {name!r}")
