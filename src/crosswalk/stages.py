"""Configuration and entry points for the streaming stages after inference.

One YAML file (``configs/stages.yaml``, rendered into a ConfigMap by the
chart) holds a section per stage, so the camera calibration that the
indicators stage needs and the window length the ranking expects are
reviewed side by side.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from crosswalk.config import StreamConfig, StrictModel, TrackingStageConfig, load_config
from crosswalk.ranking.service import RankingStageConfig


class IndicatorsStageConfig(StrictModel):
    kafka: StreamConfig = Field(default_factory=lambda: StreamConfig(group="crosswalk-indicators"))
    # Directory of per-camera YAML files (configs/cameras/ in the repository,
    # a ConfigMap in the cluster). Cameras without a file get pixel-space
    # indicators over the whole frame.
    cameras_dir: str | None = "configs/cameras"
    # Video seconds per indicator window. Ranking needs several windows per
    # crossing, so shorter windows give more, noisier observations.
    window_s: float = Field(60.0, gt=0)


class StagesConfig(StrictModel):
    tracking: TrackingStageConfig = Field(default_factory=TrackingStageConfig)
    indicators: IndicatorsStageConfig = Field(default_factory=IndicatorsStageConfig)
    ranking: RankingStageConfig = Field(default_factory=RankingStageConfig)


def run_stage(name: str, path: str, overrides: list[str], port: int = 8080) -> dict[str, Any]:
    from crosswalk.stream.service import serve

    config = load_config(StagesConfig, path, overrides)
    if name == "tracking":
        from crosswalk.tracking.stage import build_runner

        return serve("tracking", lambda bus, m: build_runner(config.tracking, bus, m), port)
    if name == "indicators":
        from crosswalk.indicators.stage import build_runner

        return serve("indicators", lambda bus, m: build_runner(config.indicators, bus, m), port)
    if name == "ranking":
        from crosswalk.ranking.service import serve_ranking

        return serve_ranking(config.ranking, port)
    raise SystemExit(f"unknown stage {name!r}")
