"""Turn stored indicator windows into a leaderboard.

The question on the page is deliberately absurd, "where is the safest place
on Earth to cross the street right now?", and the answer is deliberately
careful. A crossing's position is decided by rules anyone can check:

1. **Aggregate** each camera's windows over the observation period:
   total observed video time, pedestrian and vehicle exposure, and event
   counts.
2. **Normalise by exposure.** The headline rate is *close approaches per 100
   pedestrian-minutes*. A crossing with ten times the foot traffic sees more
   approaches for the same behaviour, so counts alone would just rank by
   footfall.
3. **Attach uncertainty.** Every rate has an exact 95% Poisson interval
   (``stats.rate``).
4. **Require enough data.** A crossing with less than ``min_observed_s`` of
   video, fewer than ``min_windows`` windows, or less than
   ``min_pedestrian_minutes`` of pedestrian exposure is listed as "not enough
   data" and is never ranked.
5. **Rank conservatively.** Ranked crossings are ordered by the *upper* bound
   of the rate, lowest first: the crossing whose data most firmly rules out
   a high rate comes first. Luck from a short observation cannot win.
6. **Separate spaces.** Metric and pixel indicators are not comparable
   (docs/indicators.md), so calibrated and uncalibrated crossings are ranked
   in separate tables, each labelled.

The output is plain JSON; the static page renders it. Nothing here claims
that a crossing is safe or dangerous: the page's limitations section says so
in words, and the API carries the same disclaimer.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from crosswalk.indicators.camera import CameraConfig
from crosswalk.ranking.stats import rate

DISCLAIMER = (
    "Experimental comparison of observable indicators in recorded footage. "
    "Not a measure of risk and not safety advice. See the limitations section."
)

PER = 100.0  # rates per 100 pedestrian-minutes


@dataclass(frozen=True)
class RankingRules:
    min_observed_s: float = 600.0  # 10 minutes of video
    min_windows: int = 5
    min_pedestrian_minutes: float = 30.0
    confidence: float = 0.95
    # Only windows received within this many seconds count ("right now"):
    # None means all history.
    period_s: float | None = None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "windows": 0, "observed_s": 0.0, "frames": 0, "pedestrians": 0, "vehicles": 0,
            "pedestrian_seconds": 0.0, "vehicle_seconds": 0.0, "close_approaches": 0,
            "path_crossings": 0, "ttc_events": 0, "pet_events": 0, "min_separation": None,
            "min_ttc_s": None, "min_pet_s": None, "spaces": set(), "last_received_at": 0.0,
            "streams": set(),
        }
    )  # fmt: skip
    for r in rows:
        a = out[r["camera_id"]]
        a["windows"] += 1
        for key in ("observed_s", "pedestrian_seconds", "vehicle_seconds"):
            a[key] += r[key]
        for key in (
            "frames",
            "pedestrians",
            "vehicles",
            "close_approaches",
            "path_crossings",
            "ttc_events",
            "pet_events",
        ):
            a[key] += r[key]
        for key in ("min_separation", "min_ttc_s", "min_pet_s"):
            if r[key] is not None:
                a[key] = r[key] if a[key] is None else min(a[key], r[key])
        a["spaces"].add(r["space"])
        a["streams"].add(r["stream_id"])
        a["last_received_at"] = max(a["last_received_at"], r["received_at"])
    return out


def build(
    rows: list[dict[str, Any]],
    cameras: dict[str, CameraConfig],
    rules: RankingRules,
    previous: dict[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    if rules.period_s is not None:
        rows = [r for r in rows if r["received_at"] >= now - rules.period_s]
    entries = []
    for camera_id, a in sorted(aggregate(rows).items()):
        cam = cameras.get(camera_id)
        # A camera whose calibration was added later has both kinds of
        # window; only the metric ones can be compared with metric cameras.
        space = (
            "metric"
            if a["spaces"] == {"metric"}
            else "pixel"
            if a["spaces"] == {"pixel"}
            else "mixed"
        )
        ped_minutes = a["pedestrian_seconds"] / 60.0
        exposure_units = ped_minutes / PER
        close = rate(a["close_approaches"], exposure_units, rules.confidence)
        crossings = rate(a["path_crossings"], exposure_units, rules.confidence)
        reasons = []
        if a["observed_s"] < rules.min_observed_s:
            reasons.append(
                f"{a['observed_s'] / 60:.1f} of {rules.min_observed_s / 60:.0f} minutes of video"
            )
        if a["windows"] < rules.min_windows:
            reasons.append(f"{a['windows']} of {rules.min_windows} windows")
        if ped_minutes < rules.min_pedestrian_minutes:
            reasons.append(
                f"{ped_minutes:.1f} of {rules.min_pedestrian_minutes:.0f} pedestrian-minutes"
            )
        if space == "mixed":
            reasons.append("mixes calibrated and uncalibrated windows")
        entry = {
            "camera_id": camera_id,
            "crossing": (cam.crossing if cam and cam.crossing else camera_id),
            "location": cam.location if cam else "",
            "space": space,
            "observation": {
                "windows": a["windows"],
                "observed_minutes": round(a["observed_s"] / 60, 2),
                "streams": len(a["streams"]),
                "pedestrians": a["pedestrians"],
                "vehicles": a["vehicles"],
                "pedestrian_minutes": round(ped_minutes, 2),
                "vehicle_minutes": round(a["vehicle_seconds"] / 60, 2),
            },
            "counts": {
                "close_approaches": a["close_approaches"],
                "path_crossings": a["path_crossings"],
                "ttc_events": a["ttc_events"] if space == "metric" else None,
                "pet_events": a["pet_events"] if space == "metric" else None,
            },
            "extremes": {
                "min_separation": a["min_separation"],
                "min_separation_unit": "m" if space == "metric" else "px (apparent)",
                "min_ttc_s": a["min_ttc_s"] if space == "metric" else None,
                "min_pet_s": a["min_pet_s"] if space == "metric" else None,
            },
            "rates_per_100_ped_minutes": {
                "close_approaches": close.to_json(),
                "path_crossings": crossings.to_json(),
            },
            "freshness": {
                "last_update": a["last_received_at"],
                "age_s": round(max(0.0, now - a["last_received_at"]), 1),
            },
            "eligible": not reasons,
            "not_ranked_because": reasons,
            "source": {
                "source": cam.source if cam else "",
                "license": cam.license if cam else "",
                "attribution": cam.attribution if cam else "",
            },
            "rank": None,
            "previous_rank": None,
            "change": None,
        }
        entries.append(entry)

    prev_ranks = {}
    if previous:
        for table in previous.get("tables", {}).values():
            for e in table:
                prev_ranks[(e["camera_id"], e["space"])] = e["rank"]

    tables: dict[str, list[dict[str, Any]]] = {"metric": [], "pixel": []}
    for space in tables:
        ranked = [e for e in entries if e["space"] == space and e["eligible"]]
        # Lowest upper bound first; ties broken by more observation (more
        # certain), then name, so the order is deterministic.
        ranked.sort(
            key=lambda e: (
                e["rates_per_100_ped_minutes"]["close_approaches"]["upper"],
                -e["observation"]["pedestrian_minutes"],
                e["camera_id"],
            )
        )
        for i, e in enumerate(ranked, start=1):
            e["rank"] = i
            before = prev_ranks.get((e["camera_id"], space))
            e["previous_rank"] = before
            e["change"] = "new" if before is None else (before - i)  # positive: moved up
        tables[space] = ranked

    return {
        "schema": 1,
        "generated_at": now,
        "question": "Where is the safest place on Earth to cross the street right now?",
        "answer_style": "experimental comparison of observable indicators",
        "disclaimer": DISCLAIMER,
        "metric": "close approaches per 100 pedestrian-minutes (95% exact Poisson interval); ranked by upper bound",
        "rules": {
            "min_observed_minutes": rules.min_observed_s / 60,
            "min_windows": rules.min_windows,
            "min_pedestrian_minutes": rules.min_pedestrian_minutes,
            "confidence": rules.confidence,
            "period_s": rules.period_s,
        },
        "tables": tables,
        "not_ranked": [e for e in entries if not e["eligible"]],
        "crossings": len(entries),
    }
