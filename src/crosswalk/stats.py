"""Percentiles and summaries computed from raw samples.

Prometheus histograms are right for dashboards but only approximate quantiles
to bucket resolution, so benchmark results are computed here from every
recorded sample instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(samples: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default method), q in [0, 100]."""
    if not samples:
        return None
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(samples)
    rank = (len(ordered) - 1) * q / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def summarize(samples: Sequence[float]) -> dict[str, float | int | None]:
    if not samples:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(samples),
        "mean": sum(samples) / len(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "max": max(samples),
    }
