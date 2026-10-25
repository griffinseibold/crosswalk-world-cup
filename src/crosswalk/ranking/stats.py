"""Rates with honest uncertainty, without SciPy.

The leaderboard ranks crossings by **event rates normalised by exposure**:
close approaches per 100 pedestrian-minutes, say. Counts of rare events are
noisy, and a crossing observed for five minutes with one close approach is
not "worse" than one observed for five hours with forty. So every rate comes
with an exact Poisson confidence interval, and the ranking orders by the
interval's *upper* bound: the rate the data cannot rule out. A crossing with
little data gets a wide interval and a high upper bound, so it cannot top a
"fewest events" ranking on luck.

The Poisson interval is the exact (Garwood) one: for ``k`` events,

    lower = Gamma^-1(alpha/2;   k)        (0 when k = 0)
    upper = Gamma^-1(1 - alpha/2; k + 1)

in units of events, divided by the exposure. The inverse regularised gamma
function is found by bisection on a series/continued-fraction evaluation of
``P(a, x)`` (Numerical Recipes' ``gammp``). Tests compare against SciPy
reference values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _gamma_series(a: float, x: float) -> float:
    total = term = 1.0 / a
    ap = a
    for _ in range(10_000):
        ap += 1
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    # Lentz's method for the continued fraction of Q(a, x).
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x)."""
    if a <= 0:
        raise ValueError("a must be positive")
    if x <= 0:
        return 0.0
    if x < a + 1:
        return _gamma_series(a, x)
    return 1.0 - _gamma_cf(a, x)


def gammaincinv(a: float, p: float) -> float:
    """x such that P(a, x) = p, by bisection (monotone in x)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    lo, hi = 0.0, max(1.0, a)
    while gammainc(a, hi) < p:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if gammainc(a, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(1.0, hi):
            break
    return (lo + hi) / 2


def poisson_interval(k: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a Poisson mean, given ``k`` observed events."""
    if k < 0:
        raise ValueError("k must be non-negative")
    alpha = 1 - confidence
    lower = 0.0 if k == 0 else gammaincinv(k, alpha / 2)
    upper = gammaincinv(k + 1, 1 - alpha / 2)
    return lower, upper


@dataclass(frozen=True)
class Rate:
    events: int
    exposure: float  # in the rate's unit (e.g. 100 pedestrian-minutes)
    value: float | None
    lower: float | None
    upper: float | None

    def to_json(self) -> dict:
        r = lambda v: None if v is None else round(v, 4)  # noqa: E731
        return {"events": self.events, "exposure": round(self.exposure, 4), "rate": r(self.value),
                "lower": r(self.lower), "upper": r(self.upper)}  # fmt: skip


def rate(events: int, exposure: float, confidence: float = 0.95) -> Rate:
    """Events per unit exposure with its exact Poisson interval.

    With no exposure there is no rate (``None``), not zero: nothing was
    watched, so nothing can be said.
    """
    if exposure <= 0:
        return Rate(events, 0.0, None, None, None)
    lo, hi = poisson_interval(events, confidence)
    return Rate(events, exposure, events / exposure, lo / exposure, hi / exposure)
