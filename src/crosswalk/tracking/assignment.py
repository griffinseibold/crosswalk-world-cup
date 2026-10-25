"""Optimal one-to-one assignment (the Hungarian algorithm) and box IoU.

Association is "which detection belongs to which track", solved as a minimum
cost assignment on ``1 - IoU``. SciPy's ``linear_sum_assignment`` would do it,
but SciPy is a large dependency for one function, and the problem here is
small (tens of tracks by tens of detections per frame). This is the standard
O(n^3) shortest-augmenting-path formulation (Jonker-Volgenant style, as in
the well-known e-maxx implementation), tested against brute force.
"""

from __future__ import annotations

import itertools

import numpy as np

INF = float("inf")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` (N x 4) and ``b`` (M x 4), x1 y1 x2 y2."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def linear_sum_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost matching of rows to columns; returns (row, col) pairs.

    Rectangular matrices are allowed: every row is matched if rows <= cols,
    otherwise every column is. Costs must be finite.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape  # n <= m
    # 1-indexed potentials and matching, column 0 is a sentinel.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)  # p[j]: row matched to column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


def brute_force_assignment(cost: np.ndarray) -> float:
    """Minimum total cost by trying every matching (tests only; tiny inputs)."""
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    if n > m:
        return brute_force_assignment(cost.T)
    return min(
        sum(cost[i, c] for i, c in enumerate(cols)) for cols in itertools.permutations(range(m), n)
    )


def match(iou: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Assign on ``1 - IoU`` and keep pairs with IoU >= threshold.

    Returns (matches, unmatched rows, unmatched columns).
    """
    rows, cols = iou.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    pairs = [(r, c) for r, c in linear_sum_assignment(1.0 - iou) if iou[r, c] >= threshold]
    matched_r = {r for r, _ in pairs}
    matched_c = {c for _, c in pairs}
    return (
        pairs,
        [r for r in range(rows) if r not in matched_r],
        [c for c in range(cols) if c not in matched_c],
    )
