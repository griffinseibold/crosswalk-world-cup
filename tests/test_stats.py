import numpy as np
import pytest

from crosswalk.stats import percentile, summarize


@pytest.mark.parametrize("q", [0, 25, 50, 90, 95, 99, 100])
def test_percentile_matches_numpy_linear(q):
    rng = np.random.default_rng(0)
    samples = rng.exponential(0.02, size=137).tolist()
    assert percentile(samples, q) == pytest.approx(float(np.percentile(samples, q)))


def test_percentile_edge_cases():
    assert percentile([], 50) is None
    assert percentile([3.0], 99) == 3.0
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["count"] == 4 and s["mean"] == 2.5 and s["p50"] == 2.5 and s["max"] == 4.0
    assert summarize([])["p95"] is None
