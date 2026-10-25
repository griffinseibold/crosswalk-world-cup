"""Ranking: exact Poisson intervals, eligibility, ordering, store, API and page."""

from __future__ import annotations

import json
import urllib.request
from html.parser import HTMLParser

import pytest

from crosswalk.indicators.camera import CameraConfig
from crosswalk.ranking import leaderboard
from crosswalk.ranking.stats import gammainc, poisson_interval, rate
from crosswalk.ranking.store import Store
from crosswalk.stream import events as ev

# Reference values computed with SciPy (scipy.stats.chi2.ppf / special.gammainc).
POISSON_95 = {
    0: (0.0, 3.6888794541),
    1: (0.025317808, 5.5716433909),
    2: (0.2422092785, 7.2246876677),
    5: (1.6234863901, 11.6683320793),
    10: (4.7953886961, 18.390356042),
    50: (37.1109637375, 65.9187666643),
    200: (173.2408826815, 229.7219816654),
}


@pytest.mark.parametrize("k", sorted(POISSON_95))
def test_exact_poisson_interval_matches_scipy(k):
    lo, hi = poisson_interval(k)
    assert lo == pytest.approx(POISSON_95[k][0], rel=1e-7, abs=1e-9)
    assert hi == pytest.approx(POISSON_95[k][1], rel=1e-7)


def test_other_confidence_levels_and_gammainc():
    lo, hi = poisson_interval(3, 0.90)
    assert (lo, hi) == pytest.approx((0.8176914471639534, 7.753656527932725), rel=1e-7)
    for a, x, expected in [(0.5, 0.1, 0.34527915398142317), (1, 1, 0.6321205588285577),
                           (3, 2.5, 0.45618688411667035), (10, 12, 0.7576078383294875),
                           (100, 90, 0.15822098918643007)]:  # fmt: skip
        assert gammainc(a, x) == pytest.approx(expected, rel=1e-10)


def test_rate_scales_the_interval_by_exposure_and_refuses_zero_exposure():
    r = rate(10, 2.0)
    assert r.value == 5.0
    assert r.lower == pytest.approx(4.7953886961 / 2) and r.upper == pytest.approx(18.390356042 / 2)
    none = rate(3, 0.0)
    assert none.value is None and none.upper is None


# -- leaderboard -------------------------------------------------------------------


def window(
    camera, i, *, space="metric", observed=60.0, ped_s=600.0, close=1, received=1000.0, stream=None
):
    return {
        "window_id": f"{camera}/s/r-0/{i}", "camera_id": camera, "stream_id": stream or f"{camera}/s/r-0",
        "space": space, "window_start_s": i * 60.0, "window_end_s": (i + 1) * 60.0, "observed_s": observed,
        "frames": 300, "pedestrians": 10, "vehicles": 5, "pedestrian_seconds": ped_s, "vehicle_seconds": 120.0,
        "close_approaches": close, "min_separation": 1.5, "path_crossings": 0, "ttc_events": 0, "min_ttc_s": None,
        "pet_events": 0, "min_pet_s": None, "received_at": received,
    }  # fmt: skip


RULES = leaderboard.RankingRules(min_observed_s=600, min_windows=5, min_pedestrian_minutes=30)


def test_crossings_without_enough_data_are_listed_but_not_ranked():
    rows = [window("big", i) for i in range(20)] + [window("tiny", i) for i in range(2)]
    board = leaderboard.build(rows, {}, RULES, now=2000)
    assert [e["camera_id"] for e in board["tables"]["metric"]] == ["big"]
    [tiny] = board["not_ranked"]
    assert tiny["camera_id"] == "tiny" and tiny["rank"] is None
    assert any("minutes of video" in r for r in tiny["not_ranked_because"])
    assert any("windows" in r for r in tiny["not_ranked_because"])


def test_ranking_uses_the_upper_bound_so_a_short_lucky_observation_cannot_win():
    # "lucky": zero close approaches, but only the minimum of data
    # (10 windows x 600 ped-s = 100 ped-minutes = 1 unit) -> upper 3.69.
    # "steady": 1 approach per window over 200 windows: 200 events in 20
    # units -> rate 10, upper 11.5. "calm": 1 approach per 4 windows over
    # 200 windows: 50 events in 20 units -> rate 2.5, upper 3.30.
    rows = (
        [window("lucky", i, close=0) for i in range(10)]
        + [window("steady", i, close=1) for i in range(200)]
        + [window("calm", i, close=1 if i % 4 == 0 else 0) for i in range(200)]
    )
    board = leaderboard.build(rows, {}, RULES, now=5000)
    order = [e["camera_id"] for e in board["tables"]["metric"]]
    # lucky's point estimate (0) is the lowest, but its upper bound (3.69)
    # is above calm's (3.30): calm ranks first.
    assert order == ["calm", "lucky", "steady"]
    calm = board["tables"]["metric"][0]["rates_per_100_ped_minutes"]["close_approaches"]
    assert calm["rate"] == pytest.approx(2.5)
    assert calm["upper"] == pytest.approx(65.9187666643 / 20, abs=1e-4)  # JSON rounds to 4 places


def test_metric_and_pixel_crossings_are_ranked_separately_and_mixed_ones_not_at_all():
    rows = (
        [window("m", i) for i in range(10)]
        + [window("p", i, space="pixel") for i in range(10)]
        + [window("x", i, space="pixel" if i < 5 else "metric") for i in range(10)]
    )
    board = leaderboard.build(rows, {}, RULES, now=5000)
    assert [e["camera_id"] for e in board["tables"]["metric"]] == ["m"]
    assert [e["camera_id"] for e in board["tables"]["pixel"]] == ["p"]
    [mixed] = board["not_ranked"]
    assert mixed["camera_id"] == "x" and "mixes" in mixed["not_ranked_because"][0]
    pixel = board["tables"]["pixel"][0]
    assert (
        pixel["counts"]["ttc_events"] is None
        and pixel["extremes"]["min_separation_unit"] == "px (apparent)"
    )


def test_rank_changes_are_relative_to_the_previous_snapshot():
    rows = [window("a", i, close=0) for i in range(10)] + [
        window("b", i, close=3) for i in range(10)
    ]
    before = leaderboard.build(rows, {}, RULES, now=5000)
    assert [e["camera_id"] for e in before["tables"]["metric"]] == ["a", "b"]
    rows += [window("a", i, close=9) for i in range(10, 30)] + [
        window("c", i, close=0) for i in range(30)
    ]
    after = leaderboard.build(rows, {}, RULES, previous=before, now=6000)
    by_id = {e["camera_id"]: e for e in after["tables"]["metric"]}
    assert by_id["c"]["change"] == "new"
    assert by_id["b"]["previous_rank"] == 2 and by_id["b"]["change"] == 2 - by_id["b"]["rank"]
    assert by_id["a"]["change"] == 1 - by_id["a"]["rank"]


def test_period_limits_the_leaderboard_to_recent_windows_and_freshness_is_reported():
    rows = [window("old", i, received=100.0) for i in range(10)] + [
        window("new", i, received=9000.0) for i in range(10)
    ]
    rules = leaderboard.RankingRules(
        min_observed_s=600, min_windows=5, min_pedestrian_minutes=30, period_s=3600
    )
    board = leaderboard.build(
        rows, {"new": CameraConfig(camera_id="new", crossing="New St")}, rules, now=10000
    )
    assert board["crossings"] == 1
    entry = board["tables"]["metric"][0]
    assert entry["crossing"] == "New St" and entry["freshness"]["age_s"] == 1000.0
    assert "not safety advice" in board["disclaimer"]


# -- store ----------------------------------------------------------------------------


def indicator(camera, i, close=1) -> ev.IndicatorEvent:
    return ev.IndicatorEvent(
        window_id=f"{camera}/s/r-0/{i}", stream_id=f"{camera}/s/r-0", camera_id=camera, window_start_s=i * 60.0,
        window_end_s=(i + 1) * 60.0, observed_s=59.8, frames=300, space="metric", pedestrians=4, vehicles=3,
        pedestrian_seconds=600.0, vehicle_seconds=100.0, close_approaches=close, min_separation=1.2,
        path_crossings=0,
    )  # fmt: skip


def test_store_upserts_by_window_id_so_replays_do_not_double_count(tmp_path):
    store = Store(tmp_path / "r.sqlite3", clock=lambda: 1000.0)
    store.upsert_windows([indicator("a", i) for i in range(3)])
    store.upsert_windows([indicator("a", 1, close=7)])  # redelivered / recomputed
    rows = store.windows()
    assert len(rows) == 3
    assert [r["close_approaches"] for r in rows] == [1, 7, 1]
    store.close()
    # Data survives a restart (the PVC case).
    again = Store(tmp_path / "r.sqlite3")
    assert len(again.windows()) == 3
    assert again._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_snapshots_are_kept_and_looked_up_by_time(tmp_path):
    now = [100.0]
    store = Store(tmp_path / "r.sqlite3", clock=lambda: now[0])
    store.save_snapshot({"tables": {"metric": [{"camera_id": "a", "space": "metric", "rank": 1}]}})
    now[0] = 200.0
    store.save_snapshot({"tables": {"metric": [{"camera_id": "a", "space": "metric", "rank": 2}]}})
    assert store.snapshot_before(150)["tables"]["metric"][0]["rank"] == 1
    assert store.snapshot_before(50) is None
    assert [s["id"] for s in store.snapshots()] == [2, 1]


# -- the service: stage + HTTP --------------------------------------------------------------


class _Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[str] = []
        self.external: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        for key in ("src", "href"):
            url = attrs.get(key) or ""
            if tag in ("script", "link", "img") and url.startswith(("http:", "https:", "//")):
                self.external.append(url)

    def handle_data(self, data):
        self.text.append(data)


def test_leaderboard_page_is_self_contained_accessible_and_states_its_limits():
    from importlib import resources

    html = resources.files("crosswalk.ranking").joinpath("leaderboard.html").read_text()
    page = _Page()
    page.feed(html)
    text = " ".join(page.text)
    assert page.external == []  # works offline: no external scripts, styles or images
    assert 'lang="en"' in html and 'name="viewport"' in html
    assert "prefers-color-scheme: dark" in html and ":focus-visible" in html
    assert "What this cannot tell you" in text and "not safety advice" in text
    assert "aria-live" in html and 'scope: "col"' in html  # header cells are built in the script


def test_ranking_service_consumes_windows_and_serves_the_api(tmp_path):
    pytest.importorskip("prometheus_client")
    from prometheus_client import CollectorRegistry

    from crosswalk.inference.server import StatusServer
    from crosswalk.ranking.service import (
        Leaderboard,
        RankingStageConfig,
        build_runner,
        ranking_metrics,
    )
    from crosswalk.stream.bus import MemoryBus

    config = RankingStageConfig(
        database=str(tmp_path / "r.sqlite3"),
        min_observed_s=60,
        min_windows=2,
        min_pedestrian_minutes=10,
    )
    store = Store(config.database)
    registry = CollectorRegistry()
    extra = ranking_metrics(registry)
    bus = MemoryBus()
    runner = build_runner(config, bus, store, extra=extra)
    runner.config.poll_timeout_s = 0.01
    runner.start()
    producer = bus.producer()
    events = [indicator("a", i, close=0) for i in range(4)] + [
        indicator("b", i, close=5) for i in range(4)
    ]
    for e in events + events[:2]:  # two duplicates
        producer.send(ev.TOPIC_INDICATORS, e.camera_id, e.to_bytes())
    counts = runner.run(until_idle=True)
    assert counts["processed"] == 8 and counts["duplicate"] == 2
    assert len(store.windows()) == 8

    board = Leaderboard(store, {"a": CameraConfig(camera_id="a", crossing="A St")}, config)

    class _M:
        def render(self):
            return b""

    server = StatusServer(0, _M(), lambda: {}, host="127.0.0.1", routes=board.routes(extra))
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        data = json.loads(urllib.request.urlopen(f"{base}/api/leaderboard").read())
        assert [e["crossing"] for e in data["tables"]["metric"]] == ["A St", "b"]
        crossing = json.loads(urllib.request.urlopen(f"{base}/api/crossings?camera=a").read())
        assert len(crossing["windows"]) == 4
        history = json.loads(urllib.request.urlopen(f"{base}/api/history?limit=5").read())
        assert len(history["snapshots"]) == 1
        page = urllib.request.urlopen(f"{base}/")
        assert (
            page.headers["Content-Type"].startswith("text/html")
            and b"Crosswalk World Cup" in page.read()
        )
        assert page.headers["X-Content-Type-Options"] == "nosniff"
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{base}/api/crossings")
        assert err.value.code == 400
    finally:
        server.stop()
