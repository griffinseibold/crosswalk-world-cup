# Ranking and the leaderboard

How the ranking service turns indicator windows into the leaderboard at
<http://crosswalk.localhost:8080/>, and why each rule exists. Code:
`src/crosswalk/ranking/`. Tests: `tests/test_ranking.py`.

## The rules

1. **Aggregate per crossing.** All of a camera's stored windows (or only
   those received in the last `period_s`, if set) are summed: observed video
   time, pedestrian and vehicle exposure, and event counts.
2. **Normalise by exposure.** The headline number is **close approaches per
   100 pedestrian-minutes**. Raw counts would mostly rank crossings by how
   busy they are.
3. **Attach uncertainty.** Each rate carries an exact 95% Poisson (Garwood)
   interval, computed in `ranking/stats.py` without SciPy and tested against
   SciPy's values.
4. **Require enough data.** A crossing is ranked only with at least 10
   minutes of video (`min_observed_s`), 5 windows (`min_windows`) and 30
   pedestrian-minutes (`min_pedestrian_minutes`). Crossings below any of
   these are listed under "Not ranked yet" with the reason.
5. **Rank by the upper bound.** Ranked crossings are ordered by the interval's
   upper end, lowest first: the highest rate the data cannot rule out. Zero
   events in a short observation still gives a high upper bound (3.7 events
   divided by the exposure), so a short, lucky observation cannot win. Ties
   go to the crossing with more exposure.
6. **Keep spaces apart.** Calibrated (metric) and uncalibrated (pixel)
   crossings are ranked in separate tables. A crossing with both kinds of
   window (calibrated part-way through) is not ranked until its history is
   one kind. Delete its old windows or wait for `period_s` to age them out.

All thresholds are in `configs/stages.yaml` under `ranking`.

## Ranking changes and freshness

A snapshot of the leaderboard is stored every 5 minutes (`snapshot_every_s`).
The change shown next to a crossing compares with the newest snapshot that is
at least a day old (`change_window_s`): "Up 2 since a day ago". Each crossing
also shows when its newest window arrived. `CrosswalkRankingsStale` alerts
when nothing has arrived for an hour while the pipeline should be running.

## Storage: SQLite on a volume

The ranking service is the only writer, the data is small (one row per camera
per minute of video), and every query scans one table. SQLite in WAL mode on
the service's PersistentVolumeClaim gives durable, transactional storage with
no database server, credentials or backups to run. The trade-off is a single
writer: the Deployment runs exactly one replica with the `Recreate` strategy.
If the leaderboard ever needs several replicas, Postgres is the next step.
The store is one small class.

Windows are upserted by `window_id`, so a replayed or redelivered window
replaces its row instead of being counted twice.

## API

| Path | Returns |
| --- | --- |
| `/` | The leaderboard page: one self-contained HTML file with no external requests, light and dark, usable on a phone and with a keyboard |
| `/api/leaderboard` | The current leaderboard (JSON), as the page renders it |
| `/api/crossings?camera=<id>` | Every stored window for one camera |
| `/api/history?limit=N` | The most recent snapshots |
| `/healthz`, `/readyz`, `/metrics` | Probes and Prometheus metrics, as for every stage |

Every leaderboard response carries the same `disclaimer` as the page:
it is an experimental comparison of observable indicators, not a measure of
risk and not safety advice.
