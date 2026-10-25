# Conflict indicators

What the indicators stage measures, how, and what the numbers do **not**
mean. Code: `src/crosswalk/indicators/`. Tests with hand-computed answers:
`tests/test_indicators.py`.

> These are observable quantities from recorded footage, computed by fixed
> rules. They are not a risk model, they do not predict crashes, and a low
> number does not mean a crossing is safe. The leaderboard compares crossings
> on these numbers as an experiment. It is never safety advice.

## Inputs

Track events from the tracking stage. Each one holds the confirmed
pedestrian and vehicle tracks in one frame, with their boxes. Every
indicator is computed from each track's **foot point**, the bottom centre of
its box, roughly where a person or a wheel touches the road.

Bicycles count as vehicles: they are what a pedestrian crosses in front of.
Riders are detected as people by the COCO model, and their bicycle as a
bicycle. A rider is therefore usually tracked twice (a "pedestrian" on a
bicycle), which inflates close approaches for crossings with bike traffic.
This is a known limitation of a detector-only approach.

## Two spaces: metric and pixel

| | Metric space | Pixel space |
| --- | --- | --- |
| When | The camera has a hand-set calibration in `configs/cameras/<camera>.yaml` | No calibration (the default for a new camera) |
| Distances | Metres on the road plane | Pixels in the image: *apparent* distances |
| Speeds | Metres per second | Not used |
| TTC, PET | Computed | **Not computed**, never estimated from pixels |
| Comparable across cameras? | Roughly, if the calibrations are good | No. A pixel near the camera covers far less road than one far away, and every camera's angle differs |

The leaderboard labels every value with its space and never ranks pixel
values against metric ones (see [ranking](ranking.md)).

### Calibration

A homography maps the road plane in the image to metres. Four points with
known positions determine it exactly. The natural reference is the four
corners of the crosswalk, with its width and length taken from a public map
or measured on site:

```yaml
calibration:
  image_points: [[420, 430], [860, 430], [940, 560], [340, 560]]   # pixels
  world_points: [[0, 0], [12.0, 0], [12.0, 3.0], [0, 3.0]]          # metres
  reference: "crosswalk corners, 12.0 m x 3.0 m, measured on the city's open map"
frame_size: [1280, 720]   # the frame size the image points were picked on
```

- Pick the image points on a frame of `frame_size`. The stage rescales the
  mapping if inference runs on resized frames.
- The `reference` field is required. It records where the metres came from,
  so a reader can judge how far to trust metric numbers.
- The mapping is only right **on the road plane**. Foot points are on it;
  heads and car roofs are not, which is why only foot points are used.
- Points above the horizon line map to infinity and are discarded.
- Error grows with distance from the four reference points. Keep the region
  of interest close to them.

### Region of interest

`roi` is a polygon in image pixels. Only foot points inside it count. Draw
it around the crosswalk and its immediate approaches, so the far pavement,
parked cars and the next junction stay out of the numbers.

## Indicators

All are computed per **window** of video (default 60 s, `window_s`), per
camera, aligned to the start of the clip. A pair of tracks counts **at most
once per window** for each indicator, so a pedestrian standing next to a
parked car for 30 seconds is one close approach, not 150.

| Indicator | Definition | Default threshold |
| --- | --- | --- |
| `pedestrians`, `vehicles` | Distinct confirmed tracks with a foot point in the ROI | none |
| `pedestrian_seconds`, `vehicle_seconds` | Sum over tracks of (last seen − first seen): **exposure** | none |
| `observed_s` | Video time covered by frames in the window | none |
| `close_approaches` | Pedestrian–vehicle pairs whose foot points were closer than the threshold in the same frame | metric: 2.0 m; pixel: 1 pedestrian box height |
| `min_separation` | Smallest such distance seen | metres or pixels |
| `path_crossings` | Pedestrian–vehicle pairs whose paths cross, with the two passing the crossing point within the window of each other | 3.0 s |
| `ttc_events`, `min_ttc_s` | Pairs whose **time to collision** fell below the threshold: if both kept their current velocities, the time until they come within `collision_radius_m` (1.0 m). Velocities are least-squares fits over the last second of foot points. TTC 0 (already within the radius) is a close approach, not a TTC event | 1.5 s (FHWA SSAM's default maximum TTC) |
| `pet_events`, `min_pet_s` | Pairs whose paths cross with a **post-encroachment time** below the threshold: the time between the first object leaving the conflict point and the second arriving | 3.0 s (a project default, not from a source) |

The thresholds live in each camera's `thresholds` section, so a reviewer can
change them and re-run.

TTC and PET are long-established *surrogate safety measures* in traffic
engineering. FHWA's Surrogate Safety Assessment Model uses TTC ≤ 1.5 s by
default to classify a conflict ([FHWA-HRT-08-051, ch. 2](https://www.fhwa.dot.gov/publications/research/safety/08051/02.cfm)).
That literature works from high-quality trajectories, often from simulation.
Trajectories from a pretrained detector on a single camera are much noisier.
Treat the values here as indicative.

## What the numbers do not tell you

- **Not risk.** Close approaches happen at every busy crossing, and almost
  none lead to harm. The indicators count what is visible, not what matters
  in an incident.
- **Not causation or blame.** A close approach says nothing about who
  yielded, who had right of way, or whether a signal was green.
- **Detector errors pass straight through.** Missed pedestrians (small,
  occluded, at night), false positives, ID switches and fragmented tracks
  change every count. FCOS pretrained on COCO was not trained on fixed
  intersection cameras. The fine-tuned model is scored on the hand-labelled
  fixed-camera set ([benchmarks](benchmarks.md#model-quality)) so that error
  can be quantified.
- **Pixel space is apparent.** Pixel distances depend on camera height,
  angle and zoom. Two cameras' pixel numbers cannot be compared.
- **Calibration error.** Four hand-picked points give a mapping that is
  exact at those points and increasingly wrong away from them, and nowhere
  right off the road plane.
- **Sampling.** At the sampler's default 5 fps, fast vehicles move about
  2.8 m between frames at 50 km/h. Minimum separation and TTC can miss the
  true minimum between frames. The sampler's `motion` policy samples faster
  while the scene changes and slower while it is still; a camera's windows
  are only comparable with another's if both used the same policy.
- **Recorded footage, not the present.** A clip shows one period on one day.
  The leaderboard's "right now" is the time the indicators were computed, not
  the time the footage was recorded.
