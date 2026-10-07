# Filtering Abnormal Detections

_Research date: 2026-10-06_

Goal: remove detections that are not real people (splashes, reflections, posters) and, later, everything that is
not our swimmer, without losing the swimmer when they are half underwater.

## Main idea: two stages

Filtering is normally split around the tracker:

- **Before tracking:** cheap rules on each detection alone, in a single frame. Remove only what is clearly impossible.
- **After (and during) tracking:** rules that need history across frames: how long a track lives, how it moves.

## Stage 1: Per-frame filters (before tracking)

| Filter | Idea | Swimming notes |
|---|---|---|
| **Very low confidence** | Drop below a low floor, e.g. 0.1–0.3 | Keep the floor low; see the confidence section below |
| **Size** | Min/max box area (relative to the frame), min width/height in px | Tiny boxes are usually splashes or reflections; huge boxes are usually a person close to the camera |
| **Pool area** | Draw the pool outline once per camera position; drop boxes whose center is outside it | **Usually the strongest filter** for a fixed camera: removes people on the deck, spectators, posters (e.g. the banner with a printed person on the wall in `IMG_4884`) |
| **Box shape** | Width/height ratio | A swimmer seen from the side is wide, a person standing on the deck is tall. Careful: dives, starts and wall turns are briefly upright, so use it as a weak signal |
| **Duplicates** | Merge boxes that overlap heavily (high IoU) | Water sometimes splits one swimmer into two boxes, e.g. head and legs |
| **Frame edge** | Flag boxes cut by the frame edge | Keep them, but their size and center are unreliable |

## Stage 2: Track filters (after / during tracking)

| Filter | Idea |
|---|---|
| **Short tracks** | Drop tracks that live only a few frames: flickering false detections |
| **Confirmation** | A track becomes "real" only after being seen in e.g. 3 of the last 5 frames (built into SORT / ByteTrack) |
| **Impossible motion** | Jumps larger than a swimmer can move (~2.5 m/s at the very most, converted to px per frame), sudden big box-size changes, movement across lanes instead of along them |
| **Static tracks** | Something that never moves for a long time is a poster, chair or spectator. Exception: a swimmer resting at the wall |
| **Pick the target** | Among the real swimmers left, choose ours: longest track, the right lane, most movement along the lane |
| **Fill gaps** | Interpolate the swimmer's box over short gaps (e.g. the 3 missed frames in `IMG_4886`) |

## Confidence: don't filter it too hard before tracking

**ByteTrack**, the best-known modern tracker, is built on one observation: low-confidence detections are often real
objects that are partly hidden. Its rule:

1. Match existing tracks with **high-confidence** detections first (and only these can start new tracks).
2. Then use **low-confidence** detections (~0.1–0.5) only to **continue** tracks that already exist.

A low-confidence blob that continues no track is dropped automatically.

This fits swimming exactly: a half-submerged swimmer drops to ~0.6 confidence (seen on `IMG_4886`) and sometimes
below our current 0.5 detector threshold. So the usual setup is a **low detector threshold** and letting the tracker
decide.

## Suggested pipeline

```
detector (low threshold, ~0.2)
  → frame filters (pool area, size)
  → tracker (ByteTrack-style)
  → track filters (length, motion, static)
  → pick the swimmer's track
```

## Implementation notes

- Make each filter a small function that also records **why** it removed a detection.
- In the visualization, draw removed detections in gray with the reason, so it is easy to see when a filter is too
  aggressive.
- Choose thresholds from our own data, not guesses.

## Next step

Analyze our saved detections (`data/output/*_rfdetr.json`) before choosing numbers:

- Distribution of confidence scores
- Box sizes and width/height ratios
- Where boxes fall in the frame (inside / outside the pool)

This shows which filters actually matter for our videos.

## References

- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box (2021)](https://arxiv.org/abs/2110.06864): keeping low-confidence detections to continue tracks
- [SORT: Simple Online and Realtime Tracking (2016)](https://arxiv.org/abs/1602.00763): Kalman filter + IoU matching, track confirmation
- [DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric (2017)](https://arxiv.org/abs/1703.07402): adds appearance features to SORT
