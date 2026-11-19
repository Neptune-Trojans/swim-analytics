# Pipeline Design

_Agreed: 2026-10-07; pool stage added as stage 2 on 2026-11-17. Status: detection, pool, filtering, tracking
implemented (see "Build order")._

Goal: one clear pipeline, used everywhere (command line, notebooks, future apps), where every step is visible:
detection, the pool's water, filtering, tracking, choosing the swimmer, pose (skeleton), metrics.

## Stages

```
video ──► 1 Detection ──► 2 Pool ──► 3 Filtering ──► 4 Tracking ──► 5 Swimmer ──► 6 Pose ──► 7 Metrics
            (pixels)      (pixels)     (rules)         (ids)        (pick+fill)   (pixels)
              │              │            │               │              │            │           │
              ▼              ▼            ▼               ▼              ▼            ▼           ▼
     1_detections.json  2_pool.json 3_filtered.json 4_tracks.json 5_swimmer.json 6_poses.json 7_metrics.json

Visualization: draws any of these outputs as a layer on the video (it is not a step in the chain)
```

Videos can also be processed, and their outputs watched, in the browser (the app's Videos and Play pages): see
[app.md](app.md).

| # | Stage | Input | Output | Needs video frames? |
|---|---|---|---|---|
| 1 | **Detection** | video | `VideoDetections`: all persons per frame | yes |
| 2 | **Pool** | video | `VideoPool`: the pool's water per frame, as polygons (the pool model: pool-segmentation package) | yes |
| 3 | **Filtering** | detections | Same format; removed detections are kept with the **reason** they were removed | no |
| 4 | **Tracking** | detections | Same format, with `track_id` filled in; boxes the tracker didn't use in `rejected` as "untracked" | no |
| 5 | **Swimmer** | tracks | One track: our swimmer, with short gaps filled | no |
| 6 | **Pose** | video + swimmer boxes | Joint positions (keypoints) per frame | yes |
| 7 | **Metrics** | poses + swimmer track + fps | Joint angles, velocity, speed, stroke rate | no |

Stage names (used on the command line): `detection`, `pool`, `filtering`, `tracking`, `swimmer`, `pose`, `metrics`.

The pool sits between detection and filtering (2026-11-17): filtering uses it to drop boxes outside the water
(`outside_pool`, since 2026-11-18), and in that place, changing filtering or tracking settings (the Play page's Apply)
doesn't re-run it. Before 2026-11-17, filtering and tracking were stages 2 and 3 (`2_filtered.json`, `3_tracks.json`):
such results show "Needs update", and the old files are removed when a stage is saved.

## Design rules

1. **The runner and the stages are separate.** `swim/pipeline/` runs the stages; `swim/stages/` holds them.
   One folder per stage, each containing:
   - `types.py`: the stage's output data classes (+ the contract its models follow, e.g. `Detector`)
   - `stage.py`: the one function the runner calls (e.g. `detect(...)`)
   - model or rule files (e.g. `rfdetr_detector.py`)
2. **The pipeline is a plain function, not a framework.** `run_pipeline()` in `swim/pipeline/runner.py` calls the
   stages in order and reads like a table of contents.
3. **Recorded videos only.** Each stage processes the whole video, then saves its output. Later steps need the whole
   video anyway (removing short tracks, filling gaps, smoothing speed look backwards and forwards in time).
4. **Every stage output is saved**, so any stage can be re-run from saved results without re-running the models.
   Re-running a stage deletes the saved results of the stages after it (they are outdated), and `config.json`
   always records the settings each saved result was made with. Both happen when the stage's new output is saved,
   so a run that stops halfway (error, Ctrl+C, Cancel in the app) leaves consistent results.
5. **One output folder per video**, files numbered by stage, plus the settings used.
6. **One command for everything:** `python -m swim.run`. No separate scripts with their own processing loops. The
   app (`python -m swim.app`) calls the same `run_pipeline()`; its `progress` callback reports the stage and frame and
   can cancel the run.
7. **Visualization draws layers**, one per stage output, so each stage can be viewed (and debugged) on its own.

## Folder structure

```
swim/
  run.py             # command line: python -m swim.run ...
  pipeline/          # the runner
    runner.py        #   run_pipeline(): all stages in order, save / load results
    config.py        #   settings for every stage (model, thresholds, ...)
  stages/            # the steps, one folder each, in pipeline order
    detection/       #   1  types.py, stage.py, rfdetr_detector.py
    pool/            #   2  types.py, stage.py, unet_model.py (the network itself: the pool-segmentation package)
    filtering/       #   3  stage.py, filter rules (min_score.py, outside_pool.py, duplicates.py)
    tracking/        #   4  stage.py, roboflow_trackers.py (ByteTrack, OC-SORT, SORT)
    swimmer/         #   5  stage.py: pick the swimmer's track, join its pieces, fill gaps
    pose/            #   6  types.py (keypoints), stage.py, pose models
    metrics/         #   7  types.py, stage.py
  visualization/     # one drawing layer per stage output + video rendering
  utils/             # shared helpers, not specific to any stage
    video.py         #   read frames, write video, video info
```

The pipeline function:

```python
def run_pipeline(video_path, config):
    detections = detect(video_path, config.detection)              # 1
    pool = segment_pool(video_path, config.pool)                   # 2
    detections = filter_detections(detections, pool, config.filtering)   # 3
    tracks = track(detections, config.tracking)                    # 4
    swimmer = select_swimmer(tracks, config.swimmer)               # 5
    poses = estimate_poses(video_path, swimmer, config.pose)       # 6
    metrics = compute_metrics(poses, swimmer)                      # 7
```

## Output layout

```
data/output/IMG_4884/
  config.json                   # exact settings used, so results can be reproduced
  1_detections.json
  2_pool.json                   # compact JSON: a polygon per frame
  3_filtered.json
  4_tracks.json
  5_swimmer.json
  6_poses.json
  7_metrics.json
  video.mp4                     # rendered with the chosen layers (default: pool + the last stage's boxes)
```

## Command line

```bash
python -m swim.run --input data/input/IMG_4884.MOV                      # all stages + video
python -m swim.run --input data/input/IMG_4884.MOV --to detection       # stop after a stage
python -m swim.run --input data/input/IMG_4884.MOV --from tracking      # re-use saved results before tracking
python -m swim.run --input data/input/IMG_4884.MOV --from pool --pool-model 20261116-224516   # another pool model
python -m swim.run --input data/input/IMG_4884.MOV --render-only --layers pool,detections,tracks
```

## Build order

Each step leaves a pipeline that works end to end.

- [x] 1. `run.py`, `pipeline/` (runner + config); move detection into `stages/detection/`; pipeline runs detection only and
  renders the detections layer. Replaces `scripts/run_rfdetr.py` and `scripts/visualize_detections.py`.
- [x] 2. Filtering: `min_score` and `duplicate` (IoU > 0.5, keep the higher score) filters; removed
  detections are kept in `rejected` with the reason and drawn in gray. Unit tests in `tests/stages/filtering/`.
- [x] 3. Tracking: ByteTrack (Roboflow `trackers` library, also OC-SORT and SORT) behind `track()`; IDs 1, 2, ... in
  order of first appearance; boxes not used in `rejected` as "untracked"; `tracks` layer (Python + viewer); settings
  in the viewer's panel. Unit tests in `tests/stages/tracking/`, runner tests in `tests/pipeline/`.
- [x] Pool (added 2026-11-17 as stage 2): the water per frame from the pool model trained on our frames
  (pool-segmentation package, its `docs/model.md`), saved as polygons; `pool` layer (Python + viewer), drawn under the boxes and shown by
  default. Unit tests in `tests/stages/pool/`; tests use a stand-in model (`tests/conftest.py`).
- [x] Filtering uses the pool (2026-11-18): `outside_pool` filter, between `min_score` and `duplicate`. Unit tests in
  `tests/stages/filtering/test_outside_pool.py`.
- [ ] 5. Choosing the swimmer
- [ ] 6. Pose
- [ ] 7. Metrics

## Decisions

- One folder per stage: **yes** (`swim/stages/<stage>/`)
- Runner and stages in separate folders: **yes** (`swim/pipeline/` and `swim/stages/`)
- Live / real-time video: **not needed**, recorded videos only
- Command line entry point: **`python -m swim.run`**
- Detection threshold **0.1** and filtering `min_score` **0.1** (2026-11-02, were 0.5 and 0.4): RF-DETR has no NMS,
  the threshold is only a score cut. Tracking (ByteTrack) uses low-score boxes to continue tracks of a half-submerged
  swimmer, so they must reach it; duplicates are removed by the filtering stage's NMS (`max_iou`). `min_score` stays
  as a setting (viewer settings panel) for trackers without a low-score pass.
- The pool model in its own repo (2026-11-18): **pool-segmentation** (`../pool-segmentation`, pip package
  `pool_segmentation`) holds the network, its training, the labeling tools and the Labels page; it knows nothing about
  swimmers, so others can use it for any pool video. This repo depends on it (`pyproject.toml`), uses `PoolModel` and
  `find_model` in `swim/stages/pool/` and mounts its Labels page at `/labels/`. The labeling dataset lives in that
  repo (`../pool-segmentation/data/labels/pool/`, since 2026-11-18); the trained models stay here
  (`data/models/pool/`), where the pool stage finds them.
- People outside the pool: filter **`outside_pool`** (2026-11-18). A box is removed when its bottom-center (where the
  person floats or stands) is more than `pool_margin` × the frame height outside the pool stage's water; a frame
  without water keeps all its boxes. Measured on IMG_5198 (1080p, 19,827 boxes kept by the other filters):
  - IoU of the box with the pool: useless, about 0 for every box (median 0, max 0.29): the pool is far larger.
  - Share of the box in the water: removes real swimmers. At the far wall the head and shoulders stand above the
    water line, so most of the box is outside (seen: swimmers only 7-21% in the water).
  - Bottom-center: swimmers at the far wall are within 10 px of the water; the first person standing on the deck
    starts at 13 px. Default **0.01** (11 px at 1080p) keeps them all and removes the person on the deck; 0.02 would
    keep that person (and give them a track). Distances are a share of the height so the setting works at any
    resolution. Result: kept boxes 19,827 -> 8,495 (734 removed boxes scored 0.5 or more), tracks 60 -> 33.
  - Runs before `duplicate`, so a higher-scoring box on the deck in front of a swimmer can't remove the swimmer's box.
  - Limits: swimmers in water the pool model misses are removed (IMG_5198's beige-floored shallow area); a swimmer
    on the starting block or walking out is removed (feet on the deck); the swimmer stage could keep a track's
    frames out of the water later.
- Stage 5 (swimmer; stage 4 before the pool stage) gets its own folder `swim/stages/swimmer/` (2026-11-02, was planned inside `tracking/`): the
  runner expects one settings field per stage.
- Tracker: **ByteTrack** from Roboflow's `trackers` library (Apache-2.0), not our own implementation (2026-11-02).
  Defaults: `start_score` 0.5 (library: 0.7, too high for an underwater swimmer), `high_score` 0.5, `lost_seconds` 1,
  `min_iou` 0.1, `min_frames` 2. The library reports a track's first box(es) as untracked (only confirmed tracks get
  an ID); `track()` gives them the track's ID afterwards (`give_ids_to_unconfirmed_boxes`).

## Tracking: measured on our videos (2026-11-02)

Detection at 0.1, filtering defaults. "Swimmer's track" = the longest track.

| Video | Tracker | Tracks | Swimmer's track | Frames with a track |
|---|---|---|---|---|
| `IMG_4887` (one swimmer, others later) | ByteTrack, `min_iou` 0.2 | 5 | split in two: frames 0–118 and 117–944 | 965 / 978 |
| | **ByteTrack, `min_iou` 0.1** | 4 | **one track, frames 0–944 (94%)**, 8 short breaks | 965 / 978 |
| | OC-SORT | 2 | 75%, 19 breaks (ignores boxes below 0.5) | 761 / 978 |
| | SORT | 17 | 35%, fragmented | 965 / 978 |
| `IMG_4884` (wide view, many small swimmers) | ByteTrack, `min_iou` 0.1 | 24 | longest 18%: every swimmer breaks into several tracks | 479 / 639 |

- `lost_seconds` 2 instead of 1 and `start_score` 0.6 changed almost nothing.
- `min_iou` 0.2 split the swimmer where the box grows from torso to whole body (frame 117→118, same right edge).
- Checked in the viewer: no ID switch on `IMG_4887`'s swimmer; a swimmer in the next lane gets its own track (#3);
  weak boxes on the deck (chairs, poles, 0.10–0.15) never start a track.
- Left for stage 4: joining a swimmer's pieces (`IMG_4884`) and filling the short breaks.
- Left for filtering: weak boxes of body parts inside the swimmer's box (head, torso) survive NMS (small box inside a
  big one = low IoU); a "box mostly inside a stronger box" filter would remove them.
