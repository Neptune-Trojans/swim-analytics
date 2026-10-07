# Pipeline Design

_Agreed: 2026-10-07. Status: steps 1–2 implemented (see "Build order")._

Goal: one clear pipeline, used everywhere (command line, notebooks, future apps), where every step is visible:
detection, filtering, tracking, choosing the swimmer, pose (skeleton), metrics.

## Stages

```
video ──► 1 Detection ──► 2 Filtering ──► 3 Tracking ──► 4 Swimmer ──► 5 Pose ──► 6 Metrics
            (pixels)       (rules)        (ids)         (pick+fill)   (pixels)
              │               │              │              │            │           │
              ▼               ▼              ▼              ▼            ▼           ▼
       1_detections.json 2_filtered.json 3_tracks.json 4_swimmer.json 5_poses.json 6_metrics.json

Visualization: draws any of these outputs as a layer on the video (it is not a step in the chain)
```

| # | Stage | Input | Output | Needs video frames? |
|---|---|---|---|---|
| 1 | **Detection** | video | `VideoDetections`: all persons per frame | yes |
| 2 | **Filtering** | detections | Same format; removed detections are kept with the **reason** they were removed | no |
| 3 | **Tracking** | detections | Detections with `track_id` filled in | no |
| 4 | **Swimmer** | tracks | One track: our swimmer, with short gaps filled | no |
| 5 | **Pose** | video + swimmer boxes | Joint positions (keypoints) per frame | yes |
| 6 | **Metrics** | poses + swimmer track + fps | Joint angles, velocity, speed, stroke rate | no |

Stage names (used on the command line): `detection`, `filtering`, `tracking`, `swimmer`, `pose`, `metrics`.

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
   always records the settings each saved result was made with.
5. **One output folder per video**, files numbered by stage, plus the settings used.
6. **One command for everything:** `python -m swim.run`. No separate scripts with their own processing loops.
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
    filtering/       #   2  types.py?, stage.py, filter rules
    tracking/        #   3+4  types.py, stage.py, tracker (ByteTrack-style), swimmer selection
    pose/            #   5  types.py (keypoints), stage.py, pose models
    metrics/         #   6  types.py, stage.py
  visualization/     # one drawing layer per stage output + video rendering
  utils/             # shared helpers, not specific to any stage
    video.py         #   read frames, write video, video info
```

The pipeline function:

```python
def run_pipeline(video_path, config):
    detections = detect(video_path, config.detection)              # 1
    detections = filter_detections(detections, config.filtering)   # 2
    tracks = track(detections, config.tracking)                    # 3
    swimmer = select_swimmer(tracks, config.swimmer)               # 4
    poses = estimate_poses(video_path, swimmer, config.pose)       # 5
    metrics = compute_metrics(poses, swimmer)                      # 6
```

## Output layout

```
data/output/IMG_4884/
  config.json                   # exact settings used, so results can be reproduced
  1_detections.json
  2_filtered.json
  3_tracks.json
  4_swimmer.json
  5_poses.json
  6_metrics.json
  video.mp4                     # rendered with the chosen layers
```

## Command line (planned)

```bash
python -m swim.run --input data/input/IMG_4884.MOV                      # all stages + video
python -m swim.run --input data/input/IMG_4884.MOV --to detection       # stop after a stage
python -m swim.run --input data/input/IMG_4884.MOV --from tracking      # re-use saved results before tracking
python -m swim.run --input data/input/IMG_4884.MOV --render-only --layers detections,tracks
```

## Build order

Each step leaves a pipeline that works end to end.

- [x] 1. `run.py`, `pipeline/` (runner + config); move detection into `stages/detection/`; pipeline runs detection only and
  renders the detections layer. Replaces `scripts/run_rfdetr.py` and `scripts/visualize_detections.py`.
- [x] 2. Filtering: `min_score` (default 0.4) and `duplicate` (IoU > 0.5, keep the higher score) filters; removed
  detections are kept in `rejected` with the reason and drawn in gray. Unit tests in `tests/stages/filtering/`.
- [ ] 3. Tracking
- [ ] 4. Choosing the swimmer
- [ ] 5. Pose
- [ ] 6. Metrics

## Decisions

- One folder per stage: **yes** (`swim/stages/<stage>/`)
- Runner and stages in separate folders: **yes** (`swim/pipeline/` and `swim/stages/`)
- Live / real-time video: **not needed**, recorded videos only
- Command line entry point: **`python -m swim.run`**
