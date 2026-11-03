# swim-analytics

Process a swimming video and overlay analysis layers (skeleton, joint angles, velocity, speed) on it.

The analysis is one pipeline of stages: detection → filtering → tracking → swimmer → pose → metrics
(implemented: detection, filtering, tracking).
See [docs/pipeline.md](docs/pipeline.md) for the design and which stages are implemented.

## Structure

```
swim/
  run.py              # command line: python -m swim.run ...
  pipeline/           # the runner
    runner.py         #   run_pipeline(): all stages in order, save / load results
    config.py         #   settings for every stage
  stages/             # the steps, one folder each, in pipeline order
    detection/        #   stage 1: find all persons in every frame
      types.py        #     output data classes (BBox, Detection, ...) + Detector contract
      stage.py        #     detect(): the function the runner calls
      rfdetr_detector.py
    filtering/        #   stage 2: remove problematic detections (kept as "rejected" with the reason)
      stage.py        #     filter_detections(): runs the filters in order
      min_score.py    #     low confidence
      duplicates.py   #     boxes overlapping too much (keeps the higher score)
    tracking/         #   stage 3: the same person keeps the same track_id from frame to frame
      stage.py        #     track(): runs the tracker frame by frame, numbers tracks 1, 2, ...
      roboflow_trackers.py  # ByteTrack, OC-SORT, SORT from the `trackers` library
  visualization/      # one drawing layer per stage output
    detections.py     #   draw detection boxes (rejected ones in gray with the reason)
    tracks.py         #   draw tracks: one color per track with its number
    render.py         #   draw the chosen layers on the video
  utils/              # shared helpers, not specific to any stage (one file per topic)
    video.py          #   read frames, write video, video info
  app/                # local web app: python -m swim.app (docs/app.md)
    server.py         #   the API: folders and videos, processing jobs, video + stage results
    library.py        #   videos in a folder and the state of their results
    jobs.py           #   background processing (one video at a time) through run_pipeline()
    static/           #   the pages: videos.js (Videos), viewer.js (Play), layers.js (drawing layers, mirrors visualization/)
tests/                # unit tests, same folder structure as swim/
data/
  input/              # source swim videos
  output/             # results, one folder per video
docs/                 # design documents
notebooks/            # experiments (detection_stats.ipynb: statistics of detected boxes)
research/             # research notes (one .md per topic; local only, git-ignored)
```

## Usage

Run all commands from the repo root.

### Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[app,dev,notebooks]"   # dependencies from pyproject.toml (app: web app, dev: tests)
```

### 1. Process a video

Runs the pipeline stages (detection, filtering, tracking, ...) on a video and saves the results:

```bash
.venv/bin/python -m swim.run --input data/input/IMG_4887.MOV
```

Results go to `data/output/<video name>/`: `config.json` (settings used), one JSON per stage (`1_detections.json`,
`2_filtered.json`, `3_tracks.json`, ...) and `video.mp4` (the video with the last stage's boxes drawn on it).

More options:

```bash
.venv/bin/python -m swim.run --input data/input/IMG_4887.MOV --to detection                            # stop after a stage
.venv/bin/python -m swim.run --input data/input/IMG_4887.MOV --from filtering --filter-min-score 0.6   # re-run from a stage, re-using saved results before it
.venv/bin/python -m swim.run --input data/input/IMG_4887.MOV --from tracking --tracker ocsort         # try another tracker
.venv/bin/python -m swim.run --input data/input/IMG_4887.MOV --render-only --layers filtered,tracks  # only re-draw video.mp4
.venv/bin/python -m swim.run --help                                                                     # all options
```

### 2. Or use the app: process videos and watch them in the browser

```bash
.venv/bin/python -m swim.app
```

Then open <http://127.0.0.1:8000> in **Chrome or Safari** (iPhone videos are HEVC; Firefox may not play them).

- **Videos page:** the videos in a folder (`data/input` first; type another path or click a subfolder), each with
  the state of its results: *Not processed*, *Up to date*, or *Needs update* with what changed (e.g. results made
  with another detection threshold, or a stage not run yet). **Process** / **Update** runs the pipeline in the
  background (only the stages that need it), with progress and **Cancel**; tick several videos to queue them.
  **Settings for processing** changes the settings used. Results go to `data/output/<video name>/`, as on the
  command line.
- **Play page** (**Open**): the video with the results drawn on top, frame by frame. In its **Settings** panel you
  can change the filtering and tracking settings (e.g. `min_score`, the tracker) and press **Apply**: they re-run
  from the saved detections in under a second, and the results are saved and shown on the video.

Stop the app with `Ctrl+C` (this also stops a video being processed). Options: `--input <folder>` (folder opened
first), `--output <folder>` (another results folder), `--port 8080`. See [docs/app.md](docs/app.md).

## Tests

```bash
.venv/bin/python -m pytest                    # all tests (the app's browser tests need Chrome, ~4 s)
.venv/bin/python -m pytest -m "not browser"   # without the browser test
```
