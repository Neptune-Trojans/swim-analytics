# swim-analytics

Process a swimming video and overlay analysis layers (skeleton, joint angles, velocity, speed) on it.

The analysis is one pipeline of stages: detection → pool → filtering → tracking → swimmer → pose → metrics
(implemented: detection, pool, filtering, tracking). The pool's water comes from our own segmentation model, in the
separate [pool-segmentation](https://github.com/Neptune-Trojans/pool-segmentation) package (model, training data
tools, Labels page).
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
    pool/             #   stage 2: the pool's water in every frame (the pool model), as polygons
      stage.py        #     segment_pool(): runs the model frame by frame
      unet_model.py   #     the trained U-Net (pool-segmentation package) -> water outline
    filtering/        #   stage 3: remove problematic detections (kept as "rejected" with the reason)
      stage.py        #     filter_detections(): runs the filters in order
      min_score.py    #     low confidence
      outside_pool.py #     people outside the pool (the box's bottom-center not in the water)
      duplicates.py   #     boxes overlapping too much (keeps the higher score)
    tracking/         #   stage 4: the same person keeps the same track_id from frame to frame
      stage.py        #     track(): runs the tracker frame by frame, numbers tracks 1, 2, ...
      roboflow_trackers.py  # ByteTrack, OC-SORT, SORT from the `trackers` library
  visualization/      # one drawing layer per stage output
    detections.py     #   draw detection boxes (rejected ones in gray with the reason)
    pool.py           #   draw the pool's water (blue tint and outline)
    tracks.py         #   draw tracks: one color per track with its number
    render.py         #   draw the chosen layers on the video
  utils/              # shared helpers, not specific to any stage (one file per topic)
    video.py          #   read frames, write video, video info
  app/                # local web app: python -m swim.app (docs/app.md)
    server.py         #   the API: folders and videos, processing jobs, video + stage results
    library.py        #   videos in a folder and the state of their results
    jobs.py           #   background processing (one video at a time) through run_pipeline()
    static/           #   the pages: videos.js (Videos), viewer.js (Play), layers.js (drawing layers)
tests/                # unit tests, same folder structure as swim/
data/
  input/              # source swim videos
  output/             # results, one folder per video
  labels/pool/        # labeling dataset: frames, SAM3's labels, checked labels, training masks (git-ignored)
  models/pool/        # trained pool models, one folder per training run (git-ignored)
docs/                 # design documents
notebooks/            # experiments (detection_stats.ipynb: statistics of detected boxes)
research/             # research notes (one .md per topic; local only, git-ignored)
```

## Usage

Run all commands from the repo root.

### Setup (once)

```bash
python3 -m venv .venv
git clone git@github.com:Neptune-Trojans/pool-segmentation.git ../pool-segmentation   # next to this repo
.venv/bin/pip install -e ../pool-segmentation             # the pool model package, editable (changes there are live here)
.venv/bin/pip install -e ".[app,dev,notebooks,train]"   # dependencies from pyproject.toml (app: web app, dev: tests, train: TensorBoard)
```

### 1. Process a video

Runs the pipeline stages (detection, pool, filtering, tracking, ...) on a video and saves the results:

```bash
.venv/bin/python -m swim.run --input data/input/IMG_4887.MOV
```

Results go to `data/output/<video name>/`: `config.json` (settings used), one JSON per stage (`1_detections.json`,
`2_pool.json`, `3_filtered.json`, `4_tracks.json`, ...) and `video.mp4` (the video with the pool's water and the last
stage's boxes drawn on it). The pool stage needs a trained pool model (step 4 below).

More options:

```bash
.venv/bin/python -m swim.run --input data/input/IMG_4887.MOV --to detection                            # stop after a stage
.venv/bin/python -m swim.run --input data/input/IMG_4887.MOV --from filtering --filter-min-score 0.6   # re-run from a stage, re-using saved results before it
.venv/bin/python -m swim.run --input data/input/IMG_4887.MOV --from tracking --tracker ocsort         # try another tracker
.venv/bin/python -m swim.run --input data/input/IMG_4887.MOV --render-only --layers pool,tracks      # only re-draw video.mp4
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
  can change the filtering and tracking settings (e.g. `min_score`, `pool_margin`, the tracker) and press **Apply**:
  they re-run from the saved detections and pool in under a second, and the results are saved and shown on the video.

Stop the app with `Ctrl+C` (this also stops a video being processed). Options: `--input <folder>` (folder opened
first), `--output <folder>` (another results folder), `--port 8080`. See [docs/app.md](docs/app.md).

### 3. Label pool frames (training data for the pool model)

SAM3-LiteText labels the pool's water and lane ropes on frames picked from the videos; you check and fix them on the
**Labels** page of the app (link at the top of the Videos page). The tools come from the pool-segmentation package
(its `docs/labeling.md`); run them here, from the repo root, so they use `data/labels/pool/`.

```bash
.venv/bin/python -m pool_segmentation.labeling pick data/input/*.MOV   # pick frames (one every 2 s, at most 25 per video)
.venv/bin/python -m pool_segmentation.labeling auto                     # SAM3-LiteText labels them (about 2 s per frame; downloads 2 GB once)
.venv/bin/python -m swim.app                                            # check and fix: http://127.0.0.1:8000/labels/
.venv/bin/python -m pool_segmentation.labeling masks                    # training masks from the checked frames
```

### 4. Train the pool model

A U-Net learns the pool's water (water / other) from the checked frames; about 5 minutes on a MacBook (M4 Pro). See
the pool-segmentation package's `docs/model.md`. The pipeline's pool stage uses the newest run (setting `pool.model`).

```bash
.venv/bin/python -m pool_segmentation.model train   # a run folder in data/models/pool/: model.pt, summary.json, val/ pictures
.venv/bin/tensorboard --logdir data/models/pool   # watch the progress: http://localhost:6006
```

## Tests

```bash
.venv/bin/python -m pytest                    # all tests (the app's browser tests need Chrome, ~4 s)
.venv/bin/python -m pytest -m "not browser"   # without the browser test
```
