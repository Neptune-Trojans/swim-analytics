# swim-analytics

Process a swimming video and overlay analysis layers (skeleton, joint angles, velocity, speed) on it.

The analysis is one pipeline of stages: detection → filtering → tracking → swimmer → pose → metrics.
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
  visualization/      # one drawing layer per stage output
    detections.py     #   draw detection boxes (rejected ones in gray with the reason)
    render.py         #   draw the chosen layers on the video
  utils/              # shared helpers, not specific to any stage (one file per topic)
    video.py          #   read frames, write video, video info
  app/                # local web app: python -m swim.app (docs/app.md)
    server.py         #   lists processed videos, serves video + stage results
    static/           #   the page: viewer.js (Play mode), layers.js (drawing layers, mirrors visualization/)
tests/                # unit tests, same folder structure as swim/
data/
  input/              # source swim videos
  output/             # results, one folder per video
docs/                 # design documents
notebooks/            # experiments (detection_stats.ipynb: statistics of detected boxes)
research/             # research notes (one .md per topic; local only, git-ignored)
```

## Usage

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[app,dev,notebooks]"   # dependencies from pyproject.toml (app: viewer, dev: tests)

.venv/bin/python -m swim.run --input data/input/IMG_4884.MOV                  # run all stages + video
.venv/bin/python -m swim.run --input data/input/IMG_4884.MOV --render-only    # re-draw saved results only
.venv/bin/python -m swim.run --help

.venv/bin/python -m swim.app        # viewer: open http://127.0.0.1:8000 (Safari or Chrome: iPhone videos are HEVC)
```

## Tests

```bash
.venv/bin/python -m pytest
```

Results are saved to `data/output/<video name>/`: `config.json`, one JSON per stage (`1_detections.json`, ...)
and `video.mp4`.
