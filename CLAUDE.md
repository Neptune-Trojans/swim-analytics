# swim-analytics

Swimming video analysis: detect the swimmer, then keypoints, metrics and overlays on the video.

## Rules

- **Commit and push only when the user asks for it.** Never commit or push on your own initiative; the user reviews
  changes first.
- **Every research request gets its own file in `research/`** (`research/<topic>.md`), even if the answer is also
  given in chat. Start it with `_Research date: YYYY-MM-DD_` and end it with a sources/references list with links.
  Papers worth reading also go in `research/reading-list.md`. `research/` is git-ignored (local only).

## Environment

- Python venv in `.venv/`: run everything with `.venv/bin/python`
- Dependencies are in `pyproject.toml`: `.venv/bin/pip install -e ".[app,dev,notebooks,train]"`
- The pool model, its training data tools and the Labels page are the **pool-segmentation** package, a separate
  repo checked out next to this one (`../pool-segmentation`, its own `CLAUDE.md`), installed editable in this venv
  (`.venv/bin/pip install -e ../pool-segmentation`): changes there are live here. Changes to it go in that repo

## Conventions

- Pipeline design and build progress: `docs/pipeline.md`. Keep its "Build order" checkboxes up to date.
- All processing goes through `run_pipeline()` in `swim/pipeline/runner.py`, run with `python -m swim.run`.
  No separate scripts with their own processing loops. Exceptions, not part of the pipeline: the pool model's
  training data (`python -m pool_segmentation.labeling`) and training (`python -m pool_segmentation.model train`),
  both from the pool-segmentation package. The labeling dataset lives in that repo
  (`../pool-segmentation/data/labels/pool/`, git-ignored there); the trained models stay here in `data/models/pool/`
  (git-ignored), where the pool stage finds them
- `swim/pipeline/` is the runner (order, save/load, settings); `swim/stages/` holds the steps
- One folder per stage under `swim/stages/` (`detection/`, `pool/`, `filtering/`, `tracking/`, later `swimmer/`, `pose/`, `metrics/`),
  each with: `types.py` (output data classes + model contract, no model-specific code; only if the stage has its
  own output format), `stage.py` (the function the runner calls), and one file per model or rule
- Filtering outputs the same `VideoDetections` as detection: `detections` = kept, `rejected` = removed + reason.
  Later stages use only `detections`.
- Adding a stage: add it to `STAGES` / `STAGE_OUTPUTS` / `PipelineResult` / `load_results()` in
  `swim/pipeline/runner.py`, its settings to `swim/pipeline/config.py` as a field named exactly like the stage
  (the runner relies on this), each setting declared with `setting(default, help, min, max, step)`: a short `help`
  (the viewer's settings panel shows it, and checks the range), and its drawing layer to `swim/visualization/` + `LAYERS` in `render.py` and to
  `swim/app/static/layers.js` (`LAYERS` there too: same layer name and look, for the browser viewer)
- `swim/app/`: the local web app (`python -m swim.app`, design in `docs/app.md`): Videos page (process the videos of a
  folder in the background, `jobs.py`), Play page (watch results) and, at `/labels/`, the pool-segmentation package's
  Labels page (check and fix training labels; mounted in `server.py`). Processing goes through `run_pipeline()` there
  too (its `progress` callback reports frames and cancels). Plain HTML / CSS / JS (ES modules), no build step, no
  framework
- Import heavy model packages (torch, rfdetr) inside the function that creates the model, so `--render-only` and
  loading saved results stay fast
- `swim/utils/`: shared helpers used by several parts (runner, stages, visualization). Two rules:
  - one file per topic, named after it (`video.py`, later e.g. `json_io.py`); never `helpers.py` / `misc.py`
  - only stage-independent code; anything used by a single stage stays in that stage's folder
- `swim/utils/video.py`: use its helpers instead of raw `cv2.VideoCapture` / `cv2.VideoWriter`

## Testing

- Unit tests: `.venv/bin/python -m pytest`; files in `tests/`, same folder structure as `swim/`
- `tests/app/test_viewer_browser.py` drives headless Chrome (skipped without it, ~3 s); leave it out with
  `-m "not browser"`. Run it after changing `swim/app/static/`
- Tests that run detection use the stand-in model in `tests/app/conftest.py` (`fake_model`), not RF-DETR. The pool
  stage always uses a stand-in in tests (`tests/conftest.py`, autouse): no test loads a trained pool model
- Short clip: `data/input/IMG_4886.MOV` (93 frames, one swimmer); several persons: `IMG_4884.MOV`
- Write test outputs to a temp folder (`--output <tmp>`), not `data/output/` (the user's own results) or
  `../pool-segmentation/data/labels/` (the user's labeling dataset: checked labels are hand work)
- Look at a few output frames, not only whether the script ran

## Workflow

- Work directly on `main`

## Gotchas

- `cv2.VideoWriter` fails silently on a bad path (`video_writer()` checks for this)
- rfdetr >= 1.9 removed `rfdetr.util`; COCO class names are in `rfdetr.assets.coco_classes`
