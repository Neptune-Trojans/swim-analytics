# swim-analytics

Swimming video analysis: detect the swimmer, then keypoints, metrics and overlays on the video.

## Rules

- **Commit and push only when the user asks for it.** Never commit or push on your own initiative; the user reviews
  changes first.
- **Every research request gets its own file in `research/`** (`research/<topic>.md`), even if the answer is also
  given in chat. Start it with `_Research date: YYYY-MM-DD_` and end it with a sources/references list with links.
  Papers worth reading also go in `research/reading-list.md`.

## Environment

- Python venv in `.venv/`: run everything with `.venv/bin/python`
- `rfdetr` is installed in `.venv` but not yet listed in `pyproject.toml`

## Conventions

- Pipeline design and build progress: `docs/pipeline.md`. Keep its "Build order" checkboxes up to date.
- All processing goes through `run_pipeline()` in `swim/pipeline/runner.py`, run with `python -m swim.run`.
  No separate scripts with their own processing loops.
- `swim/pipeline/` is the runner (order, save/load, settings); `swim/stages/` holds the steps
- One folder per stage under `swim/stages/` (`detection/`, later `filtering/`, `tracking/`, `pose/`, `metrics/`),
  each with: `types.py` (output data classes + model contract, no model-specific code), `stage.py` (the function
  the runner calls), and one file per model or rule
- Adding a stage: add it to `STAGES` / `STAGE_OUTPUTS` / `PipelineResult` in `swim/pipeline/runner.py`, its settings
  to `swim/pipeline/config.py`, and its drawing layer to `swim/visualization/` + `LAYERS` in `render.py`
- Import heavy model packages (torch, rfdetr) inside the function that creates the model, so `--render-only` and
  loading saved results stay fast
- `swim/utils/`: shared helpers used by several parts (runner, stages, visualization). Two rules:
  - one file per topic, named after it (`video.py`, later e.g. `json_io.py`); never `helpers.py` / `misc.py`
  - only stage-independent code; anything used by a single stage stays in that stage's folder
- `swim/utils/video.py`: use its helpers instead of raw `cv2.VideoCapture` / `cv2.VideoWriter`

## Testing

- Short clip: `data/input/IMG_4886.MOV` (93 frames, one swimmer); several persons: `IMG_4884.MOV`
- Write test outputs to a temp folder (`--output <tmp>`), not `data/output/` (the user's own results)
- Look at a few output frames, not only whether the script ran

## Workflow

- Work directly on `main`

## Gotchas

- `cv2.VideoWriter` fails silently on a bad path (`video_writer()` checks for this)
- rfdetr >= 1.9 removed `rfdetr.util`; COCO class names are in `rfdetr.assets.coco_classes`
