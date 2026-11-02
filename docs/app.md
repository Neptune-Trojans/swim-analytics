# App Design (viewer + processing in the browser)

_Agreed: 2026-11-02. Status: Play mode (viewer) built, steps 1–4; Process mode planned (see "Build order")._

Goal: a local web app with two modes:

- **Play**: watch a processed video with the extracted data drawn on top, switch layers on and off, step frame by
  frame, see the numbers behind every box.
- **Process**: drop a video in, choose settings, run the pipeline, watch its progress, then open it in Play.

We start with **Play**, using results that already exist (e.g. `data/output/IMG_4887/`). Process comes after.

## How it works

```
Browser (swim/app/static/)                  Local server (python -m swim.app, swim/app/server.py)
┌────────────────────────────────┐          ┌─────────────────────────────────────────────────┐
│ Play                           │ ← list   │ GET /api/videos           processed videos      │
│  <video> + <canvas> on top     │ ← JSON   │ GET /api/videos/<name>/<stage>.json             │
│  layer toggles, frame stepping │ ← video  │ GET /api/videos/<name>/video   original video   │
│  timeline, side panel          │          │     (supports seeking: HTTP range requests)     │
├────────────────────────────────┤          │                                                 │
│ Process (later)                │ upload → │ saves to data/input/, runs run_pipeline() in    │
│  drop video, settings, progress│ ← status │ the background, reports progress                │
└────────────────────────────────┘          └─────────────────────────────────────────────────┘
```

- The browser plays the **original video** (iPhone HEVC `.MOV`; Safari and Chrome on macOS play it, Firefox may not)
  and draws the stage outputs on a canvas on top of it, from the saved JSON files. Nothing is re-rendered when a layer
  is switched on or off.
- The server only serves files and lists results. In Process mode it calls `run_pipeline()` like `swim/run.py` does,
  with no processing loop of its own (CLAUDE.md rule).
- Local only: the server listens on `127.0.0.1`, no login, no hosting.

## Play mode (the viewer)

| Part | What it does |
|---|---|
| **Video list** | Processed videos found in `data/output/` (folders with at least one stage result); one whose video file is missing is listed but can't be opened. The URL holds the video and, when paused, the frame (`?video=IMG_4887&frame=432`), so a link or a reload opens the same frame |
| **Player** | Original video with a canvas overlay, scaled with the video when the window is resized |
| **Layers** | A checkbox per available layer: `detections`, `filtered` (later `tracks`, `swimmer`, `pose`, `metrics`). Default: the last stage's layer, as on the command line. Same look as the Python render: kept boxes in red with the score, rejected ones in gray with the reason |
| **Controls** | Play / pause (Space), one frame back / forward (← / →), 10 frames (Shift + ← / →), speed 0.25× / 0.5× / 1× / 2×, frame number and time |
| **Timeline** | One thin bar per frame under the video, colored by the number of persons kept (0 = gap, highlighted), with a mark where boxes were rejected. Click or drag to jump; shows where the playhead is |
| **Side panel** | The current frame's detections (score, box size) and rejected ones (reason, score); the settings the results were made with (`config.json`) |
| **Checks** | A warning if the results' width × height or frame count don't match the video (wrong results folder) |

### Frame-accurate overlay

The video's `currentTime` is not precise enough to know which frame is on screen. The player uses the browser's
`requestVideoFrameCallback`, which reports the exact time of the frame being shown, and turns it into a frame number:
`frame_idx = round(media_time × fps)`. Measured on our videos: `frame_idx / fps` differs from the real frame
timestamps by at most 0.08 frames (`IMG_4887`), so this maps to the right frame. Stepping seeks to the middle of the
target frame (`(frame_idx + 0.5) / fps`) so rounding never lands on the frame before.

### Drawing layers in two places

The Python renderer (`swim/visualization/`) stays, for exporting a video file. The browser draws the same layers in
JS (`swim/app/static/layers.js`). Each new stage therefore adds a drawing layer in both places (CLAUDE.md
"Adding a stage" rule gets this added).

## Folder structure

```
swim/app/
  __main__.py      # python -m swim.app [--output data/output] [--port 8000]: starts the server
  server.py        # FastAPI app: list videos, serve video + stage JSON (later: upload, run, progress)
  static/
    index.html     # page layout
    style.css
    viewer.js      # Play mode: loading, player, controls, timeline, side panel
    layers.js      # one drawing function per layer (mirrors swim/visualization/ + LAYERS in render.py)
tests/app/
  test_server.py   # FastAPI TestClient: listing, JSON, video with range requests, bad names rejected
```

- Plain HTML / CSS / JS (ES modules), no build step, no JS framework.
- The server uses `STAGES` / `STAGE_OUTPUTS` from `swim/pipeline/runner.py` for the stage order and file names,
  so new stages show up without changing the server.
- The video path comes from the results' `video_path` (as `--render-only` uses `--input`); the server only serves
  files of listed results, never paths taken from the request.
- New packages in `pyproject.toml`: an `app` group with `fastapi`, `uvicorn`; `httpx2` in `dev` (FastAPI's test client).

## Build order

Each step leaves something that works.

Play mode (viewer):

- [x] 1. Server: `python -m swim.app` serves the page, the list of processed videos, the video (with seeking) and the
  stage JSON files. Tests in `tests/app/`.
- [x] 2. Player: video + canvas overlay drawing the `detections` and `filtered` layers, frame-accurate; layer toggles;
  video list.
- [x] 3. Controls and timeline: frame stepping, speed, keyboard shortcuts, the per-frame timeline.
- [x] 4. Side panel and checks: current frame's boxes, settings used, size / frame count warnings. Update README,
  CLAUDE.md ("Adding a stage" gets the JS layer).

Process mode (later):

- [ ] 5. Progress callback in `run_pipeline()` (stage + frame), used by the command line and the app.
- [ ] 6. Server: upload a video, run the pipeline in the background (one job at a time), report progress.
- [ ] 7. Process page: drop zone, settings form (from `PipelineConfig`), progress per stage, re-run from a stage,
  "Open in Play" when done.

## How it is checked

- Server: unit tests (`.venv/bin/python -m pytest`).
- Viewer: open `IMG_4887` in Chrome and look at paused frames, while playing and while stepping: boxes sit on the
  swimmer, a frame with rejected boxes shows them in gray, a gap frame shows nothing and is highlighted in the
  timeline.
- Frame accuracy, automated: `tests/app/test_viewer_browser.py` drives headless Chrome on a generated H.264 video
  whose frames show their own index (black / white squares), with one box per frame at a position that depends on
  the index. The frame label, the picture on screen and the box the overlay draws must all be the same frame:
  opening at a frame, stepping with keys, a timeline click, and every frame while playing. Skipped without Chrome.
- Frame accuracy on a real iPhone video (2026-11-02, once by hand): the picture Chrome shows of `IMG_4887` was
  compared with OpenCV's decoded frames n−3…n+3; the best match was always the frame number the viewer shows.

## Not now

- Processing in the browser itself (running the model in JS): much harder and slower.
- Exporting the video from the browser (the command line's `video.mp4` does that).
- Editing or labelling boxes.
- Phone layout: desktop first.
- Hosting online (logins, file storage, a GPU server).
