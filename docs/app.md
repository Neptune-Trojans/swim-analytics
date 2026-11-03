# App Design (process videos and watch the results in the browser)

_Agreed: 2026-11-02 (Play page), 2026-11-03 (Videos page). Status: both pages built, steps 1–8 (see "Build order")._

Goal: a local web app with two pages:

- **Videos** (the main page, `/`): pick a folder, see its videos and whether each has results and whether those are up
  to date; process them (the whole pipeline) in the background, follow the progress, open them in Play.
- **Play** (`/play?video=<name>`): watch a processed video with the extracted data drawn on top, switch layers on and
  off, step frame by frame, see the numbers behind every box, re-run the fast stages with other settings.

## How it works

```
Browser (swim/app/static/)                  Local server (python -m swim.app, swim/app/server.py)
┌────────────────────────────────┐          ┌─────────────────────────────────────────────────┐
│ Videos page (videos.js)        │ ← list   │ GET /api/library?folder=&settings=  subfolders, │
│  folder bar, subfolders        │          │     videos (size, length) + results state       │
│  settings for processing       │ ← image  │ GET /api/thumbnail?path=   first frame (JPEG)   │
│  one row per video: state,     │ process →│ POST /api/jobs   queue videos (library.py tells │
│  Process / Update, progress,   │ ← status │     the first stage each needs); jobs.py runs   │
│  Cancel, Open                  │ cancel → │     them one at a time through run_pipeline()   │
│                                │          │ GET /api/jobs   progress; POST /api/jobs/<id>/  │
│                                │          │     cancel                                      │
├────────────────────────────────┤          ├─────────────────────────────────────────────────┤
│ Play page (viewer.js)          │ ← JSON   │ GET /api/videos/<name>/results/<stage>          │
│  <video> + <canvas> on top     │ ← video  │ GET /api/videos/<name>/video   original video   │
│  layer toggles, frame stepping │          │     (supports seeking: HTTP range requests)     │
│  timeline, side panel          │ rerun →  │ POST /api/videos/<name>/rerun   re-runs the     │
│  settings panel: Apply         │ ← JSON   │     fast stages (filtering on) via run_pipeline │
└────────────────────────────────┘          └─────────────────────────────────────────────────┘
```

- The browser plays the **original video** (iPhone HEVC `.MOV`; Safari and Chrome on macOS play it, Firefox may not)
  and draws the stage outputs on a canvas on top of it, from the saved JSON files. Nothing is re-rendered when a layer
  is switched on or off.
- The server serves files, lists folders and results, and processes videos. All processing calls `run_pipeline()`
  like `swim/run.py` does, with no processing loop of its own (CLAUDE.md rule). The app saves only the JSON results;
  `video.mp4` (boxes drawn on the video) comes from the command line.
- Results go where `python -m swim.run` puts them: `data/output/<video name>/` (`--output` to change it).
- Local only: the server listens on `127.0.0.1` and refuses requests addressed to any other host name (so a web page
  on another site can't reach it by pointing its own domain at 127.0.0.1, "DNS rebinding"). No login, no hosting.

## Videos page

| Part | What it does |
|---|---|
| **Folder** | A path field, a parent-folder button and the subfolders as buttons. Opens `--input` (default `data/input`) the first time, then the last folder (remembered in the browser; also in the URL: `/?folder=...`, so Back returns to the previous folder). A web page can't tell the server where a folder is (the browser's folder picker gives files, not their location), so the server lists the folders |
| **Rows** | One per video (`.mov`, `.mp4`, `.m4v`): thumbnail (first frame), name, length, frames, size, fps, file size, date; the **state** of its results; one chip per stage: saved ✓, saved with other settings ≠, not run –; what is out of date (e.g. `detection.threshold 0.5 → 0.1 · tracking not run`) |
| **State** | *Not processed*, *Up to date* (every stage saved, with the settings for processing), *Needs update* (a stage missing, other settings, or no `config.json`), *Results of another video* (`data/output/<name>/` was made from another file with the same name, e.g. an iPhone `IMG_4887` from another folder), *Can't read this video* |
| **Actions** | **Process** (no results), **Update** (runs from the first stage that needs it: other tracking settings re-run only tracking, in a second), **Re-process** (up to date: runs everything again). **Open** goes to the Play page. Replacing another video's results asks first |
| **Selection** | Tick videos, **Process selected**; **Process all that need it** queues every video not up to date |
| **Settings for processing** | Every stage's settings, from the defaults in `swim/pipeline/config.py`, with the same form as the Play page. Changes are remembered in the browser; the videos' state is checked against these settings (changing the threshold to 0.5 makes results made at 0.5 *Up to date*) |
| **Progress** | One video at a time; the others wait in line. The row shows the stage, frames done and time left (detection: about 20 frames per second, ~1 min for `IMG_4887`, ~9 min for the 10,000-frame `IMG_4890`), with **Cancel**. The top bar shows what is running, wherever you are. Jobs run on the server: closing the page doesn't stop them; stopping the server does |

### Processing in the background (`swim/app/jobs.py`)

- `POST /api/jobs` with `{"videos": [paths], "settings": {changes}, "replace": false}`: for each video,
  `results_status()` (`swim/app/library.py`) compares its saved results with the settings and gives the first stage to
  run (`start`); an up-to-date video runs from the start. Refused (nothing queued): a path that isn't a video, bad
  settings, two videos with the same name (same results folder), a video already queued or running (409), another
  video's results without `replace` (409).
- One background thread runs the queue in order: `run_pipeline(video, <output>/<name>, config, start=...,
  progress=...)`. The progress callback records the stage and frame for `GET /api/jobs`; **Cancel** makes it raise,
  so the run stops at the next frame.
- A cancelled or failed run leaves consistent results: `run_pipeline()` saves each stage only when it is done, and
  only then deletes the later stages' outputs (made from the old one) and records its settings in `config.json`. So
  cancelling during detection keeps the old results as they were.
- The video path recorded in the results is relative to the repo when the video is inside it
  (`data/input/IMG_4887.MOV`, as `swim.run` records it), absolute otherwise.
- While a video is being processed, the Play page can't re-run its stages (409).
- Jobs are kept in memory: the list starts empty when the server starts.

## Play page (the viewer)

| Part | What it does |
|---|---|
| **Top bar** | The video's name and **← Videos** (back to the Videos page). The URL holds the video and, when paused, the frame (`/play?video=IMG_4887&frame=432`), so a link or a reload opens the same frame. Old links (`/?video=...`) still work |
| **Player** | Original video with a canvas overlay, scaled with the video when the window is resized |
| **Layers** | A checkbox per available layer: `detections`, `filtered`, `tracks` (each track in its own color with its number; later `swimmer`, `pose`, `metrics`). Default: the last stage's layer, as on the command line. Same look as the Python render: kept boxes in red with the score, rejected ones in gray with the reason |
| **Controls** | Play / pause (Space), one frame back / forward (← / →), 10 frames (Shift + ← / →), speed 0.25× / 0.5× / 1× / 2×, frame number and time |
| **Timeline** | One thin bar per frame under the video, colored by the number of persons kept (0 = gap, highlighted), with a mark where boxes were rejected. Click or drag to jump; shows where the playhead is |
| **Side panel** | The current frame's detections (score, box size) and rejected ones (reason, score). Hovering a row highlights its box |
| **Settings** | Every stage's settings (from `config.json`) with a short explanation each (`setting(help=...)` in `swim/pipeline/config.py`). The fast stages' settings (filtering, tracking) can be changed: **Apply** re-runs them from the saved detections and the video shows the new results right away. Detection's settings are read-only here: re-running it takes minutes (Videos page). A hint warns when `min_score` is below the detection threshold (changes nothing: those boxes were never saved) |
| **Checks** | A warning if the results' width × height or frame count don't match the video (wrong results folder) |

### Frame-accurate overlay

The video's `currentTime` is not precise enough to know which frame is on screen. The player uses the browser's
`requestVideoFrameCallback`, which reports the exact time of the frame being shown, and turns it into a frame number:
`frame_idx = round(media_time × fps)`. Measured on our videos: `frame_idx / fps` differs from the real frame
timestamps by at most 0.08 frames (`IMG_4887`), so this maps to the right frame. Stepping seeks to the middle of the
target frame (`(frame_idx + 0.5) / fps`) so rounding never lands on the frame before.

### Re-running stages from the settings panel

`POST /api/videos/<name>/rerun` with e.g. `{"settings": {"filtering": {"min_score": 0.3}}}`:

- Only stages from `filtering` on (`EDITABLE_STAGES` in `server.py`): they need neither the model nor the video
  frames, so a re-run takes well under a second (0.01 s for filtering on `IMG_4886`). A future stage that needs the
  model (pose) must not be re-run this way: it belongs in the Videos page's background jobs.
- Values are checked against each setting's type and range (`PipelineConfig.with_changes()`); a bad value changes
  nothing (HTTP 400). One re-run at a time (409 while one runs, or while the video is being processed).
- Like `python -m swim.run --from filtering`: the new results and `config.json` are **saved**, replacing the
  video's previous ones. `video.mp4` (command-line render) is not redrawn.

### Drawing layers in two places

The Python renderer (`swim/visualization/`) stays, for exporting a video file. The browser draws the same layers in
JS (`swim/app/static/layers.js`). Each new stage therefore adds a drawing layer in both places (CLAUDE.md
"Adding a stage" rule).

## Folder structure

```
swim/app/
  __main__.py      # python -m swim.app [--input data/input] [--output data/output] [--port 8000]
  server.py        # FastAPI app: the API above, the two pages
  library.py       # videos in a folder, thumbnails, state of their results (results_status)
  jobs.py          # JobQueue: background processing through run_pipeline(), progress, cancel
  static/
    videos.html    # Videos page (/)
    videos.js
    play.html      # Play page (/play)
    viewer.js      # Play page: loading, player, controls, timeline, side panel, settings
    layers.js      # one drawing function per layer (mirrors swim/visualization/ + LAYERS in render.py)
    settings.js    # settings form rows, shared by both pages
    dom.js         # elements, error banner, JSON requests, shared by both pages
    style.css
tests/app/
  conftest.py      # small test videos; a stand-in for the detection model
  test_server.py   # FastAPI TestClient: listing, JSON, video with range requests, library, thumbnails, host check
  test_library.py  # folder listing, results state, thumbnails
  test_jobs.py     # processing: new video, update from the needed stage, queue, cancel, refusals
  test_viewer_browser.py  # headless Chrome: Play page frame accuracy and settings; Videos page process + open
```

- Plain HTML / CSS / JS (ES modules), no build step, no JS framework.
- The server uses `STAGES` / `STAGE_OUTPUTS` from `swim/pipeline/runner.py` for the stage order and file names,
  so new stages show up without changing the server.
- The Play page's video path comes from the results' `video_path` (as `--render-only` uses `--input`); it only serves
  files of listed results, never paths taken from the request. The Videos page lists any folder on this computer
  and makes thumbnails of video files in it (local only, see above).
- Packages in `pyproject.toml`: an `app` group with `fastapi`, `uvicorn`; `httpx2` in `dev` (FastAPI's test client).

## Build order

Each step leaves something that works.

Play page (viewer):

- [x] 1. Server: `python -m swim.app` serves the page, the list of processed videos, the video (with seeking) and the
  stage JSON files. Tests in `tests/app/`.
- [x] 2. Player: video + canvas overlay drawing the `detections` and `filtered` layers, frame-accurate; layer toggles;
  video list.
- [x] 3. Controls and timeline: frame stepping, speed, keyboard shortcuts, the per-frame timeline.
- [x] 4. Side panel and checks: current frame's boxes, settings used, size / frame count warnings. Update README,
  CLAUDE.md ("Adding a stage" gets the JS layer).
- [x] 5. Settings panel: every setting with a short explanation; change the fast stages' settings (filtering,
  tracking) and re-run them from the saved detections (`POST /rerun`).

Videos page (replaces the planned "Process mode", which uploaded a video into the page: copying large videos is slow
and they already sit in a folder):

- [x] 6. `run_pipeline(progress=...)`: stage + frame, may raise to cancel; results stay consistent when a run stops
  halfway (each stage saved, later outputs removed and settings recorded only when it is done).
- [x] 7. Server: folder listing with each video's results state, thumbnails, background jobs (one at a time) with
  progress and cancel; host check.
- [x] 8. Videos page as the main page: folder bar, rows with state and stage chips, Process / Update / Re-process,
  selection, settings for processing, progress, Cancel, Open; the Play page links back.

## How it is checked

- Server: unit tests (`.venv/bin/python -m pytest`). Processing tests use a stand-in for the detection model, so they
  run in a second.
- Viewer: open `IMG_4887` in Chrome and look at paused frames, while playing and while stepping: boxes sit on the
  swimmer, a frame with rejected boxes shows them in gray, a gap frame shows nothing and is highlighted in the
  timeline.
- Frame accuracy, automated: `tests/app/test_viewer_browser.py` drives headless Chrome on a generated H.264 video
  whose frames show their own index (black / white squares), with one box per frame at a position that depends on
  the index. The frame label, the picture on screen and the box the overlay draws must all be the same frame:
  opening at a frame, stepping with keys, a timeline click, and every frame while playing. Also: changing
  `min_score` in the settings panel re-runs filtering and the page shows the new results; on the Videos page,
  processing the test video and opening it. Skipped without Chrome.
- Frame accuracy on a real iPhone video (2026-11-02, once by hand): the picture Chrome shows of `IMG_4887` was
  compared with OpenCV's decoded frames n−3…n+3; the best match was always the frame number the viewer shows.
- Videos page on `data/input` (2026-11-03, by hand, results in a temp folder): the saved results made at threshold
  0.5 show *Needs update* (`detection.threshold 0.5 → 0.1`), and *Up to date* once the threshold is set to 0.5 in
  the settings; processing `IMG_4886` (93 frames) with the real model took 11 s including loading it, progress and
  time left shown, then *Up to date* and Open showed its tracks; cancelling `IMG_4885` at frame 109 stopped it within
  0.3 s and left no results.

## Not now

- Processing in the browser itself (running the model in JS): much harder and slower.
- Uploading or dragging a video into the page (the Videos page reads folders instead).
- Deleting results from the page; running several videos at once.
- Exporting the video from the browser (the command line's `video.mp4` does that).
- Editing or labelling boxes.
- Phone layout: desktop first (the Videos page's rows do fit a phone's width).
- Hosting online (logins, file storage, a GPU server).
