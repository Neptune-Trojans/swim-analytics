"""Local web server for the browser app (docs/app.md): lists processed videos, serves their video and stage results,
and re-runs the fast stages with settings changed in the page.

Start it with `python -m swim.app`. Processing goes through run_pipeline(), as on the command line.
"""

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from swim.pipeline.config import PipelineConfig, describe
from swim.pipeline.runner import STAGE_OUTPUTS, STAGES, run_pipeline
from swim.utils.video import get_video_info

STATIC_DIR = Path(__file__).parent / "static"

VIDEO_MEDIA_TYPES = {".mov": "video/quicktime", ".mp4": "video/mp4", ".m4v": "video/mp4"}

# The page can change the settings of the stages from RERUN_FROM on and re-run them from the saved detections: they
# need neither the model nor the video frames, so they take seconds. Detection takes minutes: command line only.
RERUN_FROM = "filtering"
EDITABLE_STAGES = STAGES[STAGES.index(RERUN_FROM):]


class RerunRequest(BaseModel):
    settings: dict[str, dict[str, float | int | str]] = {}


def find_results(output_dir: Path) -> dict[str, Path]:
    """Folders in `output_dir` with at least one saved stage output: {video name: folder}, sorted by name."""
    if not output_dir.is_dir():
        return {}
    return {folder.name: folder for folder in sorted(output_dir.iterdir()) if folder.is_dir() and saved_stages(folder)}


def saved_stages(folder: Path) -> list[str]:
    """Stages with a saved output in `folder`, in pipeline order."""
    return [name for name in STAGES if (folder / STAGE_OUTPUTS[name][0]).exists()]


def video_path(folder: Path) -> Path:
    """The video the results were made from, as recorded in the first saved stage output.

    A relative path is relative to the folder the pipeline was run from (the repo root), like `--input`.
    """
    first = folder / STAGE_OUTPUTS[saved_stages(folder)[0]][0]
    return Path(json.loads(first.read_text())["video_path"])


def saved_config(folder: Path) -> PipelineConfig:
    """The settings the results in `folder` were made with (defaults if there is no config.json)."""
    path = folder / "config.json"
    return PipelineConfig.from_json(path) if path.exists() else PipelineConfig()


def create_app(output_dir: str | Path = "data/output") -> FastAPI:
    output_dir = Path(output_dir)
    app = FastAPI(title="swim-analytics")

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        # Results change when a stage is re-run and the page's files change while developing: always revalidate
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    def results_folder(name: str) -> Path:
        # Only names of listed result folders are accepted, so a request can't reach any other file
        folder = find_results(output_dir).get(name)
        if folder is None:
            raise HTTPException(404, f"No results for '{name}' in {output_dir}")
        return folder

    @app.get("/api/videos")
    def list_videos() -> list[dict]:
        return [
            {"name": name, "stages": saved_stages(folder), "video_found": video_path(folder).is_file()}
            for name, folder in find_results(output_dir).items()
        ]

    @app.get("/api/videos/{name}")
    def video_details(name: str) -> dict:
        """Saved stages, every stage's settings (values from config.json, with explanations and allowed ranges),
        which of them the page can change, and the video's size, fps and frame count."""
        folder = results_folder(name)
        path = video_path(folder)
        try:
            info = asdict(get_video_info(path))
            del info["path"]
        except OSError:
            info = None  # video moved, deleted, or unreadable
        config = saved_config(folder)
        return {
            "name": name,
            "stages": saved_stages(folder),
            "settings": {stage: describe(getattr(config, stage)) for stage in STAGES},
            "settings_saved": (folder / "config.json").exists(),
            "editable_stages": EDITABLE_STAGES,
            "video_path": str(path),
            "video": info,
        }

    run_lock = threading.Lock()

    @app.post("/api/videos/{name}/rerun")
    def rerun(name: str, request: RerunRequest) -> dict:
        """Re-run the stages from RERUN_FROM on with some settings changed, from the saved detections.

        The results and config.json are saved, as with `python -m swim.run --from filtering`. Returns the new details
        and how long it took.
        """
        folder = results_folder(name)
        not_editable = sorted(set(request.settings) - set(EDITABLE_STAGES))
        if not_editable:
            raise HTTPException(400, f"Settings of {', '.join(not_editable)} can't be changed here (only "
                                     f"{', '.join(EDITABLE_STAGES)}): use the command line")
        try:
            config = saved_config(folder).with_changes(request.settings)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A re-run is already in progress")
        try:
            start = time.perf_counter()
            run_pipeline(video_path(folder), folder, config, start=RERUN_FROM)
            seconds = time.perf_counter() - start
        except OSError as e:  # video missing, or no saved detections
            raise HTTPException(400, str(e)) from e
        finally:
            run_lock.release()
        return {**video_details(name), "seconds": round(seconds, 2)}

    @app.get("/api/videos/{name}/video")
    def video_file(name: str) -> FileResponse:
        # FileResponse answers HTTP range requests, which browsers need to seek in a video
        path = video_path(results_folder(name))
        if not path.is_file():
            raise HTTPException(404, f"Video not found: {path}")
        return FileResponse(path, media_type=VIDEO_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"))

    @app.get("/api/videos/{name}/results/{stage}")
    def stage_results(name: str, stage: str) -> FileResponse:
        folder = results_folder(name)
        if stage not in STAGES or not (folder / STAGE_OUTPUTS[stage][0]).exists():
            raise HTTPException(404, f"No saved '{stage}' results for '{name}'")
        return FileResponse(folder / STAGE_OUTPUTS[stage][0], media_type="application/json")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
