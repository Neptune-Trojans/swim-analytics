"""Local web server for the browser app (docs/app.md).

- Videos page (`/`): the videos in a folder, the state of their results, processing in the background (jobs.py).
- Play page (`/play?video=<name>`): a processed video with its results drawn on top; re-runs the fast stages with
  settings changed in the page.

Start it with `python -m swim.app`. Processing goes through run_pipeline(), as on the command line.
"""

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from swim.app.jobs import JobQueue
from swim.app.library import (
    VIDEO_MEDIA_TYPES, find_results, is_video, list_folder, results_status, results_video, saved_config, saved_stages,
    thumbnail, video_info,
)
from swim.pipeline.config import PipelineConfig, describe
from swim.pipeline.runner import STAGE_OUTPUTS, STAGES, run_pipeline

STATIC_DIR = Path(__file__).parent / "static"

# The Play page can change the settings of the stages from RERUN_FROM on and re-run them from the saved detections:
# they need neither the model nor the video frames, so they take seconds. Detection takes minutes: Videos page.
RERUN_FROM = "filtering"
EDITABLE_STAGES = STAGES[STAGES.index(RERUN_FROM):]

Settings = dict[str, dict[str, float | int | str]]


class RerunRequest(BaseModel):
    settings: Settings = {}


class ProcessRequest(BaseModel):
    videos: list[str]  # absolute paths, as the Videos page lists them
    settings: Settings = {}  # changes to the default settings
    replace: bool = False  # allowed to replace results that were made from another video with the same name


def create_app(output_dir: str | Path = "data/output", input_dir: str | Path = "data/input") -> FastAPI:
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    app = FastAPI(title="swim-analytics")
    # Only requests addressed to this computer: a web page on another site can't reach the server by pointing its
    # own domain name at 127.0.0.1 (DNS rebinding), now that the server lists folders and runs jobs
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    jobs = JobQueue(output_dir)
    app.state.jobs = jobs

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        # Results change when a stage is re-run and the page's files change while developing: always revalidate
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    def wanted_settings(changes: Settings) -> PipelineConfig:
        try:
            return PipelineConfig().with_changes(changes)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    # -----------------------------------------------------------------------------------------------------------
    # Videos page: folders, videos, processing

    @app.get("/api/library")
    def library(folder: str | None = None, settings: str = "{}") -> dict:
        """Subfolders and videos of `folder` (default: the input folder), each video with the state of its results
        compared with the default settings plus `settings` (JSON, e.g. {"detection": {"threshold": 0.3}})."""
        try:
            changes = json.loads(settings)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"settings is not JSON: {e}") from e
        wanted = wanted_settings(changes if isinstance(changes, dict) else {})
        path = Path(folder).expanduser() if folder else input_dir
        path = path.resolve()
        if not path.is_dir():
            raise HTTPException(404, f"Folder not found: {path}")
        try:
            listing = list_folder(path, output_dir, wanted)
        except PermissionError as e:
            raise HTTPException(403, f"Can't read {path}: {e.strerror}") from e
        return {**listing, "stages": STAGES, "output": str(output_dir.resolve())}

    @app.get("/api/settings")
    def default_settings() -> dict:
        """Every stage's default settings, with explanations and allowed ranges or values."""
        defaults = PipelineConfig()
        return {stage: describe(getattr(defaults, stage)) for stage in STAGES}

    @app.get("/api/thumbnail")
    def video_thumbnail(path: str) -> Response:
        video = Path(path)
        if not video.is_absolute() or not is_video(video):
            raise HTTPException(404, f"Not a video: {path}")
        try:
            return Response(thumbnail(video), media_type="image/jpeg")
        except OSError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return jobs.jobs()

    @app.post("/api/jobs")
    def process(request: ProcessRequest) -> list[dict]:
        """Queue videos: each runs from the first stage its results need (everything if they are up to date)."""
        wanted = wanted_settings(request.settings)
        videos = [Path(v) for v in request.videos]
        for video in videos:
            if not video.is_absolute() or not is_video(video):
                raise HTTPException(400, f"Not a video: {video}")
        names = [v.stem for v in videos]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise HTTPException(400, f"Several videos named {', '.join(repeated)}: their results would go to the same "
                                     f"folder. Process them one at a time")
        busy = [n for n in names if jobs.busy(n)]
        if busy:
            raise HTTPException(409, f"Already being processed or waiting: {', '.join(busy)}")
        starts = []
        for video in videos:
            status = results_status(video, output_dir, wanted)
            if status["made_from"] and not request.replace:
                raise HTTPException(409, f"The results in {output_dir / video.stem} were made from "
                                         f"{status['made_from']}: send replace to replace them")
            starts.append(status["start"] or STAGES[0])
        for video, start in zip(videos, starts):
            jobs.add(video, wanted, start)
        return jobs.jobs()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: int) -> list[dict]:
        if not jobs.cancel(job_id):
            raise HTTPException(404, f"No job {job_id}")
        return jobs.jobs()

    # -----------------------------------------------------------------------------------------------------------
    # Play page: one processed video, by its results folder name

    def results_folder(name: str) -> Path:
        # Only names of listed result folders are accepted, so a request can't reach any other file
        folder = find_results(output_dir).get(name)
        if folder is None:
            raise HTTPException(404, f"No results for '{name}' in {output_dir}")
        return folder

    @app.get("/api/videos")
    def list_videos() -> list[dict]:
        return [
            {"name": name, "stages": saved_stages(folder), "video_found": results_video(folder).is_file()}
            for name, folder in find_results(output_dir).items()
        ]

    @app.get("/api/videos/{name}")
    def video_details(name: str) -> dict:
        """Saved stages, every stage's settings (values from config.json, with explanations and allowed ranges),
        which of them the page can change, and the video's size, fps and frame count."""
        folder = results_folder(name)
        path = results_video(folder)
        try:
            info = video_info(path)
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
                                     f"{', '.join(EDITABLE_STAGES)}): process the video again on the Videos page")
        try:
            config = saved_config(folder).with_changes(request.settings)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if jobs.busy(name):
            raise HTTPException(409, f"'{name}' is being processed on the Videos page: wait until it is done")
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A re-run is already in progress")
        try:
            start = time.perf_counter()
            run_pipeline(results_video(folder), folder, config, start=RERUN_FROM)
            seconds = time.perf_counter() - start
        except OSError as e:  # video missing, or no saved detections
            raise HTTPException(400, str(e)) from e
        finally:
            run_lock.release()
        return {**video_details(name), "seconds": round(seconds, 2)}

    @app.get("/api/videos/{name}/video")
    def video_file(name: str) -> FileResponse:
        # FileResponse answers HTTP range requests, which browsers need to seek in a video
        path = results_video(results_folder(name))
        if not path.is_file():
            raise HTTPException(404, f"Video not found: {path}")
        return FileResponse(path, media_type=VIDEO_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"))

    @app.get("/api/videos/{name}/results/{stage}")
    def stage_results(name: str, stage: str) -> FileResponse:
        folder = results_folder(name)
        if stage not in STAGES or not (folder / STAGE_OUTPUTS[stage][0]).exists():
            raise HTTPException(404, f"No saved '{stage}' results for '{name}'")
        return FileResponse(folder / STAGE_OUTPUTS[stage][0], media_type="application/json")

    # -----------------------------------------------------------------------------------------------------------
    # Pages

    @app.get("/")
    def videos_page(request: Request) -> Response:
        if "video" in request.query_params:  # links from before the Videos page existed: /?video=<name>&frame=<n>
            return RedirectResponse(f"/play?{request.url.query}")
        return FileResponse(STATIC_DIR / "videos.html")

    @app.get("/play")
    def play_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "play.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
