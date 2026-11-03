"""The videos in a folder and the state of their saved results, for the app's Videos page (docs/app.md).

A video's results are in `<output>/<video name>/`, as `python -m swim.run` saves them. Their state is compared with
the settings chosen for processing: which stages are missing or were made with other settings, and so the first
stage to re-run.
"""

import json
import os
from dataclasses import asdict, fields
from functools import lru_cache
from pathlib import Path

import cv2

from swim.pipeline.config import PipelineConfig
from swim.pipeline.runner import STAGE_OUTPUTS, STAGES
from swim.utils.video import get_video_info, read_frames

# Videos the page lists (the browser plays these; iPhone .MOV is HEVC: Safari and Chrome on macOS)
VIDEO_MEDIA_TYPES = {".mov": "video/quicktime", ".mp4": "video/mp4", ".m4v": "video/mp4"}

THUMBNAIL_WIDTH = 320


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_MEDIA_TYPES and path.is_file() and not path.name.startswith(".")


# ---------------------------------------------------------------------------------------------------------------
# Saved results of one video

def find_results(output_dir: Path) -> dict[str, Path]:
    """Folders in `output_dir` with at least one saved stage output: {video name: folder}, sorted by name."""
    if not output_dir.is_dir():
        return {}
    return {folder.name: folder for folder in sorted(output_dir.iterdir()) if folder.is_dir() and saved_stages(folder)}


def saved_stages(folder: Path) -> list[str]:
    """Stages with a saved output in `folder`, in pipeline order."""
    return [name for name in STAGES if (folder / STAGE_OUTPUTS[name][0]).exists()]


def results_video(folder: Path) -> Path:
    """The video the results were made from, as recorded in the first saved stage output.

    A relative path is relative to the folder the pipeline was run from (the repo root), like `--input`.
    """
    first = folder / STAGE_OUTPUTS[saved_stages(folder)[0]][0]
    return Path(json.loads(first.read_text())["video_path"])


def saved_config(folder: Path) -> PipelineConfig:
    """The settings the results in `folder` were made with (defaults if there is no config.json)."""
    path = folder / "config.json"
    return PipelineConfig.from_json(path) if path.exists() else PipelineConfig()


def same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:  # one of them doesn't exist
        return a.resolve() == b.resolve()


def stored_path(video: Path) -> Path:
    """The video path to record in the results: relative to the current folder (the repo root) when inside it, as
    `python -m swim.run --input data/input/...` records it, so the results stay valid if the repo moves."""
    try:
        return video.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return video.resolve()


def results_status(video: Path, output_dir: Path, wanted: PipelineConfig) -> dict:
    """State of the video's saved results, compared with the settings `wanted` for processing.

    - stages: the stages with saved results
    - made_from: when the results in `<output>/<video name>/` were made from another video file (same name), its path
    - settings_saved: false for results without config.json (settings unknown)
    - differences: settings of saved stages that differ from `wanted`
    - start: the first stage to run to bring the results up to date (missing, other settings, or everything for
      another video's results); None = up to date
    """
    folder = output_dir / video.stem
    stages = saved_stages(folder) if folder.is_dir() else []
    status = {"name": video.stem, "stages": stages, "made_from": None, "settings_saved": False, "differences": [],
              "start": STAGES[0]}
    if not stages:
        return status
    made_from = results_video(folder)
    if not same_file(made_from, video):
        status["made_from"] = str(made_from)
        return status
    status["settings_saved"] = (folder / "config.json").exists()
    if not status["settings_saved"]:
        return status
    saved = saved_config(folder)
    for stage in stages:
        if not hasattr(wanted, stage):
            continue
        for f in fields(getattr(wanted, stage)):
            old, new = getattr(getattr(saved, stage), f.name), getattr(getattr(wanted, stage), f.name)
            if old != new:
                status["differences"].append({"stage": stage, "name": f.name, "saved": old, "wanted": new})
    changed = {d["stage"] for d in status["differences"]}
    status["start"] = next((s for s in STAGES if s not in stages or s in changed), None)
    return status


# ---------------------------------------------------------------------------------------------------------------
# Videos in a folder

def list_folder(folder: Path, output_dir: Path, wanted: PipelineConfig) -> dict:
    """Subfolders and videos of `folder` (absolute), each video with its size, length and results state."""
    entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    subfolders = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
    videos = []
    for path in entries:
        if not is_video(path):
            continue
        stat = path.stat()
        try:
            info = video_info(path)
        except OSError:
            info = None  # OpenCV can't read it
        videos.append({
            "file": path.name, "path": str(path), "size": stat.st_size, "modified": stat.st_mtime, "info": info,
            "results": results_status(path, output_dir, wanted),
        })
    return {
        "folder": str(folder),
        "parent": str(folder.parent) if folder.parent != folder else None,
        "folders": [{"name": p.name, "path": str(p)} for p in subfolders],
        "videos": videos,
    }


def video_info(path: Path) -> dict:
    """fps, width, height and frame count, cached until the file changes."""
    stat = path.stat()
    return _video_info(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4096)
def _video_info(path: str, mtime_ns: int, size: int) -> dict:
    info = asdict(get_video_info(path))
    del info["path"]
    return info


def thumbnail(path: Path) -> bytes:
    """The first frame as a small JPEG, cached until the file changes."""
    stat = path.stat()
    return _thumbnail(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=1024)
def _thumbnail(path: str, mtime_ns: int, size: int) -> bytes:
    frames = read_frames(path)
    try:
        frame = next(frames, None)
    finally:
        frames.close()
    if frame is None:
        raise OSError(f"No frame in {path}")
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (THUMBNAIL_WIDTH, round(height * THUMBNAIL_WIDTH / width)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise OSError(f"Cannot make a thumbnail of {path}")
    return jpeg.tobytes()
