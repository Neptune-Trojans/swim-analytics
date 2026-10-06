"""Read frames from a video and write processed frames to an output video."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int


def get_video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")
    try:
        return VideoInfo(
            path=str(path),
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield the video's frames in order, as BGR images."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


@contextmanager
def video_writer(path: str | Path, fps: float, width: int, height: int) -> Iterator[cv2.VideoWriter]:
    """Open an mp4 writer (creating the folder if needed) and release it when done."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    # OpenCV does not raise on a bad path, it silently writes nothing
    if not writer.isOpened():
        raise OSError(f"Cannot write video: {path}")
    try:
        yield writer
    finally:
        writer.release()


def resolve_output_path(output: str | Path, input_path: str | Path, suffix: str) -> Path:
    """Use `output` as the file path, or if it is a folder (or has no extension) put <input-name><suffix>.mp4 in it."""
    output = Path(output)
    if output.is_dir() or not output.suffix:
        output = output / f"{Path(input_path).stem}{suffix}.mp4"
    return output
