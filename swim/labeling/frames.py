"""Pick frames from videos for labeling: evenly spread over each video, a capped number per video."""

from pathlib import Path

import cv2

from swim.labeling.dataset import Dataset, Frame, frame_name
from swim.utils.video import get_video_info, read_frames

JPEG_QUALITY = 95


def pick_indices(frame_count: int, fps: float, every_seconds: float, max_frames: int) -> list[int]:
    """Frame indices one every `every_seconds`, at most `max_frames`, evenly spread (half a step from each end)."""
    if frame_count <= 0:
        return []
    step = max(every_seconds * fps, 1)
    count = max(1, min(max_frames, int(frame_count / step)))
    spacing = frame_count / count
    return sorted({min(int(spacing * (k + 0.5)), frame_count - 1) for k in range(count)})


def add_video_frames(dataset: Dataset, video: str | Path, every_seconds: float, max_frames: int) -> list[str]:
    """Save the picked frames of a video to the dataset (skipping frames already there); returns the new names."""
    info = get_video_info(video)
    known = {f.name for f in dataset.frames()}
    wanted = {i for i in pick_indices(info.frame_count, info.fps, every_seconds, max_frames)
              if frame_name(video, i) not in known}
    if not wanted:
        return []
    dataset.images.mkdir(parents=True, exist_ok=True)
    added = []
    for i, frame in enumerate(read_frames(video)):
        if i in wanted:
            name = frame_name(video, i)
            cv2.imwrite(str(dataset.image_path(name)), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            added.append(Frame(name=name, video=str(video), frame_idx=i, width=frame.shape[1], height=frame.shape[0]))
        if i >= max(wanted):
            break
    dataset.add_frames(added)
    return [f.name for f in added]
