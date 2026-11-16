"""The labeling steps run by `python -m swim.labeling` (docs/labeling.md): pick frames, label them with the model,
write training masks."""

import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from swim.labeling.dataset import Dataset
from swim.labeling.frames import add_video_frames
from swim.labeling.labels import REVIEWED, FrameLabels, Point, rasterize


class Labeler(Protocol):
    name: str

    def label(self, frame: np.ndarray) -> tuple[list[list[Point]], list[list[Point]]]: ...  # water, rope polygons


def create_labeler() -> Labeler:
    # Imported here: loading the model pulls in torch and transformers
    from swim.labeling.sam3_labeler import Sam3Labeler

    return Sam3Labeler()


def pick(dataset: Dataset, videos: list[Path], every_seconds: float, max_per_video: int) -> list[str]:
    added = []
    for video in videos:
        names = add_video_frames(dataset, video, every_seconds, max_per_video)
        print(f"  {Path(video).name}: {len(names)} new frames")
        added += names
    print(f"{len(added)} frames added, {len(dataset.frames())} in {dataset.root}")
    return added


def auto_label(dataset: Dataset, redo: bool = False, labeler: Labeler | None = None) -> int:
    """Label every frame the model has not labeled yet (all with `redo`). Reviewed labels are never replaced."""
    todo = [f for f in dataset.frames() if redo or dataset.sam3(f.name) is None]
    if not todo:
        print("Every frame already has the model's labels (use --redo to label them again)")
        return 0
    labeler = labeler or create_labeler()
    start = time.time()
    for k, frame in enumerate(todo, 1):
        image = cv2.imread(str(dataset.image_path(frame.name)))
        if image is None:
            raise OSError(f"Cannot read {dataset.image_path(frame.name)}")
        water, ropes = labeler.label(image)
        dataset.save_sam3(FrameLabels(video=frame.video, frame_idx=frame.frame_idx, image=frame.image,
                                      width=frame.width, height=frame.height, status="sam3", water=water, ropes=ropes,
                                      source=labeler.name))
        left = (len(todo) - k) * (time.time() - start) / k
        print(f"  {k}/{len(todo)} {frame.name}: {len(water)} water polygon(s), {len(ropes)} rope(s)"
              f"  (about {left / 60:.0f} min left)", flush=True)
    return len(todo)


def write_masks(dataset: Dataset, include_unreviewed: bool = False) -> int:
    """Training masks (PNG, one class number per pixel) for the reviewed frames (all frames with labels if
    `include_unreviewed`). Masks of frames no longer included are removed, so masks/ is exactly the training set."""
    dataset.masks.mkdir(parents=True, exist_ok=True)
    keep = set()
    for frame in dataset.frames():
        labels = dataset.labels(frame.name)
        if labels is None or (labels.status not in REVIEWED and not include_unreviewed):
            continue
        cv2.imwrite(str(dataset.masks / f"{frame.name}.png"), rasterize(labels))
        keep.add(f"{frame.name}.png")
    for old in dataset.masks.glob("*.png"):
        if old.name not in keep:
            old.unlink()
    print(f"{len(keep)} masks in {dataset.masks}")
    return len(keep)
