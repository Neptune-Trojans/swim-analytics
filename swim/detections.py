"""Person detection data types and the Detector contract.

These are the stable output format of every detector. Detector implementations live in swim/detectors/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel coordinates of the original frame (x1, y1 = top-left)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def iou(self, other: BBox) -> float:
        """Intersection over union: 0 = no overlap, 1 = identical boxes."""
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame."""

    bbox: BBox
    score: float  # model confidence 0..1
    label: str = "person"
    track_id: int | None = None  # filled later by the tracker; None straight from the detector


@dataclass
class FrameDetections:
    """All detections in one frame. An empty list means nothing was found."""

    frame_idx: int
    timestamp: float  # seconds from video start (frame_idx / fps)
    detections: list[Detection] = field(default_factory=list)

    def best(self) -> Detection | None:
        """Highest-score detection, or None if the frame is empty."""
        return max(self.detections, key=lambda d: d.score, default=None)


@dataclass
class VideoDetections:
    """Detector output for a whole video, saved to JSON so later stages don't re-run detection."""

    video_path: str
    fps: float
    width: int
    height: int
    model: str  # e.g. "rfdetr-medium"
    frames: list[FrameDetections] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> VideoDetections:
        data = json.loads(Path(path).read_text())
        frames = [
            FrameDetections(
                frame_idx=f["frame_idx"],
                timestamp=f["timestamp"],
                detections=[
                    Detection(
                        bbox=BBox(**d["bbox"]),
                        score=d["score"],
                        label=d["label"],
                        track_id=d["track_id"],
                    )
                    for d in f["detections"]
                ],
            )
            for f in data.pop("frames")
        ]
        return cls(**data, frames=frames)


class Detector(Protocol):
    """Any person detector: takes a BGR frame (as OpenCV reads it), returns detections."""

    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...
