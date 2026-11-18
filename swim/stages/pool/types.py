"""Pool data types and the PoolSegmenter contract.

The stable output format of stage 2 (pool): the pool's water in every frame, as polygons in pixel coordinates of the
original frame (like the labels in docs/labeling.md). Segmenter implementations live next to this file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

Polygon = list[tuple[int, int]]


class PoolSegmenter(Protocol):
    """Any model that finds the pool's water in a frame."""

    name: str

    def water(self, frame: np.ndarray) -> list[Polygon]:
        """The water's outline(s) in a BGR frame: one polygon per separate piece of water, none if there is no water."""
        ...


@dataclass
class FramePool:
    """The pool's water in one frame. An empty list means no water was found."""

    frame_idx: int
    timestamp: float  # seconds from video start (frame_idx / fps)
    water: list[Polygon] = field(default_factory=list)


@dataclass
class VideoPool:
    """Pool output for a whole video, saved to JSON (compact: thousands of frames with a polygon each)."""

    video_path: str
    fps: float
    width: int
    height: int
    model: str  # e.g. "pool-unet 20261116-224516" (the training run)
    frames: list[FramePool] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        data = asdict(self)
        for f in data["frames"]:
            f["timestamp"] = round(f["timestamp"], 4)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, separators=(",", ":")))

    @classmethod
    def from_json(cls, path: str | Path) -> VideoPool:
        data = json.loads(Path(path).read_text())
        frames = [
            FramePool(f["frame_idx"], f["timestamp"], [[(x, y) for x, y in polygon] for polygon in f["water"]])
            for f in data.pop("frames")
        ]
        return cls(**data, frames=frames)
