"""Pool labels of one frame, for training the pool segmentation model (docs/labeling.md).

A frame's labels are shapes, not pixels: the pool's water as polygons, each lane rope as a straight line with a width
at each end. One JSON file per frame, in pixel coordinates of the full-size frame. Training masks (one class number
per pixel) are drawn from the shapes with `rasterize()`.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Class number of each pixel in a training mask
CLASSES = {0: "other", 1: "water", 2: "lane rope"}
OTHER, WATER, ROPE = 0, 1, 2

# sam3: made by the model, not looked at yet; fixed: corrected by hand; accepted: the model's labels were right
STATUSES = ("sam3", "fixed", "accepted")
REVIEWED = ("fixed", "accepted")

Point = tuple[float, float]


@dataclass
class Rope:
    """A lane rope: a straight line between two points, with its width (pixels) at each end."""

    points: list[Point]
    widths: list[float]


@dataclass
class FrameLabels:
    video: str  # the video the frame comes from, as given when the frame was picked
    frame_idx: int
    image: str  # file name of the frame in the dataset's images/ folder
    width: int
    height: int
    status: str = "sam3"
    water: list[list[Point]] = field(default_factory=list)  # polygons: the pool area (swimmers included)
    ropes: list[Rope] = field(default_factory=list)
    source: str = ""  # what made the first labels, e.g. "sam3-litetext-s0"

    @property
    def name(self) -> str:
        return Path(self.image).stem

    def validate(self) -> None:
        """Raise ValueError if the labels are not usable."""
        if self.status not in STATUSES:
            raise ValueError(f"Unknown status '{self.status}' (one of: {', '.join(STATUSES)})")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        for k, polygon in enumerate(self.water):
            if len(polygon) < 3:
                raise ValueError(f"Water polygon {k + 1} has {len(polygon)} corners, needs at least 3")
            for point in polygon:
                _check_point(point, f"water polygon {k + 1}")
        for k, rope in enumerate(self.ropes):
            if len(rope.points) != 2 or len(rope.widths) != 2:
                raise ValueError(f"Rope {k + 1} needs 2 points and 2 widths")
            for point in rope.points:
                _check_point(point, f"rope {k + 1}")
            if any(not isinstance(w, (int, float)) or isinstance(w, bool) or w <= 0 for w in rope.widths):
                raise ValueError(f"Rope {k + 1}: widths must be positive numbers")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["water"] = [[[round(x, 1), round(y, 1)] for x, y in polygon] for polygon in self.water]
        data["ropes"] = [{"points": [[round(x, 1), round(y, 1)] for x, y in r.points],
                          "widths": [round(w, 1) for w in r.widths]} for r in self.ropes]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FrameLabels":
        """Labels from their JSON form; raises ValueError for a missing field or a bad shape."""
        try:
            labels = cls(
                video=str(data["video"]), frame_idx=int(data["frame_idx"]), image=str(data["image"]),
                width=int(data["width"]), height=int(data["height"]), status=str(data.get("status", "sam3")),
                water=[[(float(x), float(y)) for x, y in polygon] for polygon in data.get("water", [])],
                ropes=[Rope([(float(x), float(y)) for x, y in r["points"]], [float(w) for w in r["widths"]])
                       for r in data.get("ropes", [])],
                source=str(data.get("source", "")),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Bad labels: {e}") from e
        labels.validate()
        return labels

    def to_json(self, path: str | Path) -> None:
        self.validate()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> "FrameLabels":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _check_point(point, where: str) -> None:
    if len(point) != 2 or not all(np.isfinite(v) for v in point):
        raise ValueError(f"Bad point {point} in {where}")


def rope_corners(rope: Rope) -> np.ndarray:
    """The rope's band: 4 corners, its width at each end (a near end looks wider than a far one)."""
    p1, p2 = np.array(rope.points[0], float), np.array(rope.points[1], float)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    across = np.array([-direction[1], direction[0]]) / length if length > 0 else np.array([0.0, 1.0])
    w1, w2 = rope.widths[0] / 2, rope.widths[1] / 2
    return np.array([p1 + across * w1, p2 + across * w2, p2 - across * w2, p1 - across * w1])


def rasterize(labels: FrameLabels) -> np.ndarray:
    """Training mask: one class number per pixel (0 other, 1 water, 2 lane rope); ropes are drawn over the water."""
    mask = np.zeros((labels.height, labels.width), np.uint8)
    # Coordinates at 1/16 pixel (shift=4), so shapes are not rounded to whole pixels
    for polygon in labels.water:
        cv2.fillPoly(mask, [np.round(np.array(polygon) * 16).astype(np.int32)], WATER, lineType=cv2.LINE_8, shift=4)
    for rope in labels.ropes:
        cv2.fillPoly(mask, [np.round(rope_corners(rope) * 16).astype(np.int32)], ROPE, lineType=cv2.LINE_8, shift=4)
    return mask
