"""The pool model trained with `python -m swim.pool_model train` (a U-Net, docs/pool-model.md): the water's outline."""

from pathlib import Path

import cv2
import numpy as np

from swim.stages.pool.types import Polygon

TOLERANCE = 3  # pixels: corners of the outline stay within this distance of the model's water edge
MIN_PIECE = 0.001  # pieces of water smaller than this share of the frame are dropped (specks)


class UNetSegmenter:
    def __init__(self, model_path: Path, device: str | None = None):
        from swim.pool_model.predict import PoolModel  # imports torch

        self.model = PoolModel.load(model_path, device)
        self.name = f"pool-unet {model_path.parent.name}"

    def water(self, frame: np.ndarray) -> list[Polygon]:
        return water_polygons(self.model.predict(frame) == 1)


def water_polygons(water: np.ndarray, tolerance: float = TOLERANCE, min_piece: float = MIN_PIECE) -> list[Polygon]:
    """The outline of each piece of water in a mask (True = water), simplified to fewer corners. Holes are not kept:
    whatever the water surrounds (a swimmer the model missed) counts as pool."""
    contours, _ = cv2.findContours(water.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_piece * water.size:
            continue
        corners = cv2.approxPolyDP(contour, tolerance, True)[:, 0]
        if len(corners) >= 3:
            polygons.append([(int(x), int(y)) for x, y in corners])
    return polygons
