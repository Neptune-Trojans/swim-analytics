"""Draw the pool's water on video frames (stage 2, pool)."""

import cv2
import numpy as np

from swim.stages.pool.types import FramePool

# Same look as the pool layer in swim/app/static/layers.js (there in RGB)
WATER_COLOR = (255, 140, 0)  # BGR blue: water tint
OUTLINE_COLOR = (255, 230, 120)  # BGR light blue: the water's outline
WATER_ALPHA = 0.3


def draw_pool(frame: np.ndarray, frame_pool: FramePool) -> np.ndarray:
    """Tint the water blue and draw its outline. Draws on `frame` in place and returns it."""
    if not frame_pool.water:
        return frame
    polygons = [np.array(polygon, np.int32) for polygon in frame_pool.water]
    tinted = frame.copy()
    cv2.fillPoly(tinted, polygons, WATER_COLOR)
    cv2.addWeighted(tinted, WATER_ALPHA, frame, 1 - WATER_ALPHA, 0, dst=frame)
    cv2.polylines(frame, polygons, True, OUTLINE_COLOR, 3, cv2.LINE_AA)
    return frame
