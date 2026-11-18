"""Filter: remove detections of people outside the pool (on the deck, in the stands), using the pool stage's water.

A box counts as in the pool when its bottom-center, where the person floats or stands, is in the water. Not the share
of the box in the water: a swimmer at the far wall has the head and shoulders above the water line, so most of the box
is outside the water while the bottom is in it.
"""

import cv2
import numpy as np

from swim.stages.detection.types import Detection
from swim.stages.pool.types import Polygon


def filter_outside_pool(
    detections: list[Detection], water: list[Polygon], margin: float
) -> tuple[list[Detection], list[Detection]]:
    """Split into (kept, removed): removed are the detections whose bottom-center is more than `margin` pixels outside
    the water. A frame without water (none found) keeps every detection."""
    if not water:
        return detections, []
    outlines = [np.array(polygon, np.int32) for polygon in water]

    def in_pool(d: Detection) -> bool:
        bottom_center = ((d.bbox.x1 + d.bbox.x2) / 2, d.bbox.y2)
        # pointPolygonTest with measureDist: the distance to the outline, positive inside, negative outside
        return any(cv2.pointPolygonTest(outline, bottom_center, True) >= -margin for outline in outlines)

    inside = [in_pool(d) for d in detections]
    kept = [d for d, keep in zip(detections, inside) if keep]
    removed = [d for d, keep in zip(detections, inside) if not keep]
    return kept, removed
