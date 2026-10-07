"""Draw person detections on video frames."""

import cv2
import numpy as np

from swim.stages.detection.types import FrameDetections

BOX_COLOR = (0, 0, 255)  # BGR red: all detection boxes
INFO_COLOR = (255, 255, 255)  # BGR white: frame info text

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame: np.ndarray, frame_dets: FrameDetections) -> np.ndarray:
    """Draw every detection's box with label and score, and a frame info line.

    Draws on `frame` in place and returns it.
    """
    for det in frame_dets.detections:
        b = det.bbox
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), BOX_COLOR, 3)
        cv2.putText(
            frame, f"{det.label} {det.score:.2f}", (int(b.x1), max(int(b.y1) - 10, 20)), FONT, 0.9, BOX_COLOR, 2
        )

    info = f"frame {frame_dets.frame_idx}  persons: {len(frame_dets.detections)}"
    cv2.putText(frame, info, (20, 40), FONT, 1.0, INFO_COLOR, 2)
    return frame
