"""Draw person detections on video frames."""

import cv2
import numpy as np

from swim.stages.detection.types import FrameDetections

BOX_COLOR = (0, 0, 255)  # BGR red: all detection boxes
REJECTED_COLOR = (170, 170, 170)  # BGR gray: detections removed by filtering
INFO_COLOR = (255, 255, 255)  # BGR white: frame info text

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame: np.ndarray, frame_dets: FrameDetections) -> np.ndarray:
    """Draw every detection's box with label and score, rejected ones in gray with the reason, and a frame info line.

    Draws on `frame` in place and returns it.
    """
    height = frame.shape[0]
    # Rejected first, so kept boxes are drawn on top; their label goes below the box, so it doesn't
    # collide with the label of a kept box it duplicates
    for rej in frame_dets.rejected:
        b = rej.detection.bbox
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), REJECTED_COLOR, 2)
        cv2.putText(
            frame, f"{rej.reason} {rej.detection.score:.2f}", (int(b.x1), min(int(b.y2) + 28, height - 10)),
            FONT, 0.8, REJECTED_COLOR, 2,
        )

    for det in frame_dets.detections:
        b = det.bbox
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), BOX_COLOR, 3)
        cv2.putText(
            frame, f"{det.label} {det.score:.2f}", (int(b.x1), max(int(b.y1) - 10, 20)), FONT, 0.9, BOX_COLOR, 2
        )

    info = f"frame {frame_dets.frame_idx}  persons: {len(frame_dets.detections)}"
    if frame_dets.rejected:
        info += f"  rejected: {len(frame_dets.rejected)}"
    cv2.putText(frame, info, (20, 40), FONT, 1.0, INFO_COLOR, 2)
    return frame
