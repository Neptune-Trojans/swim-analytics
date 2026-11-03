"""Draw tracks on video frames: every track in its own color with its number, boxes the tracker didn't use in gray."""

import cv2
import numpy as np

from swim.stages.detection.types import FrameDetections
from swim.visualization.detections import FONT, INFO_COLOR, REJECTED_COLOR

# Track colors (RGB), chosen to stand out on blue pool water: track N gets TRACK_COLORS[(N - 1) % len(TRACK_COLORS)].
# Same list as TRACK_COLORS in swim/app/static/layers.js
TRACK_COLORS = [
    (255, 214, 10),  # yellow
    (255, 61, 242),  # magenta
    (255, 140, 0),  # orange
    (124, 255, 0),  # lime
    (255, 255, 255),  # white
    (184, 77, 255),  # violet
    (0, 230, 118),  # mint
    (30, 30, 30),  # black
]


def track_color(track_id: int) -> tuple[int, int, int]:
    """BGR color of a track (OpenCV order)."""
    r, g, b = TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]
    return b, g, r


def draw_tracks(frame: np.ndarray, frame_dets: FrameDetections) -> np.ndarray:
    """Draw every tracked box in its track's color with "#id score", untracked boxes thin and gray, and a frame info
    line. Draws on `frame` in place and returns it."""
    height = frame.shape[0]
    # Untracked first, so tracks are drawn on top; their label goes below the box
    for rej in frame_dets.rejected:
        b = rej.detection.bbox
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), REJECTED_COLOR, 1)
        cv2.putText(
            frame, f"{rej.reason} {rej.detection.score:.2f}", (int(b.x1), min(int(b.y2) + 24, height - 10)),
            FONT, 0.6, REJECTED_COLOR, 1,
        )

    for det in frame_dets.detections:
        b = det.bbox
        color = track_color(det.track_id)
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 3)
        cv2.putText(frame, f"#{det.track_id} {det.score:.2f}", (int(b.x1), max(int(b.y1) - 10, 20)), FONT, 0.9, color, 2)

    tracks = ", ".join(f"#{d.track_id}" for d in frame_dets.detections) or "none"
    cv2.putText(frame, f"frame {frame_dets.frame_idx}  tracks: {tracks}", (20, 40), FONT, 1.0, INFO_COLOR, 2)
    return frame
