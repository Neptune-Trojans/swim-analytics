"""ByteTrack, OC-SORT and SORT from Roboflow's `trackers` library (Apache-2.0). Requires: pip install trackers"""

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker, OCSORTTracker, SORTTracker

from swim.pipeline.config import TrackingConfig
from swim.stages.detection.types import Detection


class RoboflowTracker:
    """One of the library's trackers, behind our Tracker contract (see stage.py)."""

    def __init__(self, config: TrackingConfig, fps: float):
        common = {
            # The library counts the lost-track buffer in frames of a 30 fps video and scales it by frame_rate
            "lost_track_buffer": round(config.lost_seconds * 30),
            "frame_rate": fps,
            "minimum_consecutive_frames": config.min_frames,
            "minimum_iou_threshold": config.min_iou,
        }
        if config.tracker == "bytetrack":
            self._tracker = ByteTrackTracker(
                **common, track_activation_threshold=config.start_score, high_conf_det_threshold=config.high_score
            )
        elif config.tracker == "ocsort":
            self._tracker = OCSORTTracker(**common, high_conf_det_threshold=config.high_score)
        elif config.tracker == "sort":
            self._tracker = SORTTracker(**common, track_activation_threshold=config.start_score)
        else:
            raise ValueError(f"Unknown tracker '{config.tracker}', choose from bytetrack, ocsort, sort")
        self.name = config.tracker

    def update(self, detections: list[Detection]) -> list[int | None]:
        n = len(detections)
        boxes = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections], dtype=np.float32)
        frame_dets = sv.Detections(
            xyxy=boxes.reshape(n, 4),
            confidence=np.array([d.score for d in detections], dtype=np.float32),
            class_id=np.zeros(n, dtype=int),
            # The library returns the same detections reordered: each carries its index, to map the IDs back to ours
            data={"index": np.arange(n)},
        )
        result = self._tracker.update(frame_dets)

        ids: list[int | None] = [None] * n
        for index, track_id in zip(result.data.get("index", []), result.tracker_id):
            if track_id >= 0:  # -1: not part of a (confirmed) track
                ids[int(index)] = int(track_id)
        return ids
