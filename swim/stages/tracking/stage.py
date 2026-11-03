"""Stage 3: give every detection a track_id, so the same person keeps the same number from frame to frame.

Boxes the tracker didn't use (a weak box that continues no track, boxes of a track not yet confirmed) move to
`rejected` with the reason "untracked". Track IDs are numbered 1, 2, ... in order of first appearance.
Choosing which track is the swimmer and filling its gaps is the next stage (swimmer).
"""

from collections import Counter
from dataclasses import replace
from typing import Protocol

from swim.pipeline.config import TrackingConfig
from swim.stages.detection.types import Detection, FrameDetections, RejectedDetection, VideoDetections


class Tracker(Protocol):
    """Any tracker: called once per frame, in frame order, with that frame's detections."""

    name: str

    def update(self, detections: list[Detection]) -> list[int | None]:
        """The track ID of each detection (same order); None = not part of a track."""
        ...


def create_tracker(config: TrackingConfig, fps: float) -> Tracker:
    # Imported here so nothing loads the tracker library unless tracking actually runs
    from swim.stages.tracking.roboflow_trackers import RoboflowTracker

    return RoboflowTracker(config, fps)


def track(video_dets: VideoDetections, config: TrackingConfig) -> VideoDetections:
    tracker = create_tracker(config, video_dets.fps)
    ids = [tracker.update(f.detections) for f in video_dets.frames]  # tracker's ID per detection, None = untracked
    give_ids_to_unconfirmed_boxes(video_dets.frames, ids, frames_back=config.min_frames, min_iou=config.min_iou)

    our_ids: dict[int, int] = {}  # tracker's ID -> ours (1, 2, ... in order of first appearance)
    frames = []
    for f, frame_ids in zip(video_dets.frames, ids):
        tracked, untracked = [], []
        for detection, track_id in zip(f.detections, frame_ids):
            if track_id is None:
                untracked.append(RejectedDetection(detection, "untracked"))
            else:
                our_ids.setdefault(track_id, len(our_ids) + 1)
                tracked.append(replace(detection, track_id=our_ids[track_id]))
        frames.append(FrameDetections(frame_idx=f.frame_idx, timestamp=f.timestamp, detections=tracked, rejected=untracked))
    result = replace(video_dets, frames=frames)

    frames_per_track = Counter(d.track_id for f in frames for d in f.detections)  # a track has one box per frame
    n = max(len(frames), 1)
    longest = ", ".join(f"#{tid} {count} frames ({100 * count / n:.0f}%)" for tid, count in frames_per_track.most_common(3))
    print(f"  {tracker.name}: {len(frames_per_track)} tracks (longest: {longest or 'none'}), "
          f"{sum(len(f.rejected) for f in frames)} boxes untracked")
    return result


def give_ids_to_unconfirmed_boxes(
    frames: list[FrameDetections], ids: list[list[int | None]], frames_back: int, min_iou: float
) -> None:
    """Give a track's ID to the boxes the tracker matched to it but reported as untracked, while not yet confirmed.

    The trackers report a track only once confirmed: the box that starts a track is always untracked, and so are the
    following ones until `min_frames` matches; OC-SORT does the same when a track comes back after missed frames.
    With the whole video we can fix that afterwards: from each box where a track starts or comes back, walk back up
    to `frames_back` frames, each time taking the untracked box that overlaps the previous one most
    (IoU >= `min_iou`). Changes `ids` in place.
    """
    starts: list[tuple[int, int, int]] = []  # (track ID, frame position, detection position) where a track (re)starts
    previous: set[int] = set()
    for i, frame_ids in enumerate(ids):
        starts += [(tid, i, j) for j, tid in enumerate(frame_ids) if tid is not None and tid not in previous]
        previous = {tid for tid in frame_ids if tid is not None}

    for track_id, i, j in starts:
        box = frames[i].detections[j].bbox
        for k in range(i - 1, max(i - 1 - frames_back, -1), -1):
            if track_id in ids[k]:
                break
            overlaps = [(box.iou(d.bbox), m) for m, d in enumerate(frames[k].detections) if ids[k][m] is None]
            best_iou, best = max(overlaps, default=(0.0, None))
            if best is None or best_iou < min_iou:
                break
            ids[k][best] = track_id
            box = frames[k].detections[best].bbox
