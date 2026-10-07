"""Stage 2: remove problematic detections. Removed ones are kept in `rejected` with the reason, for debugging."""

from collections import Counter
from dataclasses import replace

from swim.pipeline.config import FilteringConfig
from swim.stages.detection.types import Detection, FrameDetections, RejectedDetection, VideoDetections
from swim.stages.filtering.duplicates import filter_duplicates
from swim.stages.filtering.min_score import filter_min_score


def filter_frame(
    detections: list[Detection], config: FilteringConfig
) -> tuple[list[Detection], list[RejectedDetection]]:
    """Run the filters in order; each filter only sees the detections the previous ones kept."""
    rejected = []

    detections, removed = filter_min_score(detections, config.min_score)
    rejected += [RejectedDetection(d, "min_score") for d in removed]

    detections, removed = filter_duplicates(detections, config.max_iou)
    rejected += [RejectedDetection(d, "duplicate") for d in removed]

    return detections, rejected


def filter_detections(video_dets: VideoDetections, config: FilteringConfig) -> VideoDetections:
    frames = []
    for f in video_dets.frames:
        kept, rejected = filter_frame(f.detections, config)
        frames.append(FrameDetections(frame_idx=f.frame_idx, timestamp=f.timestamp, detections=kept, rejected=rejected))
    result = replace(video_dets, frames=frames)

    reasons = Counter(r.reason for f in frames for r in f.rejected)
    before = sum(1 for f in video_dets.frames if f.detections)
    after = sum(1 for f in frames if f.detections)
    print(f"  removed {sum(reasons.values())} detections ({', '.join(f'{k}: {v}' for k, v in reasons.items()) or 'none'}), "
          f"frames with a person: {before} -> {after}")
    return result
