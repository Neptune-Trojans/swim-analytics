"""Stage 3: remove problematic detections. Removed ones are kept in `rejected` with the reason, for debugging."""

from collections import Counter
from dataclasses import replace

from swim.pipeline.config import FilteringConfig
from swim.stages.detection.types import Detection, FrameDetections, RejectedDetection, VideoDetections
from swim.stages.filtering.duplicates import filter_duplicates
from swim.stages.filtering.min_score import filter_min_score
from swim.stages.filtering.outside_pool import filter_outside_pool
from swim.stages.pool.types import Polygon, VideoPool


def filter_frame(
    detections: list[Detection], water: list[Polygon], height: int, config: FilteringConfig
) -> tuple[list[Detection], list[RejectedDetection]]:
    """Run the filters in order; each filter only sees the detections the previous ones kept. `water`: the frame's
    water from the pool stage, `height`: the frame height in pixels."""
    rejected = []

    detections, removed = filter_min_score(detections, config.min_score)
    rejected += [RejectedDetection(d, "min_score") for d in removed]

    # Before the duplicates: a box on the deck overlapping a swimmer's box must not remove it as a duplicate
    if config.pool_margin < 1:
        detections, removed = filter_outside_pool(detections, water, config.pool_margin * height)
        rejected += [RejectedDetection(d, "outside_pool") for d in removed]

    detections, removed = filter_duplicates(detections, config.max_iou)
    rejected += [RejectedDetection(d, "duplicate") for d in removed]

    return detections, rejected


def filter_detections(video_dets: VideoDetections, pool: VideoPool, config: FilteringConfig) -> VideoDetections:
    water = {f.frame_idx: f.water for f in pool.frames}
    frames = []
    for f in video_dets.frames:
        kept, rejected = filter_frame(f.detections, water.get(f.frame_idx, []), video_dets.height, config)
        frames.append(FrameDetections(frame_idx=f.frame_idx, timestamp=f.timestamp, detections=kept, rejected=rejected))
    result = replace(video_dets, frames=frames)

    reasons = Counter(r.reason for f in frames for r in f.rejected)
    before = sum(1 for f in video_dets.frames if f.detections)
    after = sum(1 for f in frames if f.detections)
    print(f"  removed {sum(reasons.values())} detections ({', '.join(f'{k}: {v}' for k, v in reasons.items()) or 'none'}), "
          f"frames with a person: {before} -> {after}")
    return result
