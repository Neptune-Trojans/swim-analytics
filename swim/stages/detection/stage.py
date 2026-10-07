"""Stage 1: detect all persons in every frame of the video."""

import time
from pathlib import Path

from swim.pipeline.config import DetectionConfig
from swim.stages.detection.types import Detector, FrameDetections, VideoDetections
from swim.utils.video import get_video_info, read_frames


def create_detector(config: DetectionConfig) -> Detector:
    if config.model == "rfdetr":
        # Imported here so nothing loads rfdetr / torch unless detection actually runs
        from swim.stages.detection.rfdetr_detector import RFDETRDetector

        return RFDETRDetector(size=config.size, threshold=config.threshold)
    raise ValueError(f"Unknown detection model: '{config.model}'")


def detect(video_path: str | Path, config: DetectionConfig) -> VideoDetections:
    info = get_video_info(video_path)
    detector = create_detector(config)
    result = VideoDetections(
        video_path=str(video_path), fps=info.fps, width=info.width, height=info.height, model=detector.name
    )

    start = time.time()
    for frame_idx, frame in enumerate(read_frames(video_path)):
        result.frames.append(
            FrameDetections(frame_idx=frame_idx, timestamp=frame_idx / info.fps, detections=detector.detect(frame))
        )
        if (frame_idx + 1) % 30 == 0:
            print(f"  {frame_idx + 1}/{info.frame_count} frames ({(frame_idx + 1) / (time.time() - start):.1f} fps)")

    n = len(result.frames)
    found = sum(1 for f in result.frames if f.detections)
    multi = sum(1 for f in result.frames if len(f.detections) > 1)
    print(f"  person found in {found}/{n} frames ({100 * found / max(n, 1):.0f}%), more than one in {multi} frames")
    return result
