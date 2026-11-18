"""Stage 2: find the pool's water in every frame, with the pool model trained on our own frames (the pool-segmentation
package, ../pool-segmentation: its docs/model.md).

Filtering drops the boxes outside the water (outside_pool). Later, with lane ropes: the lane of each box.
"""

import time
from collections.abc import Callable
from pathlib import Path

from pool_segmentation import find_model

from swim.pipeline.config import PoolConfig
from swim.stages.pool.types import FramePool, PoolSegmenter, VideoPool
from swim.utils.video import get_video_info, read_frames


def create_segmenter(config: PoolConfig) -> PoolSegmenter:
    # Imported here so nothing loads torch unless the pool stage actually runs
    from swim.stages.pool.unet_model import UNetSegmenter

    return UNetSegmenter(find_model(config.model))  # config.model: a run folder in data/models/pool/, or latest


def segment_pool(
    video_path: str | Path, config: PoolConfig, on_frame: Callable[[int, int], None] | None = None
) -> VideoPool:
    """The water in every frame. `on_frame(frames done, frames in the video)` is called after each frame; it may raise
    to stop."""
    info = get_video_info(video_path)
    segmenter = create_segmenter(config)
    result = VideoPool(video_path=str(video_path), fps=info.fps, width=info.width, height=info.height, model=segmenter.name)

    start = time.time()
    for frame_idx, frame in enumerate(read_frames(video_path)):
        result.frames.append(FramePool(frame_idx=frame_idx, timestamp=frame_idx / info.fps, water=segmenter.water(frame)))
        if on_frame:
            on_frame(frame_idx + 1, info.frame_count)
        if (frame_idx + 1) % 30 == 0:
            print(f"  {frame_idx + 1}/{info.frame_count} frames ({(frame_idx + 1) / (time.time() - start):.1f} fps)")

    n = len(result.frames)
    found = sum(1 for f in result.frames if f.water)
    print(f"  water found in {found}/{n} frames ({100 * found / max(n, 1):.0f}%), model {segmenter.name}")
    return result
