"""Stage 2: find the pool's water in every frame, with the pool model trained on our own frames (docs/pool-model.md).

Nothing uses it yet. Next: filtering drops boxes outside the water; with lane ropes (later), the lane of each box.
"""

import time
from collections.abc import Callable
from pathlib import Path

from swim.pipeline.config import PoolConfig
from swim.stages.pool.types import FramePool, PoolSegmenter, VideoPool
from swim.utils.video import get_video_info, read_frames

MODELS_DIR = Path("data/models/pool")  # training runs of python -m swim.pool_model train, one folder each


def find_model(name: str, root: Path = MODELS_DIR) -> Path:
    """The model.pt of a training run: `name` is the run's folder name, or "latest" for the newest run."""
    if name == "latest":
        runs = sorted(path.parent for path in root.glob("*/model.pt"))  # folder names are date-times: sorted = in time
        if not runs:
            raise FileNotFoundError(f"No trained pool model in {root}: train one with python -m swim.pool_model train")
        return runs[-1] / "model.pt"
    path = root / name / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"No pool model {path} (pool.model is the name of a run folder in {root}, or latest)")
    return path


def create_segmenter(config: PoolConfig) -> PoolSegmenter:
    # Imported here so nothing loads torch unless the pool stage actually runs
    from swim.stages.pool.unet_model import UNetSegmenter

    return UNetSegmenter(find_model(config.model))


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
