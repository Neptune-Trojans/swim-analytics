"""Render stage outputs as layers on the video."""

import time
from pathlib import Path

from swim.pipeline.runner import PipelineResult
from swim.stages.detection.types import FrameDetections
from swim.stages.pool.types import FramePool
from swim.utils.video import get_video_info, read_frames, video_writer
from swim.visualization.detections import draw_detections
from swim.visualization.pool import draw_pool
from swim.visualization.tracks import draw_tracks

# Layer name -> (the PipelineResult field it draws, the function that draws one frame of it, the frame data class: a
# frame missing from the results is drawn as an empty one). In drawing order: the pool's water underneath, then the
# boxes in pipeline order.
LAYERS = {
    "pool": ("pool", draw_pool, FramePool),  # stage 2: the pool's water
    "detections": ("detections", draw_detections, FrameDetections),  # stage 1: everything the detector found
    "filtered": ("filtered", draw_detections, FrameDetections),  # stage 3: kept boxes, rejected ones in gray with the reason
    "tracks": ("tracks", draw_tracks, FrameDetections),  # stage 4: each track in its color with its number, untracked in gray
}
BOX_LAYERS = ("detections", "filtered", "tracks")


def available_layers(result: PipelineResult) -> list[str]:
    """Layers whose data is present in `result`, in drawing order."""
    return [name for name, (field_name, _, _) in LAYERS.items() if getattr(result, field_name) is not None]


def default_layers(result: PipelineResult) -> list[str]:
    """The pool's water (if found) and the boxes of the last stage that has them: earlier box layers show the same
    boxes before later stages changed them."""
    available = available_layers(result)
    boxes = [name for name in available if name in BOX_LAYERS][-1:]
    return [name for name in available if name == "pool"] + boxes


def render_video(video_path: str | Path, output_path: str | Path, result: PipelineResult, layers: list[str]) -> None:
    info = get_video_info(video_path)
    frames_by_layer = {}
    for name in layers:
        if name not in LAYERS:
            raise ValueError(f"Unknown layer '{name}', choose from {list(LAYERS)}")
        data = getattr(result, LAYERS[name][0])
        if data is None:
            raise ValueError(f"Layer '{name}' has no data: run its stage first")
        if (info.width, info.height) != (data.width, data.height):
            raise ValueError(
                f"Video is {info.width}x{info.height} but layer '{name}' was made on {data.width}x{data.height} "
                f"({data.video_path}): wrong results folder?"
            )
        frames_by_layer[name] = {f.frame_idx: f for f in data.frames}

    order = [name for name in LAYERS if name in frames_by_layer]  # drawing order, whatever order they were given in
    start = time.time()
    with video_writer(output_path, info.fps, info.width, info.height) as writer:
        for frame_idx, frame in enumerate(read_frames(video_path)):
            for name in order:
                _, draw, empty = LAYERS[name]
                # Frames missing from the results are drawn as "nothing found"
                draw(frame, frames_by_layer[name].get(frame_idx) or empty(frame_idx, frame_idx / info.fps))
            writer.write(frame)
    print(f"[video] layers {', '.join(layers) or '(none)'} -> {output_path} ({time.time() - start:.1f}s)")
