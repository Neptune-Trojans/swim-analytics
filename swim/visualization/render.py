"""Render stage outputs as layers on the video."""

import time
from pathlib import Path

from swim.pipeline.runner import PipelineResult
from swim.stages.detection.types import FrameDetections
from swim.utils.video import get_video_info, read_frames, video_writer
from swim.visualization.detections import draw_detections

# Layer name -> (the PipelineResult field it draws, the function that draws one frame of it). In pipeline order.
LAYERS = {
    "detections": ("detections", draw_detections),  # stage 1: everything the detector found
    "filtered": ("filtered", draw_detections),  # stage 2: kept detections, rejected ones in gray with the reason
}


def available_layers(result: PipelineResult) -> list[str]:
    """Layers whose data is present in `result`, in pipeline order."""
    return [name for name, (field_name, _) in LAYERS.items() if getattr(result, field_name) is not None]


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

    start = time.time()
    with video_writer(output_path, info.fps, info.width, info.height) as writer:
        for frame_idx, frame in enumerate(read_frames(video_path)):
            for name, frames in frames_by_layer.items():
                # Frames missing from the results are drawn as "nothing found"
                frame_data = frames.get(frame_idx) or FrameDetections(frame_idx, frame_idx / info.fps)
                LAYERS[name][1](frame, frame_data)
            writer.write(frame)
    print(f"[video] layers {', '.join(layers) or '(none)'} -> {output_path} ({time.time() - start:.1f}s)")
