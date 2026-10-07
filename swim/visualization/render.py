"""Render stage outputs as layers on the video."""

import time
from pathlib import Path

from swim.pipeline.runner import PipelineResult
from swim.stages.detection.types import FrameDetections
from swim.utils.video import get_video_info, read_frames, video_writer
from swim.visualization.detections import draw_detections

# Layer name -> the PipelineResult field it draws
LAYERS = {
    "detections": "detections",
}


def available_layers(result: PipelineResult) -> list[str]:
    """Layers whose data is present in `result`."""
    return [name for name, field_name in LAYERS.items() if getattr(result, field_name) is not None]


def render_video(video_path: str | Path, output_path: str | Path, result: PipelineResult, layers: list[str]) -> None:
    for name in layers:
        if name not in LAYERS:
            raise ValueError(f"Unknown layer '{name}', choose from {list(LAYERS)}")
        if getattr(result, LAYERS[name]) is None:
            raise ValueError(f"Layer '{name}' has no data: run its stage first")

    info = get_video_info(video_path)
    dets = result.detections
    if dets is not None and (info.width, info.height) != (dets.width, dets.height):
        raise ValueError(
            f"Video is {info.width}x{info.height} but detections were made on {dets.width}x{dets.height} "
            f"({dets.video_path}): wrong results folder?"
        )
    detections_by_frame = {f.frame_idx: f for f in dets.frames} if "detections" in layers else {}

    start = time.time()
    with video_writer(output_path, info.fps, info.width, info.height) as writer:
        for frame_idx, frame in enumerate(read_frames(video_path)):
            if "detections" in layers:
                # Frames missing from the results are drawn as "no detections"
                frame_dets = detections_by_frame.get(frame_idx) or FrameDetections(frame_idx, frame_idx / info.fps)
                draw_detections(frame, frame_dets)
            writer.write(frame)
    print(f"[video] layers {', '.join(layers) or '(none)'} -> {output_path} ({time.time() - start:.1f}s)")
