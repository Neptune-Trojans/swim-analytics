"""Draw saved person detections (JSON from run_rfdetr.py) on the video, without running the model again.

The highest-score person in each frame is drawn in green, other persons in red.

Usage:
    python scripts/visualize_detections.py --input data/input/IMG_4886.MOV
    python scripts/visualize_detections.py --input data/input/IMG_4886.MOV --detections data/output/IMG_4886_rfdetr.json --output data/output/test.mp4
"""

import argparse
import sys
import time
from pathlib import Path

# Allow `import swim` when running this script directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swim.detections import FrameDetections, VideoDetections  # noqa: E402
from swim.video import get_video_info, read_frames, resolve_output_path, video_writer  # noqa: E402
from swim.visualization.detections import draw_detections  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw saved detections on a video.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--detections", help="Detections JSON (default: data/output/<name>_rfdetr.json)")
    parser.add_argument("--output", default="data/output", help="Output video file or folder (default: data/output)")
    args = parser.parse_args()

    detections_path = args.detections or f"data/output/{Path(args.input).stem}_rfdetr.json"
    if not Path(detections_path).exists():
        raise SystemExit(f"Detections file not found: {detections_path} (run scripts/run_rfdetr.py first)")
    video_dets = VideoDetections.from_json(detections_path)

    info = get_video_info(args.input)
    if (info.width, info.height) != (video_dets.width, video_dets.height):
        raise SystemExit(
            f"Video is {info.width}x{info.height} but detections were made on "
            f"{video_dets.width}x{video_dets.height} ({video_dets.video_path}): wrong detections file?"
        )

    output = resolve_output_path(args.output, args.input, "_detections")
    frames_by_idx = {f.frame_idx: f for f in video_dets.frames}

    print(f"{args.input} + {detections_path} ({video_dets.model}, {len(video_dets.frames)} frames)")
    start = time.time()
    with video_writer(output, info.fps, info.width, info.height) as writer:
        for frame_idx, frame in enumerate(read_frames(args.input)):
            # Frames missing from the JSON are drawn as "no detections"
            frame_dets = frames_by_idx.get(frame_idx) or FrameDetections(frame_idx, frame_idx / info.fps)
            writer.write(draw_detections(frame, frame_dets))

    print(f"Done in {time.time() - start:.1f}s -> {output}")


if __name__ == "__main__":
    main()
