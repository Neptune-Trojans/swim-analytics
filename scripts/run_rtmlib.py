"""Quick check: run RTMPose (rtmlib) on a video and save it with the skeleton drawn.

Usage:
    python scripts/run_rtmlib.py --input data/input/IMG_4854.MOV
    python scripts/run_rtmlib.py --input data/input/IMG_4854.MOV --output data/output/test.mp4 --mode performance
"""

import argparse
import sys
import time
from pathlib import Path

from rtmlib import Body, draw_skeleton

# Allow `import swim` when running this script directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swim.video import get_video_info, read_frames, resolve_output_path, video_writer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rtmlib pose detection on a video.")
    parser.add_argument("--input", help="Path to input video")
    parser.add_argument("--output", default="data/output", help="Output video file or folder (default: data/output)")
    parser.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Hide keypoints below this confidence")
    args = parser.parse_args()

    output = resolve_output_path(args.output, args.input, "_rtmlib")
    info = get_video_info(args.input)
    body = Body(mode=args.mode, backend="onnxruntime", device="cpu")

    print(f"{args.input}: {info.width}x{info.height} @ {info.fps:.1f} fps, {info.frame_count} frames")
    start = time.time()
    frame_idx = 0
    with video_writer(output, info.fps, info.width, info.height) as writer:
        for frame in read_frames(args.input):
            keypoints, scores = body(frame)
            frame = draw_skeleton(frame, keypoints, scores, kpt_thr=args.kpt_thr, radius=4, line_width=3)
            writer.write(frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  {frame_idx}/{info.frame_count} frames ({frame_idx / (time.time() - start):.1f} fps)")

    print(f"Done: {frame_idx} frames in {time.time() - start:.1f}s -> {output}")


if __name__ == "__main__":
    main()
