"""Quick check: run RTMPose (rtmlib) on a video and save it with the skeleton drawn.

Usage:
    python scripts/run_rtmlib.py --input data/input/IMG_4854.MOV
    python scripts/run_rtmlib.py --input data/input/IMG_4854.MOV --output data/output/test.mp4 --mode performance
"""

import argparse
import time
from pathlib import Path

import cv2
from rtmlib import Body, draw_skeleton


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rtmlib pose detection on a video.")
    parser.add_argument("--input", help="Path to input video")
    parser.add_argument("--output", default="data/output", help="Output video file or folder (default: data/output)")
    parser.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Hide keypoints below this confidence")
    args = parser.parse_args()

    # A folder (or a path without extension) gets the default file name <input>_rtmlib.mp4
    output = Path(args.output)
    if output.is_dir() or not output.suffix:
        output = output / f"{Path(args.input).stem}_rtmlib.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output = str(output)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot write video: {output}")
    body = Body(mode=args.mode, backend="onnxruntime", device="cpu")

    print(f"{args.input}: {width}x{height} @ {fps:.1f} fps, {total} frames")
    start = time.time()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        keypoints, scores = body(frame)
        frame = draw_skeleton(frame, keypoints, scores, kpt_thr=args.kpt_thr, radius=4, line_width=3)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} frames ({frame_idx / (time.time() - start):.1f} fps)")

    cap.release()
    writer.release()
    print(f"Done: {frame_idx} frames in {time.time() - start:.1f}s -> {output}")


if __name__ == "__main__":
    main()
