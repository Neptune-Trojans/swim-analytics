"""Quick check: run RF-DETR person detection on a video and save it with the boxes drawn.

Saves two files: the video with boxes (<name>_rfdetr.mp4) and the detections (<name>_rfdetr.json).
The highest-score person in each frame is drawn in green, other persons in red.

Install:
    pip install rfdetr

Usage:
    python scripts/run_rfdetr.py --input data/input/IMG_4854.MOV
    python scripts/run_rfdetr.py --input data/input/IMG_4854.MOV --output data/output/test.mp4 --model large --threshold 0.3
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Allow `import swim` when running this script directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swim.detections import FrameDetections, VideoDetections  # noqa: E402
from swim.detectors.rfdetr_detector import RFDETRDetector  # noqa: E402

BEST_COLOR = (0, 200, 0)  # BGR green
OTHER_COLOR = (0, 0, 255)  # BGR red


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RF-DETR person detection on a video.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="data/output", help="Output video file or folder (default: data/output)")
    parser.add_argument("--model", default="medium", choices=RFDETRDetector.MODELS.keys(), help="Model size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Hide detections below this confidence")
    args = parser.parse_args()

    # A folder (or a path without extension) gets the default file name <input>_rfdetr.mp4
    output = Path(args.output)
    if output.is_dir() or not output.suffix:
        output = output / f"{Path(args.input).stem}_rfdetr.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output = output.with_suffix(".json")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot write video: {output}")

    detector = RFDETRDetector(size=args.model, threshold=args.threshold)
    video_dets = VideoDetections(video_path=args.input, fps=fps, width=width, height=height, model=detector.name)

    print(f"{args.input}: {width}x{height} @ {fps:.1f} fps, {total} frames")
    start = time.time()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_dets = FrameDetections(frame_idx=frame_idx, timestamp=frame_idx / fps, detections=detector.detect(frame))
        video_dets.frames.append(frame_dets)

        best = frame_dets.best()
        for det in frame_dets.detections:
            color = BEST_COLOR if det is best else OTHER_COLOR
            b = det.bbox
            cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 3)
            cv2.putText(frame, f"{det.label} {det.score:.2f}", (int(b.x1), max(int(b.y1) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(frame, f"frame {frame_idx}  persons: {len(frame_dets.detections)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} frames ({frame_idx / (time.time() - start):.1f} fps)")

    cap.release()
    writer.release()
    video_dets.to_json(json_output)

    print(f"Done: {frame_idx} frames in {time.time() - start:.1f}s -> {output}")
    print(f"Detections saved -> {json_output}")
    if frame_idx:
        found = sum(1 for f in video_dets.frames if f.detections)
        multi = sum(1 for f in video_dets.frames if len(f.detections) > 1)
        print(f"Person found in {found}/{frame_idx} frames ({100 * found / frame_idx:.0f}%), "
              f"more than one person in {multi} frames")


if __name__ == "__main__":
    main()
