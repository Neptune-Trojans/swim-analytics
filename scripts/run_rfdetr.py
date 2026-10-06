"""Quick check: run RF-DETR person detection on a video and save it with the boxes drawn.

Saves two files: the video with boxes (<name>_rfdetr.mp4) and the detections (<name>_rfdetr.json).
The highest-score person in each frame is drawn in green, other persons in red.
To re-draw from the saved JSON without running the model again, use scripts/visualize_detections.py.

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

# Allow `import swim` when running this script directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swim.detections import FrameDetections, VideoDetections  # noqa: E402
from swim.detectors.rfdetr_detector import RFDETRDetector  # noqa: E402
from swim.video import get_video_info, read_frames, resolve_output_path, video_writer  # noqa: E402
from swim.visualization.detections import draw_detections  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RF-DETR person detection on a video.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="data/output", help="Output video file or folder (default: data/output)")
    parser.add_argument("--model", default="medium", choices=RFDETRDetector.MODELS.keys(), help="Model size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Hide detections below this confidence")
    args = parser.parse_args()

    output = resolve_output_path(args.output, args.input, "_rfdetr")
    json_output = output.with_suffix(".json")
    info = get_video_info(args.input)

    detector = RFDETRDetector(size=args.model, threshold=args.threshold)
    video_dets = VideoDetections(
        video_path=args.input, fps=info.fps, width=info.width, height=info.height, model=detector.name
    )

    print(f"{args.input}: {info.width}x{info.height} @ {info.fps:.1f} fps, {info.frame_count} frames")
    start = time.time()
    with video_writer(output, info.fps, info.width, info.height) as writer:
        for frame_idx, frame in enumerate(read_frames(args.input)):
            frame_dets = FrameDetections(
                frame_idx=frame_idx, timestamp=frame_idx / info.fps, detections=detector.detect(frame)
            )
            video_dets.frames.append(frame_dets)
            writer.write(draw_detections(frame, frame_dets))

            if (frame_idx + 1) % 30 == 0:
                print(f"  {frame_idx + 1}/{info.frame_count} frames ({(frame_idx + 1) / (time.time() - start):.1f} fps)")

    video_dets.to_json(json_output)

    n = len(video_dets.frames)
    print(f"Done: {n} frames in {time.time() - start:.1f}s -> {output}")
    print(f"Detections saved -> {json_output}")
    if n:
        found = sum(1 for f in video_dets.frames if f.detections)
        multi = sum(1 for f in video_dets.frames if len(f.detections) > 1)
        print(f"Person found in {found}/{n} frames ({100 * found / n:.0f}%), more than one person in {multi} frames")


if __name__ == "__main__":
    main()
