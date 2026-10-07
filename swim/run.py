"""Command line: run the analysis pipeline on a video.

Results go to <output>/<video name>/: config.json, one JSON per stage (1_detections.json, ...) and video.mp4.

Usage:
    python -m swim.run --input data/input/IMG_4884.MOV
    python -m swim.run --input data/input/IMG_4884.MOV --to detection
    python -m swim.run --input data/input/IMG_4884.MOV --from detection --detection-size large --detection-threshold 0.3
    python -m swim.run --input data/input/IMG_4884.MOV --from filtering --filter-min-score 0.6
    python -m swim.run --input data/input/IMG_4884.MOV --render-only --layers detections,filtered
"""

import argparse
from pathlib import Path

from swim.pipeline.config import DetectionConfig, FilteringConfig, PipelineConfig
from swim.pipeline.runner import STAGES, load_results, run_pipeline
from swim.visualization.render import LAYERS, available_layers, render_video


def main() -> None:
    defaults = DetectionConfig()
    filter_defaults = FilteringConfig()
    parser = argparse.ArgumentParser(prog="python -m swim.run", description="Analyze a swimming video.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="data/output", help="Results root folder; results go to <output>/<video name>/")
    parser.add_argument("--from", dest="start", choices=STAGES, help="First stage to run (earlier ones are loaded from saved results)")
    parser.add_argument("--to", dest="stop", choices=STAGES, help="Last stage to run")
    parser.add_argument("--render-only", action="store_true", help="Only draw saved results on the video, run no stage")
    parser.add_argument("--layers", help=f"Comma-separated layers to draw: {', '.join(LAYERS)} "
                                         "(default: the layer of the last stage with results)")
    parser.add_argument("--detection-size", default=defaults.size, help=f"Detection model size (default: {defaults.size})")
    parser.add_argument("--detection-threshold", type=float, default=defaults.threshold,
                        help=f"Drop detections below this confidence (default: {defaults.threshold})")
    parser.add_argument("--filter-min-score", type=float, default=filter_defaults.min_score,
                        help=f"Filtering: remove detections below this confidence, 0 = off "
                             f"(default: {filter_defaults.min_score})")
    parser.add_argument("--filter-max-iou", type=float, default=filter_defaults.max_iou,
                        help=f"Filtering: of two boxes overlapping more than this (IoU), keep the higher score, "
                             f"1 = off (default: {filter_defaults.max_iou})")
    args = parser.parse_args()

    output_dir = Path(args.output) / Path(args.input).stem

    if args.render_only:
        result = load_results(output_dir)
    else:
        config = PipelineConfig(
            detection=DetectionConfig(size=args.detection_size, threshold=args.detection_threshold),
            filtering=FilteringConfig(min_score=args.filter_min_score, max_iou=args.filter_max_iou),
        )
        result = run_pipeline(args.input, output_dir, config, start=args.start, stop=args.stop)

    # By default draw only the last stage's layer: earlier layers show the same boxes before later stages changed them
    layers = args.layers.split(",") if args.layers else available_layers(result)[-1:]
    render_video(args.input, output_dir / "video.mp4", result, layers)


if __name__ == "__main__":
    main()
