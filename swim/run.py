"""Command line: run the analysis pipeline on a video.

Results go to <output>/<video name>/: config.json, one JSON per stage (1_detections.json, ...) and video.mp4.

Usage:
    python -m swim.run --input data/input/IMG_4884.MOV
    python -m swim.run --input data/input/IMG_4884.MOV --to detection
    python -m swim.run --input data/input/IMG_4884.MOV --from detection --detection-size large --detection-threshold 0.3
    python -m swim.run --input data/input/IMG_4884.MOV --from filtering --filter-min-score 0.6
    python -m swim.run --input data/input/IMG_4884.MOV --from tracking --tracker ocsort
    python -m swim.run --input data/input/IMG_4884.MOV --render-only --layers detections,tracks
"""

import argparse
from dataclasses import fields
from pathlib import Path

from swim.pipeline.config import PipelineConfig
from swim.pipeline.runner import STAGES, load_results, run_pipeline
from swim.visualization.render import LAYERS, available_layers, render_video

# Command-line option -> (stage, setting). Explanation, type, default and allowed values come from
# swim/pipeline/config.py, as in the viewer's settings panel.
SETTING_OPTIONS = {
    "--detection-size": ("detection", "size"),
    "--detection-threshold": ("detection", "threshold"),
    "--filter-min-score": ("filtering", "min_score"),
    "--filter-max-iou": ("filtering", "max_iou"),
    "--tracker": ("tracking", "tracker"),
    "--track-start-score": ("tracking", "start_score"),
    "--track-high-score": ("tracking", "high_score"),
    "--track-lost-seconds": ("tracking", "lost_seconds"),
    "--track-min-iou": ("tracking", "min_iou"),
    "--track-min-frames": ("tracking", "min_frames"),
}


def add_setting_options(parser: argparse.ArgumentParser) -> None:
    defaults = PipelineConfig()
    group = parser.add_argument_group("settings (only the ones given change; see swim/pipeline/config.py)")
    for option, (stage, name) in SETTING_OPTIONS.items():
        settings = getattr(defaults, stage)
        meta = next(f.metadata for f in fields(settings) if f.name == name)
        default = getattr(settings, name)
        group.add_argument(
            option, dest=f"{stage}.{name}", type=type(default), choices=meta["choices"],
            metavar=None if meta["choices"] else name.upper(),
            help=f"{stage}: {meta['help']} (default: {default})".replace("%", "%%"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m swim.run", description="Analyze a swimming video.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="data/output", help="Results root folder; results go to <output>/<video name>/")
    parser.add_argument("--from", dest="start", choices=STAGES, help="First stage to run (earlier ones are loaded from saved results)")
    parser.add_argument("--to", dest="stop", choices=STAGES, help="Last stage to run")
    parser.add_argument("--render-only", action="store_true", help="Only draw saved results on the video, run no stage")
    parser.add_argument("--layers", help=f"Comma-separated layers to draw: {', '.join(LAYERS)} "
                                         "(default: the layer of the last stage with results)")
    add_setting_options(parser)
    args = parser.parse_args()

    output_dir = Path(args.output) / Path(args.input).stem

    if args.render_only:
        result = load_results(output_dir)
    else:
        changes: dict[str, dict] = {}
        for stage, name in SETTING_OPTIONS.values():
            value = vars(args)[f"{stage}.{name}"]
            if value is not None:
                changes.setdefault(stage, {})[name] = value
        try:
            config = PipelineConfig().with_changes(changes)
        except ValueError as e:
            parser.error(str(e))
        result = run_pipeline(args.input, output_dir, config, start=args.start, stop=args.stop)

    # By default draw only the last stage's layer: earlier layers show the same boxes before later stages changed them
    layers = args.layers.split(",") if args.layers else available_layers(result)[-1:]
    render_video(args.input, output_dir / "video.mp4", result, layers)


if __name__ == "__main__":
    main()
