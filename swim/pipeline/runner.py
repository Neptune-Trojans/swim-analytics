"""The pipeline runner: runs the stages (swim/stages/) in order and saves each stage's output.

Planned stages: detection -> filtering -> tracking -> swimmer -> pose -> metrics (see docs/pipeline.md).
Implemented so far: detection.
"""

from dataclasses import dataclass
from pathlib import Path

from swim.pipeline.config import PipelineConfig
from swim.stages.detection.stage import detect
from swim.stages.detection.types import VideoDetections
from swim.utils.video import get_video_info

# Implemented stages, in order
STAGES = ["detection"]

# Saved output of each stage: file name in the video's output folder, and its data class
STAGE_OUTPUTS = {
    "detection": ("1_detections.json", VideoDetections),
}


@dataclass
class PipelineResult:
    """Output of every stage. None = that stage was not run (and not loaded)."""

    detections: VideoDetections | None = None


def run_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig | None = None,
    start: str | None = None,
    stop: str | None = None,
) -> PipelineResult:
    """Run stages `start` to `stop` (default: all) and save their outputs to `output_dir`.

    Stages before `start` are loaded from their saved files; stages after `stop` are skipped.
    """
    config = config or PipelineConfig()
    output_dir = Path(output_dir)
    start_idx = STAGES.index(start or STAGES[0])
    stop_idx = STAGES.index(stop or STAGES[-1])
    if start_idx > stop_idx:
        raise ValueError(f"Stage '{start}' comes after stage '{stop}'")
    get_video_info(video_path)  # fail before creating any output if the video can't be opened

    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(output_dir / "config.json")

    def stage(name, run):
        idx = STAGES.index(name)
        filename, data_cls = STAGE_OUTPUTS[name]
        path = output_dir / filename
        if idx < start_idx:
            if not path.exists():
                raise FileNotFoundError(f"No saved output for stage '{name}' ({path}): run that stage first")
            print(f"[{name}] loaded {path}")
            return data_cls.from_json(path)
        if idx > stop_idx:
            return None
        print(f"[{name}] running")
        run().to_json(path)
        print(f"[{name}] saved {path}")
        # Continue from the saved file (not the in-memory result), so a full run and a run resumed
        # with --from give later stages exactly the same input
        return data_cls.from_json(path)

    result = PipelineResult()
    # 1. Detection: all persons in every frame
    result.detections = stage("detection", lambda: detect(video_path, config.detection))
    return result


def load_results(output_dir: str | Path) -> PipelineResult:
    """Load every saved stage output found in `output_dir` (missing ones stay None)."""
    output_dir = Path(output_dir)
    loaded = {}
    for name in STAGES:
        filename, data_cls = STAGE_OUTPUTS[name]
        if (output_dir / filename).exists():
            loaded[name] = data_cls.from_json(output_dir / filename)
    if not loaded:
        raise FileNotFoundError(f"No saved results in {output_dir}: run the pipeline first")
    return PipelineResult(detections=loaded.get("detection"))
