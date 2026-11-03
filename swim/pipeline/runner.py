"""The pipeline runner: runs the stages (swim/stages/) in order and saves each stage's output.

Planned stages: detection -> filtering -> tracking -> swimmer -> pose -> metrics (see docs/pipeline.md).
Implemented so far: detection, filtering, tracking.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from swim.pipeline.config import PipelineConfig
from swim.stages.detection.stage import detect
from swim.stages.detection.types import VideoDetections
from swim.stages.filtering.stage import filter_detections
from swim.stages.tracking.stage import track
from swim.utils.video import get_video_info

# Implemented stages, in order
STAGES = ["detection", "filtering", "tracking"]

# Saved output of each stage: file name in the video's output folder, and its data class
STAGE_OUTPUTS = {
    "detection": ("1_detections.json", VideoDetections),
    "filtering": ("2_filtered.json", VideoDetections),
    "tracking": ("3_tracks.json", VideoDetections),
}

# progress(stage, frames done, frames in the video): called when a stage starts (0) and ends (all frames), and after
# every frame in the slow stages (detection). It may raise to stop the run: the stage being run is not saved.
Progress = Callable[[str, int, int], None]


@dataclass
class PipelineResult:
    """Output of every stage. None = that stage was not run (and not loaded)."""

    detections: VideoDetections | None = None
    filtered: VideoDetections | None = None
    tracks: VideoDetections | None = None


def run_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig | None = None,
    start: str | None = None,
    stop: str | None = None,
    progress: Progress | None = None,
) -> PipelineResult:
    """Run stages `start` to `stop` (default: all) and save their outputs to `output_dir`.

    Stages before `start` are loaded from their saved files; stages after `stop` are skipped.

    The saved results stay consistent even if the run stops halfway (an error, or `progress` raising to cancel): each
    stage's output is saved when the stage is done, and only then are the outputs of the stages after it (made from
    the old one) deleted and its settings recorded in config.json.
    """
    config = config or PipelineConfig()
    output_dir = Path(output_dir)
    start_idx = STAGES.index(start or STAGES[0])
    stop_idx = STAGES.index(stop or STAGES[-1])
    if start_idx > stop_idx:
        raise ValueError(f"Stage '{start}' comes after stage '{stop}'")
    frame_count = get_video_info(video_path).frame_count  # fail before creating any output if the video can't be opened

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    # The settings the saved results were made with, updated as each stage is saved
    recorded = PipelineConfig.from_json(config_path) if config_path.exists() else config
    if start_idx > 0:
        # Stages before `start` are loaded, not run: keep the settings their saved results were made with.
        # Each stage's settings are the PipelineConfig field with the stage's name.
        config = replace(config, **{name: getattr(recorded, name) for name in STAGES[:start_idx] if hasattr(config, name)})
    report = progress or (lambda stage, done, total: None)

    def stage(name, run):
        nonlocal recorded
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
        report(name, 0, frame_count)
        run().to_json(path)
        print(f"[{name}] saved {path}")
        # Saved results of the later stages were made from this stage's old output: remove them (the ones up to
        # `stop` are re-made next)
        for later in STAGES[idx + 1:]:
            later_path = output_dir / STAGE_OUTPUTS[later][0]
            if later_path.exists():
                later_path.unlink()
                if STAGES.index(later) > stop_idx:
                    print(f"[{later}] removed outdated {later_path}")
        if hasattr(config, name):
            recorded = replace(recorded, **{name: getattr(config, name)})
        recorded.to_json(config_path)
        report(name, frame_count, frame_count)
        # Continue from the saved file (not the in-memory result), so a full run and a run resumed
        # with --from give later stages exactly the same input
        return data_cls.from_json(path)

    def on_frame(done: int, total: int) -> None:
        report("detection", done, total)

    result = PipelineResult()
    # 1. Detection: all persons in every frame
    result.detections = stage("detection", lambda: detect(video_path, config.detection, on_frame))
    # 2. Filtering: remove problematic detections (kept as "rejected" with the reason)
    result.filtered = stage("filtering", lambda: filter_detections(result.detections, config.filtering))
    # 3. Tracking: the same person keeps the same track_id from frame to frame
    result.tracks = stage("tracking", lambda: track(result.filtered, config.tracking))
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
    return PipelineResult(
        detections=loaded.get("detection"), filtered=loaded.get("filtering"), tracks=loaded.get("tracking")
    )
