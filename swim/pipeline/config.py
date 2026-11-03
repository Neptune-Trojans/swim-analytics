"""Settings for every pipeline stage. Saved as config.json next to the results.

Every setting is declared with `setting()`: its default, a short explanation and, for numbers, the allowed range.
The viewer's settings panel shows the explanation and checks the range.
"""

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any


def setting(default: Any, help: str, minimum: float | None = None, maximum: float | None = None,
            step: float | None = None, choices: tuple[str, ...] | None = None) -> Any:
    """A setting: default value, short explanation; for numbers the allowed range and a step for the viewer, for text
    the allowed values."""
    return field(default=default,
                 metadata={"help": help, "min": minimum, "max": maximum, "step": step, "choices": choices})


@dataclass
class DetectionConfig:
    model: str = setting("rfdetr", "Detection model.")
    size: str = setting("medium", "Model size. Larger finds more but is slower.",
                        choices=("nano", "small", "medium", "base", "large"))
    threshold: float = setting(
        0.1, "Lowest score the detector keeps. Low on purpose: tracking uses weak boxes to keep following a "
             "half-submerged swimmer. Changing it means re-running detection.", 0, 1, 0.05)


@dataclass
class FilteringConfig:
    min_score: float = setting(
        0.1, "Removes boxes scoring below this; they stay visible in gray as \"min_score\". Keep it low so tracking "
             "can still use weak boxes. 0 = off.", 0, 1, 0.05)
    max_iou: float = setting(
        0.5, "Of two boxes overlapping more than this (IoU), removes the lower-scoring one as a \"duplicate\". "
             "1 = off.", 0, 1, 0.05)


@dataclass
class TrackingConfig:
    tracker: str = setting(
        "bytetrack", "bytetrack: also uses weak boxes to keep following a half-submerged swimmer. ocsort: steadier after "
                     "missed frames, but ignores boxes below high_score. sort: the simple baseline.",
        choices=("bytetrack", "ocsort", "sort"))
    start_score: float = setting(
        0.5, "Lowest score that can start a new track; a weak box alone never starts one (bytetrack, sort).",
        0, 1, 0.05)
    high_score: float = setting(
        0.5, "bytetrack: boxes at or above this are matched first, weaker ones can only continue a track. "
             "ocsort: boxes below this are ignored.", 0, 1, 0.05)
    lost_seconds: float = setting(
        1.0, "How long a track without boxes waits to be found again before it ends.", 0, 5, 0.1)
    min_iou: float = setting(
        0.1, "How much a box must overlap (IoU) a track's predicted box to continue that track. Low, because the "
             "swimmer's box changes shape a lot between strokes.", 0, 1, 0.05)
    min_frames: int = setting(
        2, "Frames in a row a new person must be seen before becoming a track; higher values drop brief false boxes. "
           "The track's first frames get its ID too.", 2, 10, 1)


@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        """Settings saved with to_json(). Stages or settings missing from the file get their defaults; unknown ones
        (from an older version) are ignored."""
        saved = json.loads(Path(path).read_text())
        config = cls()
        for stage in fields(config):
            if isinstance(saved.get(stage.name), dict):
                current = getattr(config, stage.name)
                known = {f.name for f in fields(current)}
                values = {key: value for key, value in saved[stage.name].items() if key in known}
                config = replace(config, **{stage.name: replace(current, **values)})
        return config

    def with_changes(self, changes: dict[str, dict[str, Any]]) -> "PipelineConfig":
        """A copy with some settings changed, e.g. {"filtering": {"min_score": 0.3}}.

        Raises ValueError for an unknown stage or setting, a value of the wrong type or outside the allowed range.
        """
        config = self
        stages = {f.name for f in fields(config)}
        for stage, values in changes.items():
            if stage not in stages:
                raise ValueError(f"Unknown stage '{stage}'")
            current = getattr(config, stage)
            declared = {f.name: f for f in fields(current)}
            checked = {}
            for key, value in values.items():
                if key not in declared:
                    raise ValueError(f"Unknown setting '{stage}.{key}'")
                kind = type(getattr(current, key))
                meta = declared[key].metadata
                if kind in (int, float):
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"'{stage}.{key}' must be a number")
                    if kind is int and value != int(value):
                        raise ValueError(f"'{stage}.{key}' must be a whole number")
                    low, high = meta.get("min"), meta.get("max")
                    if (low is not None and value < low) or (high is not None and value > high):
                        raise ValueError(f"'{stage}.{key}' must be between {low} and {high}")
                    value = kind(value)
                elif not isinstance(value, kind):
                    raise ValueError(f"'{stage}.{key}' must be a {kind.__name__}")
                elif meta.get("choices") and value not in meta["choices"]:
                    raise ValueError(f"'{stage}.{key}' must be one of: {', '.join(meta['choices'])}")
                checked[key] = value
            config = replace(config, **{stage: replace(current, **checked)})
        return config


def describe(settings: Any) -> list[dict]:
    """One stage's settings for display: name, value, type, explanation and allowed range or values of each."""
    return [
        {"name": f.name, "value": getattr(settings, f.name), "type": type(getattr(settings, f.name)).__name__, **f.metadata}
        for f in fields(settings)
    ]
