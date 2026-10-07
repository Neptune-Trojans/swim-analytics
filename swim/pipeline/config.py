"""Settings for every pipeline stage. Saved as config.json next to the results."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DetectionConfig:
    model: str = "rfdetr"
    size: str = "medium"  # rfdetr: nano, small, medium, base, large
    threshold: float = 0.5  # drop detections below this confidence


@dataclass
class FilteringConfig:
    min_score: float = 0.4  # remove detections below this confidence (0 = off)
    max_iou: float = 0.5  # of two boxes overlapping more than this (IoU), keep the higher score (1 = off)


@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))
