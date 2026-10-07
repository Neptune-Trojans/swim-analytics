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
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))
