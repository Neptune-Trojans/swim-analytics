"""RF-DETR (Roboflow) person detector. Requires: pip install rfdetr"""

import cv2
import numpy as np
import rfdetr
from rfdetr.assets.coco_classes import COCO_CLASSES

from swim.stages.detection.types import BBox, Detection


class RFDETRDetector:
    """RF-DETR detector, keeping only the 'person' class."""

    MODELS = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "base": "RFDETRBase",
        "large": "RFDETRLarge",
    }

    def __init__(self, size: str = "medium", threshold: float = 0.5):
        if size not in self.MODELS:
            raise ValueError(f"Unknown RF-DETR size '{size}', choose from {list(self.MODELS)}")
        model_cls = getattr(rfdetr, self.MODELS[size], None)
        if model_cls is None:
            raise ValueError(f"{self.MODELS[size]} is not available in this rfdetr version")

        self.name = f"rfdetr-{size}"
        self.threshold = threshold
        self._model = model_cls()
        self._person_ids = {class_id for class_id, label in COCO_CLASSES.items() if label == "person"}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # RF-DETR expects RGB, OpenCV reads BGR
        result = self._model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=self.threshold)
        return [
            Detection(bbox=BBox(*map(float, box)), score=float(score))
            for box, class_id, score in zip(result.xyxy, result.class_id, result.confidence)
            if int(class_id) in self._person_ids
        ]
