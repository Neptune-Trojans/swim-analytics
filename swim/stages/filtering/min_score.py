"""Filter: remove detections with a low confidence score."""

from swim.stages.detection.types import Detection


def filter_min_score(detections: list[Detection], min_score: float) -> tuple[list[Detection], list[Detection]]:
    """Split into (kept, removed): removed are the detections scoring below `min_score`."""
    kept = [d for d in detections if d.score >= min_score]
    removed = [d for d in detections if d.score < min_score]
    return kept, removed
