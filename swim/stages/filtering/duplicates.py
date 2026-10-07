"""Filter: remove duplicate detections (boxes overlapping too much), keeping the higher-score one."""

from swim.stages.detection.types import Detection


def filter_duplicates(detections: list[Detection], max_iou: float) -> tuple[list[Detection], list[Detection]]:
    """Split into (kept, removed).

    Of two boxes overlapping with IoU > `max_iou`, the higher-score one is kept (on equal scores, the earlier one).
    Boxes are checked from the highest score down, only against boxes already kept, so a box that was removed
    never removes another one (greedy non-maximum suppression). Both lists keep the input order.
    """
    # sorted() is stable: equal scores keep their input order
    by_score = sorted(range(len(detections)), key=lambda i: -detections[i].score)
    kept_idx: list[int] = []
    for i in by_score:
        if all(detections[i].bbox.iou(detections[k].bbox) <= max_iou for k in kept_idx):
            kept_idx.append(i)

    kept_set = set(kept_idx)
    kept = [d for i, d in enumerate(detections) if i in kept_set]
    removed = [d for i, d in enumerate(detections) if i not in kept_set]
    return kept, removed
