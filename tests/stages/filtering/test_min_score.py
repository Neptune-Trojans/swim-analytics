from swim.stages.detection.types import BBox, Detection
from swim.stages.filtering.min_score import filter_min_score


def det(score: float, x: float = 0) -> Detection:
    return Detection(bbox=BBox(x, 0, x + 100, 100), score=score)


def test_removes_detections_below_min_score():
    low, ok, high = det(0.3), det(0.5), det(0.9)

    kept, removed = filter_min_score([low, ok, high], min_score=0.4)

    assert kept == [ok, high]
    assert removed == [low]


def test_score_equal_to_min_score_is_kept():
    exact = det(0.4)

    kept, removed = filter_min_score([exact], min_score=0.4)

    assert kept == [exact]
    assert removed == []


def test_keeps_input_order():
    a, b, c, d = det(0.9, x=0), det(0.2, x=200), det(0.6, x=400), det(0.1, x=600)

    kept, removed = filter_min_score([a, b, c, d], min_score=0.4)

    assert kept == [a, c]
    assert removed == [b, d]


def test_min_score_zero_keeps_everything():
    dets = [det(0.0), det(0.01), det(0.99)]

    kept, removed = filter_min_score(dets, min_score=0)

    assert kept == dets
    assert removed == []


def test_empty_frame():
    assert filter_min_score([], min_score=0.4) == ([], [])
