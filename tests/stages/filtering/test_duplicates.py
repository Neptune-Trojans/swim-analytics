from swim.stages.detection.types import BBox, Detection
from swim.stages.filtering.duplicates import filter_duplicates


def det(x1: float, y1: float, x2: float, y2: float, score: float) -> Detection:
    return Detection(bbox=BBox(x1, y1, x2, y2), score=score)


def test_keeps_higher_score_of_two_overlapping_boxes():
    # IoU = 9000 / 11000 = 0.82
    high = det(0, 0, 100, 100, score=0.9)
    low = det(10, 0, 110, 100, score=0.6)

    kept, removed = filter_duplicates([low, high], max_iou=0.5)

    assert kept == [high]
    assert removed == [low]


def test_iou_equal_to_max_iou_is_not_a_duplicate():
    # The second box is the top half of the first: IoU = 5000 / 10000 = 0.5 exactly
    a = det(0, 0, 100, 100, score=0.9)
    b = det(0, 0, 100, 50, score=0.8)
    assert a.bbox.iou(b.bbox) == 0.5

    kept, removed = filter_duplicates([a, b], max_iou=0.5)

    assert kept == [a, b]
    assert removed == []


def test_slightly_overlapping_boxes_are_kept():
    # IoU = 2000 / 18000 = 0.11
    a = det(0, 0, 100, 100, score=0.9)
    b = det(80, 0, 180, 100, score=0.8)

    kept, removed = filter_duplicates([a, b], max_iou=0.5)

    assert kept == [a, b]
    assert removed == []


def test_equal_scores_keep_the_first_box():
    first = det(0, 0, 100, 100, score=0.8)
    second = det(10, 0, 110, 100, score=0.8)

    kept, removed = filter_duplicates([first, second], max_iou=0.5)

    assert kept == [first]
    assert removed == [second]


def test_removed_box_does_not_remove_others():
    # A-B and B-C overlap (IoU 0.54), A-C don't (IoU 0.25). B loses to A; C only overlaps the removed B,
    # so it stays
    a = det(0, 0, 100, 100, score=0.9)
    b = det(30, 0, 130, 100, score=0.8)
    c = det(60, 0, 160, 100, score=0.7)

    kept, removed = filter_duplicates([a, b, c], max_iou=0.5)

    assert kept == [a, c]
    assert removed == [b]


def test_keeps_input_order():
    a = det(0, 0, 100, 100, score=0.5)
    b = det(500, 0, 600, 100, score=0.9)
    c = det(1000, 0, 1100, 100, score=0.7)

    kept, removed = filter_duplicates([a, b, c], max_iou=0.5)

    assert kept == [a, b, c]
    assert removed == []


def test_max_iou_one_keeps_identical_boxes():
    a = det(0, 0, 100, 100, score=0.9)
    b = det(0, 0, 100, 100, score=0.8)

    kept, removed = filter_duplicates([a, b], max_iou=1.0)

    assert kept == [a, b]
    assert removed == []


def test_single_box_and_empty_frame():
    only = det(0, 0, 100, 100, score=0.9)

    assert filter_duplicates([only], max_iou=0.5) == ([only], [])
    assert filter_duplicates([], max_iou=0.5) == ([], [])
