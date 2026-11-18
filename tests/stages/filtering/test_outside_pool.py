from swim.stages.detection.types import BBox, Detection
from swim.stages.filtering.outside_pool import filter_outside_pool

# A 1920×1080 frame: the pool's water from y = 400 down (the far wall's water line at y = 400)
WATER = [[(0, 400), (1919, 400), (1919, 1079), (0, 1079)]]


def box(x1, y1, x2, y2, score=0.8) -> Detection:
    return Detection(BBox(x1, y1, x2, y2), score)


def test_swimmer_in_the_water_is_kept_and_a_person_on_the_deck_removed():
    swimmer = box(500, 600, 700, 700)
    on_deck = box(1000, 100, 1080, 380)  # feet 20 px above the water line

    kept, removed = filter_outside_pool([swimmer, on_deck], WATER, margin=10)

    assert (kept, removed) == ([swimmer], [on_deck])


def test_swimmer_at_the_far_wall_with_the_head_above_the_water_line_is_kept():
    # 78% of the box is above the water line, but the bottom-center is in the water
    swimmer = box(800, 330, 900, 420)

    assert filter_outside_pool([swimmer], WATER, margin=0) == ([swimmer], [])


def test_bottom_center_on_the_water_line_is_in_the_pool():
    on_the_line = box(800, 300, 900, 400)

    assert filter_outside_pool([on_the_line], WATER, margin=0) == ([on_the_line], [])


def test_bottom_center_within_the_margin_outside_the_water_is_kept():
    just_outside = box(100, 300, 200, 392)  # 8 px above the water line
    at_the_margin = box(300, 300, 400, 390)  # 10 px
    beyond = box(500, 300, 600, 389)  # 11 px

    kept, removed = filter_outside_pool([just_outside, at_the_margin, beyond], WATER, margin=10)

    assert (kept, removed) == ([just_outside, at_the_margin], [beyond])


def test_only_the_bottom_center_counts_not_the_corners():
    # A person standing at the pool's side: the box's lower right corner reaches into the water, the center doesn't
    water = [[(1000, 400), (1919, 400), (1919, 1079), (1000, 1079)]]
    beside = box(800, 500, 1100, 900)  # bottom-center x = 950, 50 px left of the water

    assert filter_outside_pool([beside], water, margin=10) == ([], [beside])


def test_any_piece_of_water_counts():
    # The water found in two pieces (e.g. split by a pillar in front of the camera)
    water = [[(0, 400), (900, 400), (900, 1079), (0, 1079)], [(1000, 400), (1919, 400), (1919, 1079), (1000, 1079)]]
    left, right, between = box(100, 500, 200, 700), box(1500, 500, 1600, 700), box(900, 100, 1000, 300)

    kept, removed = filter_outside_pool([left, right, between], water, margin=10)

    assert (kept, removed) == ([left, right], [between])


def test_frame_without_water_keeps_every_detection():
    dets = [box(500, 600, 700, 700), box(1000, 100, 1080, 380)]

    assert filter_outside_pool(dets, [], margin=10) == (dets, [])


def test_no_detections():
    assert filter_outside_pool([], WATER, margin=10) == ([], [])
