import cv2
import numpy as np

from swim.labeling.shapes import Rope, join_pieces, person_in_water, pool_polygons, rope_from_mask, ropes_from_masks

H, W = 200, 300


def rope_mask(p1, p2, width):
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, p1, p2, 1, width)
    return m.astype(bool)


def test_pool_polygon_fills_swimmer_holes_and_keeps_few_corners():
    water = np.zeros((H, W), bool)
    water[50:180, 20:280] = True
    water[100:120, 100:160] = False  # a swimmer inside the pool: a hole in SAM3's mask

    [polygon] = pool_polygons(water)

    assert len(polygon) == 4
    xs, ys = zip(*polygon)
    assert (min(xs), max(xs), min(ys), max(ys)) == (20, 279, 50, 179)


def test_tiny_pool_pieces_are_dropped():
    water = np.zeros((H, W), bool)
    water[50:180, 20:280] = True
    water[5:8, 5:8] = True  # a few pixels of "water" on the deck

    assert len(pool_polygons(water)) == 1


def test_swimmer_on_the_pool_edge_is_added_but_a_person_on_the_deck_is_not():
    water = np.zeros((H, W), bool)
    water[60:, :] = True
    swimmer = np.zeros((H, W), bool)
    swimmer[50:90, 100:130] = True  # head over the edge, mostly in the water
    deck_person = np.zeros((H, W), bool)
    deck_person[5:55, 200:220] = True  # standing on the deck, feet at the edge

    assert person_in_water(swimmer, water)
    assert not person_in_water(deck_person, water)
    [polygon] = pool_polygons(water, persons=[swimmer, deck_person])
    assert min(y for _, y in polygon) == 50  # the swimmer's notch is filled, the deck person left out


def test_rope_mask_becomes_a_straight_line_with_its_width():
    rope = rope_from_mask(rope_mask((20, 40), (280, 160), 7))

    (x1, y1), (x2, y2) = rope.points
    assert abs(x1 - 20) <= 3 and abs(y1 - 40) <= 3 and abs(x2 - 280) <= 3 and abs(y2 - 160) <= 3
    assert all(5 <= w <= 10 for w in rope.widths)


def test_blobs_are_not_ropes():
    blob = np.zeros((H, W), bool)
    blob[50:90, 50:90] = True

    assert rope_from_mask(blob) is None


def test_a_rope_cut_by_a_swimmer_is_one_line_but_two_ropes_stay_two():
    left = rope_mask((10, 50), (120, 70), 5)
    right = rope_mask((180, 81), (290, 101), 5)  # same line, after a gap (the swimmer)
    other = rope_mask((10, 120), (290, 170), 5)  # the next rope

    ropes = ropes_from_masks([left, right, other])

    assert len(ropes) == 2
    joined = min(ropes, key=lambda r: r.points[0][1])
    assert abs(joined.points[0][0] - 10) <= 3 and abs(joined.points[1][0] - 290) <= 3


def test_join_keeps_the_width_of_the_outer_ends():
    a = Rope([(0, 0), (100, 0)], [8, 6])
    b = Rope([(150, 0), (300, 0)], [5, 3])

    [rope] = join_pieces([b, a])

    assert rope.points == [(0, 0), (300, 0)] and rope.widths == [8, 3]


def test_pool_split_by_a_thin_gap_is_one_area():
    water = np.zeros((H, W), bool)
    water[40:190, 10:290] = True
    water[100:102, :] = False  # a rope SAM3's masks don't cover: a 2 px gap across the pool

    assert len(pool_polygons(water)) == 1


def test_corners_along_the_frame_edge_are_dropped():
    water = np.zeros((H, W), bool)
    water[60:, :] = True
    for x in range(0, W, 20):
        water[60:200, x:x + 3] = False if x % 40 else True  # jagged mask just inside the bottom-left
    water[:, :2] = False  # SAM3's mask stops 2 px short of the frame's left edge
    water[150:, 296:] = False  # and a bit short of the right edge, lower down

    [polygon] = pool_polygons(water)

    xs = sorted({x for x, _ in polygon})
    assert xs[0] == 0 and xs[-1] == W - 1  # snapped onto the frame's edges
    on_left = [p for p in polygon if p[0] == 0]
    assert len(on_left) == 2  # only the two ends of the run along the left edge


def test_a_swimmer_cut_off_by_the_frame_edge_is_filled_but_the_deck_corner_is_not():
    water = np.zeros((H, W), bool)
    water[:, :] = True
    water[80:110, :40] = False  # a swimmer entering the picture from the left: a notch, not a hole
    water[:50, 200:] = False  # the deck in the top-right corner

    [polygon] = pool_polygons(water)

    assert all(not (0 < x < 40 and 80 < y < 110) for x, y in polygon)  # no corners around the swimmer
    assert (200, 0) in polygon or any(abs(x - 200) <= 8 and y <= 8 for x, y in polygon)  # the deck stays out
    assert len(polygon) <= 7


def test_edit_water_merges_an_added_area_and_cuts_one_out():
    from swim.labeling.shapes import edit_water

    pool = [[(0, 60), (299, 60), (299, 199), (0, 199)]]
    notch = [[(0, 60), (299, 60), (299, 199), (0, 199), (0, 130), (40, 130), (40, 100), (0, 100)]]  # a swimmer cut out

    [merged] = edit_water(notch, W, H, add=[(0, 90), (60, 90), (60, 140), (0, 140)])
    assert sorted(merged) == sorted(pool[0])  # the notch is gone: back to 4 corners

    [cut] = edit_water(pool, W, H, subtract=[(200, 40), (310, 40), (310, 100), (200, 100)])  # deck wrongly marked
    assert max(y for x, y in cut if x > 250) == 199 and min(y for x, y in cut if x > 250) >= 99

    assert edit_water(pool, W, H, add=[(50, 0), (100, 0), (100, 20), (50, 20)])[1:]  # a separate area: two polygons
