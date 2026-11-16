import pytest

from swim.labeling.labels import ROPE, WATER, FrameLabels, band_corners, rasterize

ROPE_OUTLINE = [(10, 28), (90, 29), (90, 31), (10, 32)]  # 4 px wide at its left end, 2 px at its right end


def labels(**changes):
    data = dict(video="data/input/IMG_1.MOV", frame_idx=30, image="IMG_1_000030.jpg", width=100, height=60,
                water=[[(10, 10), (90, 10), (90, 50), (10, 50)]], ropes=[ROPE_OUTLINE])
    return FrameLabels(**(data | changes))


def test_round_trip(tmp_path):
    original = labels(status="fixed", source="sam3-litetext-s0")
    original.to_json(tmp_path / "x.json")

    assert FrameLabels.from_json(tmp_path / "x.json") == original
    assert original.name == "IMG_1_000030"


@pytest.mark.parametrize("changes, message", [
    ({"status": "done"}, "Unknown status"),
    ({"water": [[(0, 0), (10, 0)]]}, "at least 3"),
    ({"ropes": [[(0, 0), (5, 5)]]}, "Rope 1 has 2 corners"),
    ({"ropes": [[(0, 0), (5, float("inf")), (9, 9)]]}, "Bad point"),
    ({"water": [[(0, 0), (10, float("nan")), (5, 5)]]}, "Bad point"),
])
def test_bad_labels_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        labels(**changes).validate()


def test_ropes_from_older_files_are_read_as_their_band():
    """Label files written before 2026-11-16 store a rope as a line with a width at each end."""
    data = labels().to_dict() | {"ropes": [{"points": [[10, 30], [90, 30]], "widths": [4, 2]}]}

    assert FrameLabels.from_dict(data).ropes == [[(10, 32), (90, 31), (90, 29), (10, 28)]]
    assert band_corners([(0, 0), (0, 10)], [2, 2]) == [(-1, 0), (-1, 10), (1, 10), (1, 0)]


def test_from_dict_reports_missing_fields():
    with pytest.raises(ValueError, match="Bad labels"):
        FrameLabels.from_dict({"video": "x", "frame_idx": 1})


def test_rasterize_draws_water_then_ropes_on_top():
    mask = rasterize(labels())

    assert mask.shape == (60, 100)
    assert mask[20, 50] == WATER
    assert mask[30, 50] == ROPE  # on the rope's line, inside the water
    assert mask[5, 5] == 0  # outside the pool
    # The rope is 4 px wide at its left end, 2 px at its right end
    assert (mask[:, 15] == ROPE).sum() > (mask[:, 85] == ROPE).sum()
