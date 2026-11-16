import pytest

from swim.labeling.labels import ROPE, WATER, FrameLabels, Rope, rasterize


def labels(**changes):
    data = dict(video="data/input/IMG_1.MOV", frame_idx=30, image="IMG_1_000030.jpg", width=100, height=60,
                water=[[(10, 10), (90, 10), (90, 50), (10, 50)]], ropes=[Rope([(10, 30), (90, 30)], [4, 2])])
    return FrameLabels(**(data | changes))


def test_round_trip(tmp_path):
    original = labels(status="fixed", source="sam3-litetext-s0")
    original.to_json(tmp_path / "x.json")

    assert FrameLabels.from_json(tmp_path / "x.json") == original
    assert original.name == "IMG_1_000030"


@pytest.mark.parametrize("changes, message", [
    ({"status": "done"}, "Unknown status"),
    ({"water": [[(0, 0), (10, 0)]]}, "at least 3"),
    ({"ropes": [Rope([(0, 0)], [3])]}, "2 points and 2 widths"),
    ({"ropes": [Rope([(0, 0), (5, 5)], [3, 0])]}, "positive"),
    ({"water": [[(0, 0), (10, float("nan")), (5, 5)]]}, "Bad point"),
])
def test_bad_labels_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        labels(**changes).validate()


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
