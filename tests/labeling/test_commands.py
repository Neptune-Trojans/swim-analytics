import cv2

from swim.labeling.commands import auto_label, pick, write_masks
from swim.labeling.dataset import Dataset
from swim.labeling.frames import pick_indices
from swim.labeling.labels import FrameLabels


class FakeLabeler:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def label(self, frame):
        self.calls += 1
        h, w = frame.shape[:2]
        water = [[(0, h / 2), (w - 1, h / 2), (w - 1, h - 1), (0, h - 1)]]
        return water, [[(0, h * 0.7 - 1.5), (w - 1, h * 0.7 - 1.5), (w - 1, h * 0.7 + 1.5), (0, h * 0.7 + 1.5)]]


def test_pick_indices_spreads_frames_and_caps_them():
    assert pick_indices(300, 30.0, 2.0, 25) == [30, 90, 150, 210, 270]  # 10 s: one frame every 2 s, centered
    assert len(pick_indices(10_000, 30.0, 2.0, 25)) == 25
    assert pick_indices(20, 30.0, 2.0, 25) == [10]  # shorter than one step: one frame from the middle


def test_pick_saves_frames_once(tmp_path, make_video):
    video = make_video(tmp_path / "input" / "IMG_1.mp4", frames=120)  # 4 s at 30 fps
    dataset = Dataset(tmp_path / "labels")

    assert pick(dataset, [video], every_seconds=1.0, max_per_video=25) == [
        "IMG_1_000015", "IMG_1_000045", "IMG_1_000075", "IMG_1_000105"]
    assert pick(dataset, [video], every_seconds=1.0, max_per_video=25) == []
    frame = dataset.frame("IMG_1_000045")
    assert (frame.video, frame.frame_idx, frame.width, frame.height) == (str(video), 45, 64, 48)
    assert cv2.imread(str(dataset.image_path("IMG_1_000045"))).shape == (48, 64, 3)


def test_auto_labels_new_frames_and_never_replaces_checked_labels(tmp_path, make_video):
    dataset = Dataset(tmp_path / "labels")
    pick(dataset, [make_video(tmp_path / "IMG_1.mp4", frames=60)], every_seconds=1.0, max_per_video=25)
    labeler = FakeLabeler()

    assert auto_label(dataset, labeler=labeler) == 2
    first = dataset.labels("IMG_1_000015")
    assert (first.status, first.source, len(first.water), len(first.ropes)) == ("sam3", "fake", 1, 1)
    assert dataset.sam3("IMG_1_000015") == first
    assert auto_label(dataset, labeler=labeler) == 0  # nothing new

    first.status, first.ropes = "fixed", []
    dataset.save(first)
    assert auto_label(dataset, redo=True, labeler=labeler) == 2
    assert dataset.labels("IMG_1_000015") == first  # checked by hand: kept
    assert dataset.labels("IMG_1_000045").status == "sam3"
    assert dataset.status_counts() == {"no labels": 0, "sam3": 1, "fixed": 1, "accepted": 0}


def test_masks_only_for_checked_frames(tmp_path, make_video):
    dataset = Dataset(tmp_path / "labels")
    pick(dataset, [make_video(tmp_path / "IMG_1.mp4", frames=60)], every_seconds=1.0, max_per_video=25)
    auto_label(dataset, labeler=FakeLabeler())
    checked = dataset.labels("IMG_1_000045")
    checked.status = "accepted"
    dataset.save(checked)

    assert write_masks(dataset) == 1
    mask = cv2.imread(str(dataset.masks / "IMG_1_000045.png"), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (48, 64) and set(mask.flatten()) == {0, 1, 2}
    assert write_masks(dataset, include_unreviewed=True) == 2

    checked.status = "sam3"
    dataset.save(checked)
    assert write_masks(dataset) == 0 and not list(dataset.masks.glob("*.png"))  # old mask removed
