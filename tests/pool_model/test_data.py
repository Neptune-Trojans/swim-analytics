import numpy as np
import pytest

from swim.labeling.dataset import Frame
from swim.labeling.labels import FrameLabels
from swim.pool_model.data import IGNORE, TrainingSet, augment, checked_frames, input_size, split, water_mask


def frames(video: str, count: int) -> list[Frame]:
    return [Frame(f"{video}_{k:06d}", f"data/input/{video}.MOV", k * 60, 1920, 1080) for k in range(count)]


def test_water_mask_counts_ropes_as_water():
    labels = FrameLabels(video="v", frame_idx=0, image="v_000000.jpg", width=100, height=60,
                         water=[[(10, 10), (90, 10), (90, 50), (10, 50)]], ropes=[[(0, 28), (99, 28), (99, 32), (0, 32)]])
    mask = water_mask(labels)
    assert set(np.unique(mask)) == {0, 1}
    assert mask[20, 50] == 1 and mask[5, 5] == 0
    assert mask[30, 3] == 1  # a rope outside the water polygon: still pool


def test_split_validates_every_video_with_two_or_more_frames():
    train, val = split(frames("A", 25) + frames("B", 4) + frames("C", 2) + frames("D", 1), val_every=6)
    by_video = lambda fs: sorted({f.video for f in fs})
    assert [f.frame_idx // 60 for f in val if "A" in f.video] == [3, 9, 15, 21]
    assert [f.frame_idx // 60 for f in val if "B" in f.video] == [3]
    assert [f.frame_idx // 60 for f in val if "C" in f.video] == [1]
    assert by_video(val) == ["data/input/A.MOV", "data/input/B.MOV", "data/input/C.MOV"]  # D has 1 frame: trains
    assert len(train) + len(val) == 32 and not {f.name for f in train} & {f.name for f in val}


@pytest.mark.parametrize("frame, expected", [
    ((1080, 1920), (1024, 576)),  # landscape
    ((1920, 1080), (576, 1024)),  # upright (portrait): shape kept, not squeezed into 1024×576
    ((1080, 1440), (1024, 768)),  # 4:3
])
def test_input_keeps_the_frame_shape(frame, expected):
    assert input_size(frame, 1024) == expected


def test_only_checked_frames_train(tmp_path, make_dataset):
    dataset = make_dataset(tmp_path / "a", {"IMG_1": 2})
    make_dataset(tmp_path / "a", {"IMG_9": 1}, status="sam3")  # same folder: one more frame, not checked
    assert [f.name for f, _ in checked_frames(dataset)] == ["IMG_1_000000", "IMG_1_000060"]


@pytest.mark.parametrize("seed", range(12))
def test_augment_moves_image_and_mask_together(seed):
    """A frame whose left half is white water and right half black deck: after any view, the mask's water is
    where the image is bright (only the geometry is checked, so no light changes)."""
    image = np.zeros((90, 160, 3), np.uint8)
    image[:, :80] = 255
    mask = np.zeros((90, 160), np.uint8)
    mask[:, :80] = 1

    view, view_mask = augment(image, mask, (128, 64), np.random.default_rng(seed), photometric=False)

    assert view.shape == (64, 128, 3) and view_mask.shape == (64, 128)
    gray = view[..., 0].astype(int)
    inner = np.ones_like(view_mask, bool)  # away from the edge between the halves (blurred by scaling)
    edges = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1])) + np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    inner &= np.convolve(edges.ravel() > 0, np.ones(5), "same").reshape(edges.shape) == 0
    assert (gray[(view_mask == 1) & inner] > 200).all()
    assert (gray[(view_mask == 0) & inner] < 50).all()
    assert set(np.unique(view_mask)) <= {0, 1, IGNORE}


def test_training_set_gives_normalized_images_and_masks(dataset):
    item_image, item_mask = TrainingSet(dataset, checked_frames(dataset), (128, 64))[0]  # frames are 256×144
    assert item_image.shape == (3, 64, 128) and item_image.dtype.is_floating_point
    assert item_mask.shape == (64, 128) and set(item_mask.unique().tolist()) <= {0, 1, IGNORE}
