import cv2
import numpy as np
import pytest

from swim.labeling.dataset import Dataset, Frame
from swim.labeling.labels import FrameLabels

W, H = 256, 144
WATER = [(0, 50), (255, 40), (255, 143), (0, 143)]  # the lower part of the frame


def make_dataset(root, videos: dict[str, int], status: str = "fixed") -> Dataset:
    """A small labeling dataset: `videos` maps a video name to its number of frames. Each frame: gray deck above,
    blue water below (the WATER polygon), a white lane rope across the water."""
    dataset = Dataset(root)
    frames = []
    for video, count in videos.items():
        for k in range(count):
            frame = Frame(name=f"{video}_{k * 60:06d}", video=f"data/input/{video}.MOV", frame_idx=k * 60, width=W, height=H)
            frames.append(frame)
            image = np.full((H, W, 3), 150, np.uint8)
            cv2.fillPoly(image, [np.array(WATER, np.int32)], (200, 120, 30))
            cv2.line(image, (0, 90), (255, 85), (240, 240, 240), 3)
            dataset.images.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dataset.image_path(frame.name)), image)
            dataset.save(FrameLabels(video=frame.video, frame_idx=frame.frame_idx, image=frame.image, width=W, height=H,
                                     status=status, water=[WATER], ropes=[[(0, 88), (255, 83), (255, 87), (0, 92)]]))
    dataset.add_frames(frames)
    return dataset


@pytest.fixture(name="make_dataset")
def make_dataset_fixture():
    return make_dataset


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path / "pool", {"IMG_1": 4, "IMG_2": 2})
