"""Training data for the pool model: the checked frames of the labeling dataset and their water masks.

The masks are drawn from the label files with the same `rasterize()` as `python -m swim.labeling masks`, so training
always uses the labels as they are now. Classes: 0 other, 1 water. Lane ropes lie in the pool, so their pixels count
as water.
"""

from collections import defaultdict

import cv2
import numpy as np
import torch

from swim.labeling.dataset import Dataset, Frame
from swim.labeling.labels import OTHER, REVIEWED, FrameLabels, rasterize
from swim.pool_model.network import MEAN, STD

CLASSES = ("other", "water")
IGNORE = 255  # mask value of pixels from outside the frame (after zooming out or turning): not counted

_MEAN = np.array(MEAN, np.float32)
_STD = np.array(STD, np.float32)


def water_mask(labels: FrameLabels) -> np.ndarray:
    """0 other, 1 water, one value per pixel of the full-size frame."""
    return (rasterize(labels) != OTHER).astype(np.uint8)


def checked_frames(dataset: Dataset) -> list[tuple[Frame, FrameLabels]]:
    """The frames whose labels were checked by hand (fixed or accepted): the only ones used for training."""
    frames = []
    for frame in dataset.frames():
        labels = dataset.labels(frame.name)
        if labels is not None and labels.status in REVIEWED:
            frames.append((frame, labels))
    return frames


def split(frames: list[Frame], val_every: int = 6) -> tuple[list[Frame], list[Frame]]:
    """Training and validation frames. Validation gets every `val_every`-th frame of each video (in time order,
    starting near the middle of the first group), so every video with 2 or more frames is checked; the rest train."""
    by_video = defaultdict(list)
    for frame in frames:
        by_video[frame.video].append(frame)
    train, val = [], []
    for video in sorted(by_video):
        group = sorted(by_video[video], key=lambda f: f.frame_idx)
        picked = set(range(min(val_every // 2, len(group) - 1), len(group), val_every)) if len(group) >= 2 else set()
        for k, frame in enumerate(group):
            (val if k in picked else train).append(frame)
    return train, val


def input_size(frame_shape: tuple[int, ...], long_side: int) -> tuple[int, int]:
    """The network input (width, height) for a frame: scaled so its long side is `long_side`, shape kept, sides
    rounded to multiples of 32. 1920×1080 -> 1024×576; an upright (portrait) 1080×1920 -> 576×1024."""
    height, width = frame_shape[:2]
    scale = long_side / max(height, width)
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A frame scaled to the network's input size (width, height), the same way in training and use."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def prepare(image: np.ndarray) -> np.ndarray:
    """Network input from a BGR image at the input size: RGB, normalized, channels first (float32)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1).copy()


def augment(image: np.ndarray, mask: np.ndarray, size: tuple[int, int], rng: np.random.Generator,
            photometric: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A random view of a frame at `size` (width, height), for training.

    Geometry (image and mask alike): zoom 0.8–1.6 × the scale used at inference (the frame's long side to the view's
    long side, `input_size`), mirror, shift, a tilt of up to 10°, and in 15% of views a quarter turn (the phone held
    sideways). Upright frames and turned views don't fill a landscape view: pixels from outside the frame get IGNORE.
    Then, on the image only: brightness, contrast, color, sometimes motion blur (fast pans) and noise.
    """
    width, height = size
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    quarter = rng.random() < 0.15
    scale = max(size) / max(mask.shape) * float(np.exp(rng.uniform(np.log(0.8), np.log(1.6))))
    # Scale first, the same way as at inference (resize), then turn and shift without scaling
    scaled_size = (max(1, round(mask.shape[1] * scale)), max(1, round(mask.shape[0] * scale)))
    image = cv2.resize(np.ascontiguousarray(image), scaled_size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(np.ascontiguousarray(mask), scaled_size, interpolation=cv2.INTER_NEAREST)
    angle = (90 * rng.choice([-1, 1]) if quarter else 0) + rng.uniform(-10, 10)
    turned_w, turned_h = (scaled_size[1], scaled_size[0]) if quarter else scaled_size
    shift = [rng.uniform(-1, 1) * abs(turned_w - width) / 2, rng.uniform(-1, 1) * abs(turned_h - height) / 2]
    matrix = cv2.getRotationMatrix2D((scaled_size[0] / 2, scaled_size[1] / 2), angle, 1.0)
    matrix[:, 2] += [width / 2 + shift[0] - scaled_size[0] / 2, height / 2 + shift[1] - scaled_size[1] / 2]
    image = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE)
    if photometric:
        image = _photometric(image, rng)
    return image, mask


def _photometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light and color changes (sunlit water, glare, other cameras), motion blur, sensor noise."""
    img = image.astype(np.float32)
    mean = img.mean()
    img = (img - mean) * rng.uniform(0.7, 1.3) + mean * rng.uniform(0.7, 1.3)  # contrast, brightness
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180  # hue (OpenCV: 0..180)
    hsv[..., 1] *= rng.uniform(0.7, 1.3)  # saturation
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if rng.random() < 0.3:
        length = int(rng.integers(3, 16))  # a straight streak in a random direction
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1
        turn = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), rng.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, turn, (length, length))
        img = cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))
    if rng.random() < 0.2:
        img += rng.normal(0, rng.uniform(2, 8), img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


class TrainingSet(torch.utils.data.Dataset):
    """Training frames as random views (`augment`): a different view of each frame every epoch."""

    def __init__(self, dataset: Dataset, frames: list[tuple[Frame, FrameLabels]], size: tuple[int, int]):
        self.images = [dataset.image_path(frame.name) for frame, _ in frames]
        self.labels = [labels for _, labels in frames]
        self.size = size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.images[k]))
        if image is None:
            raise FileNotFoundError(f"Can't read {self.images[k]}")
        mask = water_mask(self.labels[k])
        if image.shape[:2] != mask.shape:
            raise ValueError(f"{self.images[k]}: image is {image.shape[1]}×{image.shape[0]}, labels are for "
                             f"{mask.shape[1]}×{mask.shape[0]}")
        # Seeded from torch, which seeds each data loader worker differently
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
        image, mask = augment(image, mask, self.size, rng)
        return torch.from_numpy(prepare(image)), torch.from_numpy(mask.astype(np.int64))
