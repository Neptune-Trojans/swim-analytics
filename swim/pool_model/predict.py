"""A trained pool model: the class of every pixel of a frame (0 other, 1 water)."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from swim.pool_model.data import input_size, prepare, resize
from swim.pool_model.network import UNet


def pick_device(name: str | None = None) -> torch.device:
    """The given device, or the fastest one here: the Mac's GPU (mps), an NVIDIA GPU (cuda), else the CPU."""
    if name and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_classes(net: UNet, frame: np.ndarray, size: tuple[int, int], device: torch.device) -> np.ndarray:
    """Class number per pixel of a BGR frame, at the frame's own size (uint8). The frame is scaled like the training
    frames (its long side to the long side of `size`, the training view; shape kept, `input_size`), and the scores
    scaled back up before picking the class."""
    net.eval()
    x = torch.from_numpy(prepare(resize(frame, input_size(frame.shape, max(size)))))[None].to(device)
    scores = F.interpolate(net(x), size=frame.shape[:2], mode="bilinear", align_corners=False)
    return scores.argmax(1)[0].to(torch.uint8).cpu().numpy()


class PoolModel:
    """A model saved by `python -m swim.pool_model train` (its model.pt)."""

    def __init__(self, net: UNet, classes: tuple[str, ...], size: tuple[int, int], device: torch.device):
        self.net = net.to(device).eval()
        self.classes = classes
        self.size = size
        self.device = device

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "PoolModel":
        saved = torch.load(path, map_location="cpu", weights_only=True)
        net = UNet(len(saved["classes"]), saved["encoder"], pretrained=False)
        net.load_state_dict(saved["network"])
        return cls(net, tuple(saved["classes"]), tuple(saved["size"]), pick_device(device))

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Class number per pixel of a BGR frame (uint8, the frame's size): 0 other, 1 water."""
        return predict_classes(self.net, frame, self.size, self.device)
