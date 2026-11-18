"""The pool model from the pool-segmentation package (a U-Net trained on our frames with
`python -m pool_segmentation.model train`; ../pool-segmentation, its docs/model.md): the water's outline."""

from pathlib import Path

import numpy as np

from swim.stages.pool.types import Polygon


class UNetSegmenter:
    def __init__(self, model_path: Path, device: str | None = None):
        from pool_segmentation import PoolModel  # imports torch

        self.model = PoolModel.load(model_path, device)
        self.name = f"pool-unet {model_path.parent.name}"

    def water(self, frame: np.ndarray) -> list[Polygon]:
        # Holes filled (a swimmer the model missed counts as pool), specks dropped, 3 px tolerance
        return self.model.water(frame)
