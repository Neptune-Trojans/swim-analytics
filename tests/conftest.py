"""Shared by all tests: a small test video, and a stand-in for the pool model."""

import numpy as np
import pytest

from swim.utils.video import video_writer


@pytest.fixture
def make_video():
    """make_video(path, frames=10): write a small video (64×48, 30 fps) whose frames get brighter."""

    def make(path, frames=10, width=64, height=48):
        with video_writer(path, 30.0, width, height) as writer:
            for i in range(frames):
                writer.write(np.full((height, width, 3), i * 20 % 256, dtype=np.uint8))
        return path

    return make


class FakeSegmenter:
    """Stands in for the pool model: the lower half of every frame is water."""

    name = "fake-pool"

    def water(self, frame):
        height, width = frame.shape[:2]
        return [[(0, height // 2), (width - 1, height // 2), (width - 1, height - 1), (0, height - 1)]]


@pytest.fixture(autouse=True)
def fake_pool_model(monkeypatch):
    """Every test: the pool stage uses FakeSegmenter, never the real model (it needs torch and a trained model in
    data/models/pool/)."""
    monkeypatch.setattr("swim.stages.pool.stage.create_segmenter", lambda config: FakeSegmenter())
