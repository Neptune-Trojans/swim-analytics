"""Shared by all tests."""

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
