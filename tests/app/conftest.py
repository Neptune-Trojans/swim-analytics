"""Shared by the app tests: small test videos and a stand-in for the detection model."""

import threading

import numpy as np
import pytest

from swim.stages.detection.types import BBox, Detection
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


class FakeModel:
    """Stands in for the detection model: one box per frame, moving 2 px per frame.

    `pause_at = n` makes detection wait at frame n until `go` is set; `paused` is set when it gets there.
    """

    def __init__(self):
        self.pause_at = None
        self.paused = threading.Event()
        self.go = threading.Event()

    def create_detector(self, config):
        return FakeDetector(self)


class FakeDetector:
    name = "fake"

    def __init__(self, model: FakeModel):
        self.model = model
        self.frames = 0

    def detect(self, frame):
        if self.frames == self.model.pause_at:
            self.model.paused.set()
            self.model.go.wait(timeout=10)
        box = Detection(BBox(2 * self.frames, 10, 2 * self.frames + 20, 30), 0.8)
        self.frames += 1
        return [box]


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr("swim.stages.detection.stage.create_detector", model.create_detector)
    yield model
    model.go.set()  # never leave a background job waiting
