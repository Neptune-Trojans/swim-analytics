"""Shared by the app tests: a stand-in for the detection model (make_video is in tests/conftest.py)."""

import threading

import pytest

from swim.stages.detection.types import BBox, Detection


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
