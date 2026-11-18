import numpy as np
import pytest

from swim.pipeline.runner import PipelineResult
from swim.stages.detection.types import BBox, Detection, FrameDetections, VideoDetections
from swim.stages.pool.types import FramePool, VideoPool
from swim.utils.video import read_frames, video_writer
from swim.visualization import render
from swim.visualization.render import render_video

FRAMES, WIDTH, HEIGHT = 3, 64, 48
LOWER_HALF = [[(0, 24), (63, 24), (63, 47), (0, 47)]]


@pytest.fixture
def video(tmp_path):
    """A gray 64×48 video, 3 frames."""
    path = tmp_path / "clip.mp4"
    with video_writer(path, 30.0, WIDTH, HEIGHT) as writer:
        for _ in range(FRAMES):
            writer.write(np.full((HEIGHT, WIDTH, 3), 128, np.uint8))
    return path


def results(pool_frames=FRAMES, width=WIDTH, height=HEIGHT) -> PipelineResult:
    boxes = [FrameDetections(i, i / 30, [Detection(BBox(10, 30, 30, 44), 0.9, track_id=1)]) for i in range(FRAMES)]
    pool = [FramePool(i, i / 30, LOWER_HALF) for i in range(pool_frames)]
    return PipelineResult(detections=VideoDetections("clip.mp4", 30.0, width, height, "test", boxes),
                          pool=VideoPool("clip.mp4", 30.0, width, height, "test", pool),
                          tracks=VideoDetections("clip.mp4", 30.0, width, height, "test", boxes))


def test_draws_the_water_in_every_frame(video, tmp_path):
    render_video(video, tmp_path / "out.mp4", results(), ["pool"])

    frames = list(read_frames(tmp_path / "out.mp4"))
    assert len(frames) == FRAMES
    for frame in frames:
        water, deck = frame[40, 50].astype(int), frame[8, 50].astype(int)  # BGR
        assert water[0] - water[2] > 50  # tinted blue
        assert abs(deck[0] - deck[2]) < 10  # still gray


def test_frames_missing_from_the_results_are_drawn_as_nothing_found(video, tmp_path):
    render_video(video, tmp_path / "out.mp4", results(pool_frames=1), ["pool"])  # water saved for frame 0 only

    first, *_, last = read_frames(tmp_path / "out.mp4")
    assert first[40, 50, 0] - int(first[40, 50, 2]) > 50
    assert abs(int(last[40, 50, 0]) - int(last[40, 50, 2])) < 10


def test_layers_are_drawn_in_a_fixed_order_water_first(video, tmp_path, monkeypatch):
    # Whatever order they are asked for in, the water is drawn first, under the boxes
    drawn = []
    layers = {name: (field, lambda frame, data, name=name: drawn.append(name), empty)
              for name, (field, _, empty) in render.LAYERS.items()}
    monkeypatch.setattr(render, "LAYERS", layers)

    render_video(video, tmp_path / "out.mp4", results(), ["tracks", "pool", "detections"])

    assert drawn == ["pool", "detections", "tracks"] * FRAMES


@pytest.mark.parametrize("layers, result, message", [
    (["water"], results(), "Unknown layer 'water'"),
    (["filtered"], results(), "Layer 'filtered' has no data"),
    (["pool"], results(width=1920, height=1080), "wrong results folder"),  # results of another video
])
def test_wrong_layers_are_refused_before_writing(video, tmp_path, layers, result, message):
    with pytest.raises(ValueError, match=message):
        render_video(video, tmp_path / "out.mp4", result, layers)

    assert not (tmp_path / "out.mp4").exists()
