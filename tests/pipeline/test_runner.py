import numpy as np
import pytest

from swim.pipeline.config import PipelineConfig
from swim.pipeline.runner import load_results, run_pipeline
from swim.stages.detection.types import BBox, Detection, FrameDetections, VideoDetections
from swim.utils.video import video_writer

FRAMES = 10


@pytest.fixture
def saved_detections(tmp_path):
    """A small video and saved detections of one moving swimmer, as if detection had run: (video, results folder)."""
    video = tmp_path / "input" / "clip.mp4"
    with video_writer(video, 30.0, 64, 48) as writer:
        for _ in range(FRAMES):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    folder = tmp_path / "output" / "clip"
    frames = [FrameDetections(i, i / 30, [Detection(BBox(2 * i, 10, 2 * i + 20, 30), 0.9)]) for i in range(FRAMES)]
    VideoDetections(str(video), 30.0, 64, 48, "test", frames).to_json(folder / "1_detections.json")
    PipelineConfig().with_changes({"detection": {"threshold": 0.3}}).to_json(folder / "config.json")
    return video, folder


def test_run_from_filtering_runs_filtering_and_tracking(saved_detections):
    video, folder = saved_detections

    result = run_pipeline(video, folder, PipelineConfig().with_changes({"tracking": {"tracker": "sort"}}), start="filtering")

    assert (folder / "2_filtered.json").exists() and (folder / "3_tracks.json").exists()
    assert [[d.track_id for d in f.detections] for f in result.tracks.frames] == [[1]] * FRAMES
    assert load_results(folder) == result


def test_settings_of_loaded_stages_stay_as_saved(saved_detections):
    video, folder = saved_detections

    run_pipeline(video, folder, PipelineConfig().with_changes({"tracking": {"tracker": "sort"}}), start="filtering")

    saved = PipelineConfig.from_json(folder / "config.json")
    assert saved.detection.threshold == 0.3  # detection was loaded, not run: its saved settings are kept
    assert saved.tracking.tracker == "sort"  # tracking ran with the new settings


def test_stopping_earlier_removes_outdated_results(saved_detections):
    video, folder = saved_detections
    run_pipeline(video, folder, start="filtering")

    run_pipeline(video, folder, start="filtering", stop="filtering")

    assert (folder / "2_filtered.json").exists()
    assert not (folder / "3_tracks.json").exists()  # made from the old filtering results


def test_starting_after_a_missing_stage_fails(saved_detections):
    video, folder = saved_detections

    with pytest.raises(FileNotFoundError, match="filtering"):
        run_pipeline(video, folder, start="tracking")


class FakeDetector:
    """Stands in for the model: one box per frame, moving 2 px per frame."""

    name = "fake"

    def __init__(self):
        self.frames = 0

    def detect(self, frame):
        box = Detection(BBox(2 * self.frames, 10, 2 * self.frames + 20, 30), 0.8)
        self.frames += 1
        return [box]


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr("swim.stages.detection.stage.create_detector", lambda config: FakeDetector())


class Cancel(Exception):
    pass


def test_progress_reports_each_stage_and_detection_frames(saved_detections, fake_model):
    video, folder = saved_detections
    calls = []

    run_pipeline(video, folder, progress=lambda stage, done, total: calls.append((stage, done, total)))

    assert calls == (
        [("detection", 0, FRAMES)] + [("detection", i, FRAMES) for i in range(1, FRAMES + 1)] + [("detection", FRAMES, FRAMES)]
        + [("filtering", 0, FRAMES), ("filtering", FRAMES, FRAMES), ("tracking", 0, FRAMES), ("tracking", FRAMES, FRAMES)]
    )


def test_cancelled_detection_leaves_the_saved_results_as_they_were(saved_detections, fake_model):
    video, folder = saved_detections
    run_pipeline(video, folder, start="filtering")
    before = {path.name: path.read_text() for path in folder.iterdir()}

    def cancel_at_frame_5(stage, done, total):
        if stage == "detection" and done == 5:
            raise Cancel

    with pytest.raises(Cancel):
        run_pipeline(video, folder, PipelineConfig().with_changes({"detection": {"threshold": 0.6}}), progress=cancel_at_frame_5)

    assert {path.name: path.read_text() for path in folder.iterdir()} == before  # config.json too: still threshold 0.3


def test_stopped_after_detection_keeps_new_detections_and_drops_what_was_made_from_the_old_ones(saved_detections, fake_model):
    video, folder = saved_detections
    run_pipeline(video, folder, start="filtering")

    def cancel_in_filtering(stage, done, total):
        if stage == "filtering":
            raise Cancel

    with pytest.raises(Cancel):
        run_pipeline(video, folder, PipelineConfig().with_changes({"detection": {"threshold": 0.6}}), progress=cancel_in_filtering)

    assert sorted(path.name for path in folder.iterdir()) == ["1_detections.json", "config.json"]
    assert load_results(folder).detections.model == "fake"  # the new detections
    assert PipelineConfig.from_json(folder / "config.json").detection.threshold == 0.6  # recorded with them
