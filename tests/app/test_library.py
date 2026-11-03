from pathlib import Path

import cv2
import numpy as np
import pytest

from swim.app.library import list_folder, results_status, stored_path, thumbnail
from swim.pipeline.config import PipelineConfig
from swim.pipeline.runner import run_pipeline

DEFAULTS = PipelineConfig()


@pytest.fixture
def folder(tmp_path, make_video):
    """input/ with two videos, a subfolder, a hidden one, a text file and a file that isn't really a video."""
    root = tmp_path / "input"
    make_video(root / "b.MOV")
    make_video(root / "a.mp4", frames=12)
    (root / "z sub").mkdir()
    (root / "A sub").mkdir()
    (root / ".hidden").mkdir()
    (root / "notes.txt").write_text("x")
    (root / "broken.mp4").write_bytes(b"not a video")
    return root


def test_lists_subfolders_and_videos_sorted(folder, tmp_path):
    listing = list_folder(folder, tmp_path / "output", DEFAULTS)

    assert listing["folder"] == str(folder)
    assert listing["parent"] == str(tmp_path)
    assert [f["name"] for f in listing["folders"]] == ["A sub", "z sub"]
    assert [v["file"] for v in listing["videos"]] == ["a.mp4", "b.MOV", "broken.mp4"]
    a = listing["videos"][0]
    assert a["info"] == {"fps": 30.0, "width": 64, "height": 48, "frame_count": 12}
    assert a["results"]["stages"] == [] and a["results"]["start"] == "detection"  # not processed
    assert listing["videos"][2]["info"] is None  # OpenCV can't read it


def test_thumbnail_is_a_small_jpeg_of_the_first_frame(folder):
    image = cv2.imdecode(np.frombuffer(thumbnail(folder / "a.mp4"), np.uint8), cv2.IMREAD_COLOR)

    assert image.shape == (240, 320, 3)  # 320 px wide, same aspect ratio as 64×48
    assert image.mean() < 10  # the first frame is black


@pytest.fixture
def processed(tmp_path, make_video, fake_model):
    """A video processed with the default settings: (video, output folder)."""
    video = make_video(tmp_path / "input" / "IMG_1.mp4")
    output = tmp_path / "output"
    run_pipeline(video, output / "IMG_1")
    return video, output


def test_results_made_with_the_wanted_settings_are_up_to_date(processed):
    video, output = processed

    status = results_status(video, output, DEFAULTS)

    assert status == {"name": "IMG_1", "stages": ["detection", "filtering", "tracking"], "made_from": None,
                      "settings_saved": True, "differences": [], "start": None}


@pytest.mark.parametrize("changes, start", [
    ({"tracking": {"tracker": "sort"}}, "tracking"),
    ({"filtering": {"min_score": 0.3}, "tracking": {"min_frames": 3}}, "filtering"),
    ({"detection": {"threshold": 0.3}}, "detection"),
])
def test_other_settings_start_from_the_first_stage_they_change(processed, changes, start):
    video, output = processed

    status = results_status(video, output, DEFAULTS.with_changes(changes))

    assert status["start"] == start
    stage, values = next(iter(changes.items()))
    name, value = next(iter(values.items()))
    assert {"stage": stage, "name": name, "saved": getattr(getattr(DEFAULTS, stage), name), "wanted": value} in status["differences"]


def test_a_missing_stage_starts_there(processed):
    video, output = processed
    (output / "IMG_1" / "3_tracks.json").unlink()

    assert results_status(video, output, DEFAULTS)["start"] == "tracking"


def test_results_without_settings_start_from_the_beginning(processed):
    video, output = processed
    (output / "IMG_1" / "config.json").unlink()

    status = results_status(video, output, DEFAULTS)

    assert (status["settings_saved"], status["start"]) == (False, "detection")


def test_results_of_another_video_with_the_same_name(processed, tmp_path, make_video):
    _, output = processed
    other = make_video(tmp_path / "elsewhere" / "IMG_1.mp4")

    status = results_status(other, output, DEFAULTS)

    assert status["made_from"].endswith(str(Path("input") / "IMG_1.mp4"))
    assert status["start"] == "detection"


def test_video_path_is_recorded_relative_to_the_repo_when_inside_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert stored_path(tmp_path / "data" / "input" / "x.MOV") == Path("data/input/x.MOV")
    assert stored_path(Path("/elsewhere/x.MOV")) == Path("/elsewhere/x.MOV").resolve()
