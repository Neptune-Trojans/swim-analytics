import json
import sys

import numpy as np
import pytest

from swim.pipeline.runner import load_results
from swim.run import main
from swim.stages.detection.types import BBox, Detection, FrameDetections, VideoDetections
from swim.stages.pool.types import FramePool, VideoPool
from swim.utils.video import read_frames, video_writer

FRAMES = 5


@pytest.fixture
def saved(tmp_path):
    """input/clip.mp4 and its saved detections and water in output/clip/, as if detection and pool had run. The
    person's box is above the water (on the deck)."""
    video = tmp_path / "input" / "clip.mp4"
    with video_writer(video, 30.0, 64, 48) as writer:
        for _ in range(FRAMES):
            writer.write(np.full((48, 64, 3), 128, np.uint8))
    folder = tmp_path / "output" / "clip"
    frames = [FrameDetections(i, i / 30, [Detection(BBox(10, 2, 30, 18), 0.9)]) for i in range(FRAMES)]
    VideoDetections(str(video), 30.0, 64, 48, "test", frames).to_json(folder / "1_detections.json")
    water = [[(0, 24), (63, 24), (63, 47), (0, 47)]]
    VideoPool(str(video), 30.0, 64, 48, "test", [FramePool(i, i / 30, water) for i in range(FRAMES)]).to_json(
        folder / "2_pool.json")
    return video, folder


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["swim.run", *map(str, args)])
    main()


def test_runs_the_stages_and_renders_the_water_and_the_last_boxes(saved, monkeypatch, capsys):
    video, folder = saved

    run(monkeypatch, "--input", video, "--output", folder.parent, "--from", "filtering")

    assert [[r.reason for r in f.rejected] for f in load_results(folder).filtered.frames] == [["outside_pool"]] * FRAMES
    assert "[video] layers pool, tracks" in capsys.readouterr().out
    assert len(list(read_frames(folder / "video.mp4"))) == FRAMES


def test_setting_options_change_the_settings(saved, monkeypatch):
    video, folder = saved

    run(monkeypatch, "--input", video, "--output", folder.parent, "--from", "pool",
        "--pool-model", "20261116-224516", "--filter-pool-margin", "1")

    config = json.loads((folder / "config.json").read_text())
    assert config["pool"]["model"] == "20261116-224516"
    assert config["filtering"]["pool_margin"] == 1.0
    assert all(f.detections and not f.rejected for f in load_results(folder).filtered.frames)  # the pool filter is off


def test_a_setting_out_of_range_stops_with_a_message(saved, monkeypatch, capsys):
    video, folder = saved

    with pytest.raises(SystemExit) as stop:
        run(monkeypatch, "--input", video, "--output", folder.parent, "--from", "filtering", "--filter-pool-margin", "2")

    assert stop.value.code == 2
    assert "pool_margin" in capsys.readouterr().err
    assert not (folder / "3_filtered.json").exists()


def test_render_only_draws_saved_results_with_the_chosen_layers(saved, monkeypatch, capsys):
    video, folder = saved

    run(monkeypatch, "--input", video, "--output", folder.parent, "--render-only", "--layers", "pool,detections")

    assert "[video] layers pool, detections" in capsys.readouterr().out
    assert sorted(path.name for path in folder.iterdir()) == ["1_detections.json", "2_pool.json", "video.mp4"]
