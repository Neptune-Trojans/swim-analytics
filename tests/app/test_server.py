import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from swim.app.server import create_app
from swim.pipeline.config import PipelineConfig
from swim.stages.detection.types import BBox, Detection, FrameDetections, VideoDetections
from swim.utils.video import video_writer


def write_video(path, frames=10, width=64, height=48):
    with video_writer(path, 30.0, width, height) as writer:
        for i in range(frames):
            writer.write(np.full((height, width, 3), i * 20, dtype=np.uint8))


def write_results(folder, video, stages=("1_detections.json",)):
    dets = VideoDetections(
        video_path=str(video), fps=30.0, width=64, height=48, model="rfdetr-medium",
        frames=[FrameDetections(i, i / 30, [Detection(BBox(1, 2, 30, 40), 0.9)]) for i in range(10)],
    )
    for filename in stages:
        dets.to_json(folder / filename)
    PipelineConfig().to_json(folder / "config.json")


@pytest.fixture
def output(tmp_path):
    """Results root with: IMG_1 (detection + filtering), IMG_2 (video missing), IMG_3 (detection only),
    empty (no results)."""
    root = tmp_path / "output"
    video = tmp_path / "input" / "IMG_1.mp4"
    write_video(video)
    write_results(root / "IMG_1", video, ("1_detections.json", "2_filtered.json"))
    write_results(root / "IMG_2", tmp_path / "input" / "gone.MOV")
    write_results(root / "IMG_3", video)
    (root / "empty").mkdir()
    return root


@pytest.fixture
def client(output):
    return TestClient(create_app(output))


def test_lists_folders_with_results(client):
    assert client.get("/api/videos").json() == [
        {"name": "IMG_1", "stages": ["detection", "filtering"], "video_found": True},
        {"name": "IMG_2", "stages": ["detection"], "video_found": False},
        {"name": "IMG_3", "stages": ["detection"], "video_found": True},
    ]


def test_missing_output_folder_lists_nothing(tmp_path):
    assert TestClient(create_app(tmp_path / "nothing")).get("/api/videos").json() == []


def test_details_have_settings_and_video_info(client):
    details = client.get("/api/videos/IMG_1").json()

    assert details["stages"] == ["detection", "filtering"]
    assert details["editable_stages"] == ["filtering"]
    min_score = details["settings"]["filtering"][0]
    assert (min_score["name"], min_score["value"], min_score["min"], min_score["max"]) == ("min_score", 0.1, 0, 1)
    assert min_score["help"]
    assert details["video"]["frame_count"] == 10
    assert (details["video"]["width"], details["video"]["height"]) == (64, 48)


def test_details_without_video(client):
    assert client.get("/api/videos/IMG_2").json()["video"] is None
    assert client.get("/api/videos/IMG_2/video").status_code == 404


def test_serves_stage_results(client, output):
    response = client.get("/api/videos/IMG_1/results/filtering")

    assert response.status_code == 200
    assert response.json() == json.loads((output / "IMG_1" / "2_filtered.json").read_text())


@pytest.mark.parametrize("url", [
    "/api/videos/IMG_2/results/filtering",  # stage not run
    "/api/videos/IMG_1/results/config",  # not a stage
    "/api/videos/unknown",
    "/api/videos/empty",  # folder without results
    "/api/videos/empty/video",
])
def test_unknown_names_are_not_found(client, url):
    assert client.get(url).status_code == 404


def test_video_supports_range_requests(client, output):
    # Browsers need range requests (206 Partial Content) to seek in a video
    full = client.get("/api/videos/IMG_1/video")
    part = client.get("/api/videos/IMG_1/video", headers={"Range": "bytes=0-99"})

    assert full.status_code == 200
    assert full.headers["content-type"] == "video/mp4"
    assert part.status_code == 206
    assert part.content == full.content[:100]


def test_serves_the_page(client):
    assert "<html" in client.get("/").text
    assert client.get("/static/viewer.js").status_code == 200
    assert client.get("/static/layers.js").status_code == 200


def test_rerun_with_changed_settings_saves_new_results(client, output):
    response = client.post("/api/videos/IMG_1/rerun", json={"settings": {"filtering": {"min_score": 0.95}}})

    assert response.status_code == 200
    assert response.json()["settings"]["filtering"][0]["value"] == 0.95
    filtered = json.loads((output / "IMG_1" / "2_filtered.json").read_text())
    assert all(f["detections"] == [] for f in filtered["frames"])  # every box scores 0.9 < 0.95
    assert {r["reason"] for f in filtered["frames"] for r in f["rejected"]} == {"min_score"}
    config = json.loads((output / "IMG_1" / "config.json").read_text())
    assert config["filtering"]["min_score"] == 0.95
    assert config["detection"] == {"model": "rfdetr", "size": "medium", "threshold": 0.1}  # unchanged


def test_rerun_creates_results_of_stages_not_run_yet(client, output):
    response = client.post("/api/videos/IMG_3/rerun", json={"settings": {"filtering": {"max_iou": 0.6}}})

    assert response.status_code == 200
    assert response.json()["stages"] == ["detection", "filtering"]
    assert (output / "IMG_3" / "2_filtered.json").exists()


@pytest.mark.parametrize("url, settings, status", [
    ("/api/videos/IMG_1/rerun", {"detection": {"threshold": 0.3}}, 400),  # detection: command line only
    ("/api/videos/IMG_1/rerun", {"filtering": {"min_score": 2}}, 400),
    ("/api/videos/IMG_1/rerun", {"filtering": {"nope": 1}}, 400),
    ("/api/videos/IMG_2/rerun", {"filtering": {"min_score": 0.3}}, 400),  # video missing
    ("/api/videos/unknown/rerun", {"filtering": {"min_score": 0.3}}, 404),
])
def test_rerun_rejects_bad_requests(client, output, url, settings, status):
    before = (output / "IMG_1" / "config.json").read_text()

    assert client.post(url, json={"settings": settings}).status_code == status
    assert (output / "IMG_1" / "config.json").read_text() == before
