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
    return TestClient(create_app(output), base_url="http://127.0.0.1")


def test_lists_folders_with_results(client):
    assert client.get("/api/videos").json() == [
        {"name": "IMG_1", "stages": ["detection", "filtering"], "video_found": True},
        {"name": "IMG_2", "stages": ["detection"], "video_found": False},
        {"name": "IMG_3", "stages": ["detection"], "video_found": True},
    ]


def test_missing_output_folder_lists_nothing(tmp_path):
    assert TestClient(create_app(tmp_path / "nothing"), base_url="http://127.0.0.1").get("/api/videos").json() == []


def test_details_have_settings_and_video_info(client):
    details = client.get("/api/videos/IMG_1").json()

    assert details["stages"] == ["detection", "filtering"]
    assert details["editable_stages"] == ["filtering", "tracking"]
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


def test_serves_the_pages(client):
    assert "videos.js" in client.get("/").text  # Videos page
    assert "viewer.js" in client.get("/play?video=IMG_1").text  # Play page
    for script in ("videos.js", "viewer.js", "layers.js", "settings.js", "dom.js"):
        assert client.get(f"/static/{script}").status_code == 200


def test_old_viewer_links_go_to_the_play_page(client):
    response = client.get("/?video=IMG_1&frame=5", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/play?video=IMG_1&frame=5"


def test_only_requests_addressed_to_this_computer(output):
    # A page on another site pointing its domain name at 127.0.0.1 (DNS rebinding) is refused
    assert TestClient(create_app(output), base_url="http://evil.example").get("/api/videos").status_code == 400
    assert TestClient(create_app(output), base_url="http://localhost:8000").get("/api/videos").status_code == 200


def test_library_opens_the_input_folder_by_default(tmp_path, output):
    client = TestClient(create_app(output, tmp_path / "input"), base_url="http://127.0.0.1")

    listing = client.get("/api/library").json()

    assert listing["folder"] == str((tmp_path / "input").resolve())
    assert [v["file"] for v in listing["videos"]] == ["IMG_1.mp4"]
    assert listing["stages"] == ["detection", "filtering", "tracking"]
    assert listing["output"] == str(output.resolve())
    # output/IMG_1 has detection and filtering made from this video: tracking is missing
    assert listing["videos"][0]["results"]["start"] == "tracking"


@pytest.mark.parametrize("params, status", [
    ({"folder": "/nowhere/at/all"}, 404),
    ({"settings": "not json"}, 400),
    ({"settings": '{"detection": {"threshold": 5}}'}, 400),
])
def test_library_bad_requests(client, params, status):
    assert client.get("/api/library", params=params).status_code == status


def test_default_settings_have_explanations(client):
    settings = client.get("/api/settings").json()

    assert list(settings) == ["detection", "filtering", "tracking"]
    assert settings["detection"][2]["name"] == "threshold" and settings["detection"][2]["help"]


def test_thumbnail(client, tmp_path):
    response = client.get("/api/thumbnail", params={"path": str(tmp_path / "input" / "IMG_1.mp4")})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


@pytest.mark.parametrize("path", ["input/IMG_1.mp4", "/nowhere/IMG_1.mp4", "{tmp}/output/IMG_1/config.json"])
def test_thumbnail_only_of_videos(client, tmp_path, path):
    assert client.get("/api/thumbnail", params={"path": path.format(tmp=tmp_path)}).status_code == 404


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
    assert response.json()["stages"] == ["detection", "filtering", "tracking"]
    assert (output / "IMG_3" / "2_filtered.json").exists() and (output / "IMG_3" / "3_tracks.json").exists()


def test_rerun_with_tracking_settings(client, output):
    settings = {"tracking": {"tracker": "ocsort", "min_frames": 3}}

    response = client.post("/api/videos/IMG_1/rerun", json={"settings": settings})

    assert response.status_code == 200
    config = json.loads((output / "IMG_1" / "config.json").read_text())
    assert (config["tracking"]["tracker"], config["tracking"]["min_frames"]) == ("ocsort", 3)
    tracks = json.loads((output / "IMG_1" / "3_tracks.json").read_text())
    assert {d["track_id"] for f in tracks["frames"] for d in f["detections"]} == {1}  # one box per frame: one track


@pytest.mark.parametrize("url, settings, status", [
    ("/api/videos/IMG_1/rerun", {"detection": {"threshold": 0.3}}, 400),  # detection: command line only
    ("/api/videos/IMG_1/rerun", {"filtering": {"min_score": 2}}, 400),
    ("/api/videos/IMG_1/rerun", {"filtering": {"nope": 1}}, 400),
    ("/api/videos/IMG_1/rerun", {"tracking": {"tracker": "deepsort"}}, 400),
    ("/api/videos/IMG_2/rerun", {"filtering": {"min_score": 0.3}}, 400),  # video missing
    ("/api/videos/unknown/rerun", {"filtering": {"min_score": 0.3}}, 404),
])
def test_rerun_rejects_bad_requests(client, output, url, settings, status):
    before = (output / "IMG_1" / "config.json").read_text()

    assert client.post(url, json={"settings": settings}).status_code == status
    assert (output / "IMG_1" / "config.json").read_text() == before
