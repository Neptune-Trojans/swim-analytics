"""Processing from the Videos page: POST /api/jobs queues videos, a background thread runs them through
run_pipeline() one at a time (with a stand-in for the detection model, see conftest.py)."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from swim.app.server import create_app

FRAMES = 10


@pytest.fixture
def setup(tmp_path, monkeypatch, make_video, fake_model):
    """Repo-like folder (the current folder) with data/input/IMG_1.mp4 and IMG_2.mp4; returns (client, root)."""
    monkeypatch.chdir(tmp_path)
    for name in ("IMG_1", "IMG_2"):
        make_video(tmp_path / "data" / "input" / f"{name}.mp4", frames=FRAMES)
    app = create_app(tmp_path / "data" / "output", tmp_path / "data" / "input")
    client = TestClient(app, base_url="http://127.0.0.1")
    client.app_jobs = app.state.jobs
    return client, tmp_path


def video(root, name="IMG_1"):
    return str(root / "data" / "input" / f"{name}.mp4")


def process(client, *videos, **options):
    return client.post("/api/jobs", json={"videos": list(videos), **options})


def finish(client):
    assert client.app_jobs.wait(timeout=20), "jobs did not finish"
    return client.get("/api/jobs").json()


def statuses(client, settings=None):
    params = {"settings": json.dumps(settings)} if settings else {}
    return {v["file"]: v["results"] for v in client.get("/api/library", params=params).json()["videos"]}


def wait_until(condition, timeout=10):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


def test_processing_a_new_video_saves_every_stage(setup):
    client, root = setup
    assert statuses(client)["IMG_1.mp4"]["stages"] == []

    assert process(client, video(root)).status_code == 200
    [job] = finish(client)

    assert (job["name"], job["state"], job["start"], job["error"]) == ("IMG_1", "done", "detection", None)
    assert (job["done"], job["total"]) == (FRAMES, FRAMES)
    folder = root / "data" / "output" / "IMG_1"
    assert sorted(p.name for p in folder.iterdir()) == ["1_detections.json", "2_filtered.json", "3_tracks.json", "config.json"]
    assert json.loads((folder / "1_detections.json").read_text())["video_path"] == "data/input/IMG_1.mp4"  # as swim.run
    assert statuses(client)["IMG_1.mp4"]["start"] is None  # up to date
    assert client.get("/api/videos").json() == [{"name": "IMG_1", "stages": ["detection", "filtering", "tracking"],
                                                 "video_found": True}]


def test_videos_run_one_after_the_other(setup, fake_model):
    client, root = setup
    fake_model.pause_at = 3

    process(client, video(root, "IMG_1"), video(root, "IMG_2"))
    assert fake_model.paused.wait(5)
    running, waiting = client.get("/api/jobs").json()

    assert (running["state"], running["stage"], running["done"]) == ("running", "detection", 3)
    assert (waiting["state"], waiting["position"]) == ("queued", 1)
    fake_model.go.set()
    assert [j["state"] for j in finish(client)] == ["done", "done"]


def test_update_runs_only_the_stages_that_need_it(setup):
    client, root = setup
    process(client, video(root))
    finish(client)
    detections = root / "data" / "output" / "IMG_1" / "1_detections.json"
    before = detections.stat().st_mtime_ns

    process(client, video(root), settings={"tracking": {"tracker": "sort"}})
    job = finish(client)[-1]

    assert (job["state"], job["start"]) == ("done", "tracking")
    assert detections.stat().st_mtime_ns == before  # detection not re-run
    config = json.loads((root / "data" / "output" / "IMG_1" / "config.json").read_text())
    assert config["tracking"]["tracker"] == "sort"
    assert statuses(client, {"tracking": {"tracker": "sort"}})["IMG_1.mp4"]["start"] is None
    assert statuses(client)["IMG_1.mp4"]["start"] == "tracking"  # compared with the defaults: other tracker


def test_up_to_date_video_is_processed_again_from_the_start(setup):
    client, root = setup
    process(client, video(root))
    finish(client)

    process(client, video(root))

    assert finish(client)[-1]["start"] == "detection"


def test_cancel_a_running_job_keeps_the_old_results(setup, fake_model):
    client, root = setup
    fake_model.pause_at = 4
    process(client, video(root))
    assert fake_model.paused.wait(5)
    job_id = client.get("/api/jobs").json()[0]["id"]

    jobs = client.post(f"/api/jobs/{job_id}/cancel").json()
    assert jobs[0]["cancel_requested"]
    fake_model.go.set()
    [job] = finish(client)

    assert job["state"] == "cancelled"
    assert not (root / "data" / "output" / "IMG_1" / "1_detections.json").exists()
    assert statuses(client)["IMG_1.mp4"]["stages"] == []


def test_cancel_a_waiting_job(setup, fake_model):
    client, root = setup
    fake_model.pause_at = 2
    process(client, video(root, "IMG_1"), video(root, "IMG_2"))
    assert fake_model.paused.wait(5)
    waiting = client.get("/api/jobs").json()[1]

    client.post(f"/api/jobs/{waiting['id']}/cancel")
    fake_model.go.set()

    assert [j["state"] for j in finish(client)] == ["done", "cancelled"]
    assert not (root / "data" / "output" / "IMG_2").exists()


def test_a_video_being_processed_cannot_be_queued_or_rerun(setup, fake_model):
    client, root = setup
    process(client, video(root))
    finish(client)
    fake_model.pause_at = 2
    process(client, video(root))
    assert fake_model.paused.wait(5)

    assert process(client, video(root)).status_code == 409
    rerun = client.post("/api/videos/IMG_1/rerun", json={"settings": {"filtering": {"min_score": 0.3}}})
    assert rerun.status_code == 409
    fake_model.go.set()
    finish(client)


def test_results_of_another_video_are_replaced_only_when_asked(setup, make_video):
    client, root = setup
    process(client, video(root))
    finish(client)
    other = make_video(root / "elsewhere" / "IMG_1.mp4", frames=FRAMES)

    refused = process(client, str(other))
    assert refused.status_code == 409 and "made from" in refused.json()["detail"]

    assert process(client, str(other), replace=True).status_code == 200
    assert finish(client)[-1]["state"] == "done"
    detections = json.loads((root / "data" / "output" / "IMG_1" / "1_detections.json").read_text())
    assert detections["video_path"] == "elsewhere/IMG_1.mp4"


def test_a_video_opencv_cannot_read_fails_with_its_reason(setup):
    client, root = setup
    broken = root / "data" / "input" / "broken.mp4"
    broken.write_bytes(b"not a video")

    process(client, str(broken))
    [job] = finish(client)

    assert job["state"] == "failed"
    assert "Cannot open video" in job["error"]


@pytest.mark.parametrize("body, status", [
    ({"videos": ["data/input/IMG_1.mp4"]}, 400),  # not absolute
    ({"videos": ["/nowhere/IMG_1.mp4"]}, 400),
    ({"videos": ["{root}/data/input/IMG_1.mp4"], "settings": {"detection": {"threshold": 2}}}, 400),
    ({"videos": ["{root}/data/input/IMG_1.mp4", "{root}/data/input/IMG_1.mp4"]}, 400),  # same results folder
])
def test_bad_process_requests(setup, body, status):
    client, root = setup
    body["videos"] = [v.format(root=root) for v in body["videos"]]

    assert client.post("/api/jobs", json=body).status_code == status
    assert client.get("/api/jobs").json() == []


def test_cancel_unknown_job(setup):
    client, _ = setup
    assert client.post("/api/jobs/99/cancel").status_code == 404
