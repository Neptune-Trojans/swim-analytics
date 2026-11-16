"""Labels page API: list the picked frames, their labels and image, save checked labels."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from swim.app.server import create_app
from swim.labeling.dataset import Dataset, Frame
from swim.labeling.labels import FrameLabels, Rope

WATER = [[[0, 20], [99, 20], [99, 59], [0, 59]]]
ROPES = [{"points": [[0, 40], [99, 35]], "widths": [4, 2]}]


@pytest.fixture
def setup(tmp_path):
    """Dataset with 2 frames: IMG_1_000010 labeled by the model, IMG_1_000040 not labeled yet."""
    dataset = Dataset(tmp_path / "labels")
    frames = [Frame(f"IMG_1_{i:06d}", "data/input/IMG_1.MOV", i, 100, 60) for i in (10, 40)]
    dataset.add_frames(frames)
    dataset.images.mkdir(parents=True)
    for f in frames:
        cv2.imwrite(str(dataset.image_path(f.name)), np.zeros((60, 100, 3), np.uint8))
    dataset.save_sam3(FrameLabels(video=frames[0].video, frame_idx=10, image=frames[0].image, width=100, height=60,
                                  water=[[(0, 20), (99, 20), (99, 59), (0, 59)]], ropes=[Rope([(0, 40), (99, 35)], [4, 2])],
                                  source="sam3-litetext-s0"))
    client = TestClient(create_app(tmp_path / "output", tmp_path / "input", dataset.root), base_url="http://127.0.0.1")
    return client, dataset


def test_lists_frames_with_their_status(setup):
    client, _ = setup

    listing = client.get("/api/labels").json()

    assert [(f["name"], f["status"], f["water"], f["ropes"]) for f in listing["frames"]] == [
        ("IMG_1_000010", "sam3", 1, 1), ("IMG_1_000040", "no labels", 0, 0)]
    assert listing["counts"] == {"no labels": 1, "sam3": 1, "fixed": 0, "accepted": 0}
    assert listing["classes"] == {"0": "other", "1": "water", "2": "lane rope"}


def test_frame_labels_and_image(setup):
    client, _ = setup

    data = client.get("/api/labels/IMG_1_000010").json()

    assert data["frame"] == {"name": "IMG_1_000010", "video": "data/input/IMG_1.MOV", "frame_idx": 10, "width": 100, "height": 60}
    assert data["labels"]["water"] == WATER and data["labels"]["ropes"] == ROPES
    assert data["sam3"]["ropes"] == ROPES
    unlabeled = client.get("/api/labels/IMG_1_000040").json()
    assert unlabeled["labels"]["water"] == [] and unlabeled["sam3"] is None
    image = client.get("/api/labels/IMG_1_000010/image")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"


def test_save_fixed_labels(setup):
    client, dataset = setup
    ropes = [{"points": [[0, 41], [99, 33]], "widths": [5, 2]}]

    response = client.put("/api/labels/IMG_1_000010", json={"status": "fixed", "water": WATER, "ropes": ropes})

    assert response.status_code == 200
    assert response.json()["counts"]["fixed"] == 1
    saved = dataset.labels("IMG_1_000010")
    assert (saved.status, saved.ropes, saved.source) == ("fixed", [Rope([(0, 41), (99, 33)], [5, 2])], "sam3-litetext-s0")
    assert dataset.sam3("IMG_1_000010").ropes == [Rope([(0, 40), (99, 35)], [4, 2])]  # the model's labels stay as they were


def test_a_frame_without_model_labels_can_be_labeled_by_hand(setup):
    client, dataset = setup

    assert client.put("/api/labels/IMG_1_000040", json={"status": "fixed", "water": WATER, "ropes": []}).status_code == 200
    assert dataset.labels("IMG_1_000040").water == [[(0, 20), (99, 20), (99, 59), (0, 59)]]


@pytest.mark.parametrize("body", [
    {"status": "done", "water": WATER, "ropes": []},
    {"status": "fixed", "water": [[[0, 0], [5, 5]]], "ropes": []},
    {"status": "fixed", "water": WATER, "ropes": [{"points": [[0, 0]], "widths": [3]}]},
    {"status": "fixed", "water": WATER, "ropes": [{"points": [[0, 0], [9, 9]], "widths": [3, -1]}]},
])
def test_bad_labels_are_refused_and_nothing_changes(setup, body):
    client, dataset = setup
    before = dataset.labels("IMG_1_000010")

    assert client.put("/api/labels/IMG_1_000010", json=body).status_code == 400
    assert dataset.labels("IMG_1_000010") == before


@pytest.mark.parametrize("request_", [("get", "/api/labels/nope"), ("get", "/api/labels/nope/image"),
                                      ("get", "/api/labels/..%2Fframes/image")])
def test_unknown_frames(setup, request_):
    client, _ = setup
    method, url = request_
    assert getattr(client, method)(url).status_code == 404


def test_empty_dataset_lists_nothing(tmp_path):
    client = TestClient(create_app(tmp_path / "o", tmp_path / "i", tmp_path / "none"), base_url="http://127.0.0.1")
    assert client.get("/api/labels").json()["frames"] == []
    assert "labels.js" in client.get("/labels").text


def test_water_edit_merges_and_cuts_without_saving(setup):
    client, dataset = setup
    notch = [[0, 20], [99, 20], [99, 59], [0, 59], [0, 45], [20, 45], [20, 35], [0, 35]]

    merged = client.post("/api/labels/IMG_1_000010/water", json={"water": [notch], "add": [[0, 30], [30, 30], [30, 50], [0, 50]]})
    assert merged.status_code == 200
    assert sorted(map(tuple, merged.json()["water"][0])) == [(0, 20), (0, 59), (99, 20), (99, 59)]

    cut = client.post("/api/labels/IMG_1_000010/water", json={"water": WATER, "subtract": [[60, 0], [100, 0], [100, 40], [60, 40]]})
    assert max(y for x, y in cut.json()["water"][0] if x >= 70) == 59
    assert dataset.labels("IMG_1_000010").water == [[(0, 20), (99, 20), (99, 59), (0, 59)]]  # nothing saved


def test_water_edit_refuses_bad_polygons(setup):
    client, _ = setup
    assert client.post("/api/labels/IMG_1_000010/water", json={"water": WATER, "add": [[0, 0], [5, 5]]}).status_code == 400
    assert client.post("/api/labels/nope/water", json={"water": WATER}).status_code == 404
