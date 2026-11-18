import numpy as np
import pytest

from swim.pipeline.config import PoolConfig
from swim.pipeline.runner import PipelineResult
from swim.stages.detection.types import VideoDetections
from swim.stages.pool.stage import find_model, segment_pool
from swim.stages.pool.types import FramePool, VideoPool
from swim.stages.pool.unet_model import water_polygons
from swim.visualization.pool import draw_pool
from swim.visualization.render import default_layers


def test_water_outline_from_the_model_mask():
    water = np.zeros((100, 200), bool)
    water[40:, :] = True  # the lower part of the frame
    water[60:70, 90:110] = False  # a swimmer inside: a hole, not kept
    water[5:7, 5:7] = True  # a speck: dropped

    [polygon] = water_polygons(water)

    assert sorted(polygon) == [(0, 40), (0, 99), (199, 40), (199, 99)]
    assert water_polygons(np.zeros((100, 200), bool)) == []  # no water found (camera on the deck)


def test_segment_pool_finds_water_in_every_frame(tmp_path, make_video):
    # The pool model is FakeSegmenter (tests/conftest.py): the lower half of every frame
    video = make_video(tmp_path / "clip.mp4", frames=5)
    done = []

    result = segment_pool(video, PoolConfig(), on_frame=lambda n, total: done.append((n, total)))

    assert (result.model, result.width, result.height, len(result.frames)) == ("fake-pool", 64, 48, 5)
    assert result.frames[3].water == [[(0, 24), (63, 24), (63, 47), (0, 47)]]
    assert done == [(n, 5) for n in range(1, 6)]


def test_saved_pool_round_trips(tmp_path):
    pool = VideoPool("v.mp4", 30.0, 64, 48, "fake-pool", [FramePool(0, 0.0, [[(0, 24), (63, 24), (63, 47)]]), FramePool(1, 0.0333)])

    pool.to_json(tmp_path / "2_pool.json")

    assert VideoPool.from_json(tmp_path / "2_pool.json") == pool


def test_latest_model_is_the_newest_training_run(tmp_path):
    for run in ("20261116-223804", "20261116-224516", "20261117-090000"):
        (tmp_path / run).mkdir()
    for run in ("20261116-223804", "20261116-224516"):  # the newest folder has no model.pt (stopped run)
        (tmp_path / run / "model.pt").write_bytes(b"")

    assert find_model("latest", tmp_path) == tmp_path / "20261116-224516" / "model.pt"
    assert find_model("20261116-223804", tmp_path) == tmp_path / "20261116-223804" / "model.pt"
    with pytest.raises(FileNotFoundError, match="No pool model"):
        find_model("20261117-090000", tmp_path)
    with pytest.raises(FileNotFoundError, match="train one"):
        find_model("latest", tmp_path / "empty")


def test_pool_layer_tints_the_water_only():
    frame = np.full((48, 64, 3), 100, np.uint8)

    draw_pool(frame, FramePool(0, 0.0, [[(0, 24), (63, 24), (63, 47), (0, 47)]]))

    assert (frame[10, 30] == 100).all()  # above the water: unchanged
    assert frame[40, 30, 0] > 100 > frame[40, 30, 2]  # in the water: bluer (BGR)
    untouched = np.full((48, 64, 3), 100, np.uint8)
    assert (draw_pool(untouched.copy(), FramePool(0, 0.0)) == untouched).all()  # no water found: nothing drawn


def test_default_layers_are_the_pool_and_the_last_boxes():
    boxes = VideoDetections("v.mp4", 30.0, 64, 48, "fake")
    pool = VideoPool("v.mp4", 30.0, 64, 48, "fake-pool")

    assert default_layers(PipelineResult(detections=boxes, pool=pool, filtered=boxes, tracks=boxes)) == ["pool", "tracks"]
    assert default_layers(PipelineResult(detections=boxes)) == ["detections"]


def test_unet_segmenter_loads_a_training_run_and_outlines_its_water(tmp_path):
    import torch

    from swim.pool_model.network import UNet
    from swim.stages.pool.unet_model import UNetSegmenter

    path = tmp_path / "20261117-090000" / "model.pt"  # as `python -m swim.pool_model train` saves it (untrained here)
    path.parent.mkdir()
    torch.save({"network": UNet(2, pretrained=False).state_dict(), "classes": ["other", "water"], "encoder": "resnet18",
                "size": [128, 64]}, path)

    segmenter = UNetSegmenter(path, device="cpu")
    water = segmenter.water(np.full((90, 160, 3), 120, np.uint8))

    assert segmenter.name == "pool-unet 20261117-090000"
    assert all(len(polygon) >= 3 and all(0 <= x < 160 and 0 <= y < 90 for x, y in polygon) for polygon in water)
