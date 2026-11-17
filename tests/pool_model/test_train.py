import importlib.util
import json

import cv2
import pytest

from swim.pool_model.predict import PoolModel
from swim.pool_model.train import train


def test_a_short_training_run_saves_a_model_that_predicts(dataset, tmp_path):
    run = train(dataset.root, tmp_path / "models", epochs=2, batch=2, size=(128, 64), pretrained=False, device="cpu",
                workers=0, log=lambda _: None)

    summary = json.loads((run / "summary.json").read_text())
    assert summary["classes"] == ["other", "water"]
    assert sorted(summary["val_frames"]) == ["IMG_1_000180", "IMG_2_000060"]  # one per video
    assert set(summary["val_iou"]) == set(summary["val_frames"]) and 0 <= summary["val_iou_min"] <= 1
    assert (run / "log.csv").read_text().count("\n") == 3  # header + 2 epochs
    assert sorted(p.name for p in (run / "val").iterdir()) == ["IMG_1_000180.jpg", "IMG_2_000060.jpg"]
    if importlib.util.find_spec("tensorboard"):
        assert list(run.glob("events.out.tfevents.*"))  # progress for TensorBoard

    model = PoolModel.load(run / "model.pt", device="cpu")
    frame = cv2.imread(str(dataset.image_path("IMG_1_000000")))
    classes = model.predict(frame)
    assert classes.shape == (144, 256) and set(classes.ravel().tolist()) <= {0, 1}
    assert model.predict(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)).shape == (256, 144)  # an upright frame


def test_training_needs_checked_frames(tmp_path, make_dataset):
    dataset = make_dataset(tmp_path / "pool", {"IMG_1": 3}, status="sam3")
    with pytest.raises(ValueError, match="0 checked frames"):
        train(dataset.root, tmp_path / "models", epochs=1, pretrained=False, device="cpu", workers=0)


def test_input_size_must_suit_the_network(dataset, tmp_path):
    with pytest.raises(ValueError, match="divisible by 32"):
        train(dataset.root, tmp_path / "models", size=(1000, 576), pretrained=False, device="cpu", workers=0)
