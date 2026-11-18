import argparse
import json
import sys

import pytest

from swim.pool_model.__main__ import main, size


@pytest.fixture
def fake_train(tmp_path, monkeypatch):
    """Stands in for train() (no training): records its arguments and writes a run folder with a summary."""
    calls = []

    def train(dataset, out, **options):
        calls.append((dataset, out, options))
        if dataset == "empty":
            raise ValueError("0 checked frames in empty")
        run = tmp_path / "20261118-120000"
        run.mkdir()
        (run / "summary.json").write_text(json.dumps({
            "seconds": 12.5, "val_iou": {"IMG_1_000010": 0.95, "IMG_2_000020": 0.99},
            "val_iou_mean": 0.97, "val_iou_min": 0.95}))
        return run

    monkeypatch.setattr("swim.pool_model.train.train", train)
    return calls


def test_train_passes_its_options_and_prints_the_result(fake_train, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["swim.pool_model", "train", "--dataset", "labels", "--out", "models",
                                      "--epochs", "3", "--size", "512x288", "--device", "cpu", "--workers", "0"])

    main()

    assert fake_train == [("labels", "models", {"epochs": 3, "batch": 8, "size": (512, 288), "lr": 1e-3,
                                                "encoder": "resnet18", "val_every": 6, "device": "cpu", "workers": 0,
                                                "seed": 0})]
    out = capsys.readouterr().out
    assert "Validation water IoU: mean 0.970, worst 0.950" in out
    assert "IMG_1_000010 0.950" in out


def test_train_without_checked_frames_stops_with_the_reason(fake_train, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["swim.pool_model", "train", "--dataset", "empty"])

    with pytest.raises(SystemExit) as stop:
        main()

    assert stop.value.code == 1
    assert "0 checked frames" in capsys.readouterr().err


def test_size_is_width_x_height():
    assert size("1024x576") == size("1024X576") == (1024, 576)
    with pytest.raises(argparse.ArgumentTypeError, match="WIDTHxHEIGHT"):
        size("1024")
