import json
from dataclasses import fields

import pytest

from swim.pipeline.config import FilteringConfig, PipelineConfig, TrackingConfig, describe


def test_saved_settings_load_back(tmp_path):
    config = PipelineConfig().with_changes({"filtering": {"min_score": 0.3}, "detection": {"size": "large"}})
    config.to_json(tmp_path / "config.json")

    assert PipelineConfig.from_json(tmp_path / "config.json") == config


def test_missing_and_unknown_settings_in_the_file(tmp_path):
    # An older config.json: no filtering stage, a setting that no longer exists
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"detection": {"model": "rfdetr", "size": "small", "threshold": 0.5, "old": 1}}))

    config = PipelineConfig.from_json(path)

    assert (config.detection.size, config.detection.threshold) == ("small", 0.5)
    assert config.filtering == FilteringConfig()


def test_with_changes_does_not_change_the_original():
    config = PipelineConfig()

    changed = config.with_changes({"filtering": {"min_score": 0.35, "max_iou": 1}})

    assert (changed.filtering.min_score, changed.filtering.max_iou) == (0.35, 1.0)
    assert isinstance(changed.filtering.max_iou, float)  # an int from JSON becomes the setting's type
    assert config == PipelineConfig()


@pytest.mark.parametrize("changes, message", [
    ({"pose": {"x": 1}}, "Unknown stage"),
    ({"filtering": {"min_scor": 0.3}}, "Unknown setting"),
    ({"filtering": {"min_score": 1.5}}, "between 0 and 1"),
    ({"filtering": {"min_score": -0.1}}, "between 0 and 1"),
    ({"filtering": {"min_score": "high"}}, "must be a number"),
    ({"filtering": {"min_score": True}}, "must be a number"),
    ({"detection": {"size": 3}}, "must be a str"),
    ({"tracking": {"tracker": "deepsort"}}, "must be one of: bytetrack, ocsort, sort"),
    ({"tracking": {"min_frames": 2.5}}, "whole number"),
    ({"tracking": {"min_frames": 1}}, "between 2 and 10"),
])
def test_with_changes_rejects_bad_values(changes, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig().with_changes(changes)


def test_every_setting_has_an_explanation():
    for stage in (f.name for f in fields(PipelineConfig())):
        for setting in describe(getattr(PipelineConfig(), stage)):
            assert setting["help"], f"{stage}.{setting['name']} has no explanation"


def test_describe_lists_value_and_range():
    assert describe(FilteringConfig(min_score=0.2))[0] | {"help": ""} == {
        "name": "min_score", "value": 0.2, "type": "float", "help": "", "min": 0, "max": 1, "step": 0.05, "choices": None,
    }


def test_describe_lists_allowed_values():
    tracker = describe(TrackingConfig())[0]

    assert (tracker["name"], tracker["type"], tracker["choices"]) == ("tracker", "str", ("bytetrack", "ocsort", "sort"))
