import copy
import json

from swim.pipeline.config import FilteringConfig
from swim.stages.detection.types import BBox, Detection, FrameDetections, RejectedDetection, VideoDetections
from swim.stages.filtering.stage import filter_detections, filter_frame


def det(x1: float, score: float) -> Detection:
    return Detection(bbox=BBox(x1, 0, x1 + 100, 100), score=score)


def video(*frames: list[Detection]) -> VideoDetections:
    return VideoDetections(
        video_path="data/input/test.MOV", fps=30.0, width=1920, height=1080, model="rfdetr-medium",
        frames=[FrameDetections(frame_idx=i, timestamp=i / 30.0, detections=dets) for i, dets in enumerate(frames)],
    )


CONFIG = FilteringConfig(min_score=0.4, max_iou=0.5)


def test_filter_frame_records_the_reason():
    swimmer = det(0, score=0.9)
    duplicate = det(10, score=0.7)  # IoU 0.82 with the swimmer
    weak = det(500, score=0.2)

    kept, rejected = filter_frame([swimmer, duplicate, weak], CONFIG)

    assert kept == [swimmer]
    assert rejected == [RejectedDetection(weak, "min_score"), RejectedDetection(duplicate, "duplicate")]


def test_low_score_box_is_rejected_as_min_score_not_duplicate():
    # Overlaps the swimmer too, but the score filter runs first
    swimmer = det(0, score=0.9)
    weak_overlap = det(10, score=0.2)

    kept, rejected = filter_frame([swimmer, weak_overlap], CONFIG)

    assert kept == [swimmer]
    assert rejected == [RejectedDetection(weak_overlap, "min_score")]


def test_filters_off_keep_everything():
    dets = [det(0, score=0.9), det(0, score=0.1)]  # identical boxes, one with a very low score

    kept, rejected = filter_frame(dets, FilteringConfig(min_score=0, max_iou=1.0))

    assert kept == dets
    assert rejected == []


def test_filter_detections_per_frame():
    swimmer, duplicate, weak = det(0, score=0.9), det(10, score=0.7), det(500, score=0.2)
    source = video([], [swimmer], [swimmer, duplicate], [weak])

    result = filter_detections(source, CONFIG)

    assert [f.detections for f in result.frames] == [[], [swimmer], [swimmer], []]
    assert [[r.reason for r in f.rejected] for f in result.frames] == [[], [], ["duplicate"], ["min_score"]]


def test_filter_detections_keeps_video_info_and_frame_timing():
    source = video([det(0, score=0.9)], [], [det(0, score=0.3)])

    result = filter_detections(source, CONFIG)

    assert (result.video_path, result.fps, result.width, result.height, result.model) == (
        source.video_path, source.fps, source.width, source.height, source.model
    )
    assert [(f.frame_idx, f.timestamp) for f in result.frames] == [(f.frame_idx, f.timestamp) for f in source.frames]


def test_filter_detections_does_not_change_its_input():
    source = video([det(0, score=0.9), det(10, score=0.7)], [det(500, score=0.2)])
    before = copy.deepcopy(source)

    filter_detections(source, CONFIG)

    assert source == before


def test_filtered_output_survives_save_and_load(tmp_path):
    swimmer, duplicate, weak = det(0, score=0.9), det(10, score=0.7), det(500, score=0.2)
    result = filter_detections(video([swimmer, duplicate, weak]), CONFIG)

    result.to_json(tmp_path / "2_filtered.json")
    loaded = VideoDetections.from_json(tmp_path / "2_filtered.json")

    assert loaded == result
    assert [r.reason for r in loaded.frames[0].rejected] == ["min_score", "duplicate"]


def test_detections_saved_before_filtering_existed_still_load(tmp_path):
    # Older 1_detections.json files have no "rejected" key
    path = tmp_path / "1_detections.json"
    video([det(0, score=0.9)]).to_json(path)
    data = json.loads(path.read_text())
    for f in data["frames"]:
        del f["rejected"]
    path.write_text(json.dumps(data))

    loaded = VideoDetections.from_json(path)

    assert loaded.frames[0].detections == [det(0, score=0.9)]
    assert loaded.frames[0].rejected == []
