import copy
import json

from swim.pipeline.config import FilteringConfig
from swim.stages.detection.types import BBox, Detection, FrameDetections, RejectedDetection, VideoDetections
from swim.stages.filtering.stage import filter_detections, filter_frame
from swim.stages.pool.types import FramePool, VideoPool


def det(x1: float, score: float) -> Detection:
    return Detection(bbox=BBox(x1, 0, x1 + 100, 100), score=score)


def video(*frames: list[Detection]) -> VideoDetections:
    return VideoDetections(
        video_path="data/input/test.MOV", fps=30.0, width=1920, height=1080, model="rfdetr-medium",
        frames=[FrameDetections(frame_idx=i, timestamp=i / 30.0, detections=dets) for i, dets in enumerate(frames)],
    )


def pool(*frames: list) -> VideoPool:
    """The pool stage's water in each frame (a list of polygons per frame)."""
    return VideoPool("data/input/test.MOV", 30.0, 1920, 1080, "fake-pool",
                     [FramePool(i, i / 30.0, water) for i, water in enumerate(frames)])


def all_water(source: VideoDetections) -> VideoPool:
    """Water over each whole frame: no box is outside the pool."""
    return pool(*[WHOLE_FRAME for _ in source.frames])


CONFIG = FilteringConfig(min_score=0.4, max_iou=0.5)
WHOLE_FRAME = [[(0, 0), (1919, 0), (1919, 1079), (0, 1079)]]
LOWER_HALF = [[(0, 540), (1919, 540), (1919, 1079), (0, 1079)]]


def test_filter_frame_records_the_reason():
    swimmer = det(0, score=0.9)
    duplicate = det(10, score=0.7)  # IoU 0.82 with the swimmer
    weak = det(500, score=0.2)

    kept, rejected = filter_frame([swimmer, duplicate, weak], WHOLE_FRAME, 1080, CONFIG)

    assert kept == [swimmer]
    assert rejected == [RejectedDetection(weak, "min_score"), RejectedDetection(duplicate, "duplicate")]


def test_low_score_box_is_rejected_as_min_score_not_duplicate():
    # Overlaps the swimmer too, but the score filter runs first
    swimmer = det(0, score=0.9)
    weak_overlap = det(10, score=0.2)

    kept, rejected = filter_frame([swimmer, weak_overlap], WHOLE_FRAME, 1080, CONFIG)

    assert kept == [swimmer]
    assert rejected == [RejectedDetection(weak_overlap, "min_score")]


def test_filters_off_keep_everything():
    dets = [det(0, score=0.9), det(0, score=0.1)]  # identical boxes, one with a very low score, outside the water

    kept, rejected = filter_frame(dets, LOWER_HALF, 1080, FilteringConfig(min_score=0, max_iou=1.0, pool_margin=1))

    assert kept == dets
    assert rejected == []


def test_box_outside_the_pool_is_rejected_as_outside_pool():
    swimmer = Detection(BBox(0, 600, 100, 700), score=0.9)
    on_deck = det(500, score=0.9)  # bottom at y = 100, the water starts at y = 540

    kept, rejected = filter_frame([swimmer, on_deck], LOWER_HALF, 1080, CONFIG)

    assert kept == [swimmer]
    assert rejected == [RejectedDetection(on_deck, "outside_pool")]


def test_pool_margin_is_a_share_of_the_frame_height():
    near = Detection(BBox(0, 400, 100, 530), score=0.9)  # bottom 10 px above the water

    assert filter_frame([near], LOWER_HALF, 1080, CONFIG)[0] == [near]  # 0.01 × 1080 = 10.8 px
    assert filter_frame([near], LOWER_HALF, 720, CONFIG)[0] == []  # 0.01 × 720 = 7.2 px


def test_box_on_the_deck_does_not_remove_a_swimmer_as_its_duplicate():
    # A person on the deck in front of a swimmer, scoring higher: the boxes overlap (IoU 0.73) but the deck box is
    # removed as outside the pool before the duplicate filter runs
    swimmer = Detection(BBox(0, 330, 100, 560), score=0.6)  # bottom 20 px into the water
    on_deck = Detection(BBox(0, 300, 100, 520), score=0.9)  # bottom 20 px above the water

    kept, rejected = filter_frame([swimmer, on_deck], LOWER_HALF, 1080, CONFIG)

    assert kept == [swimmer]
    assert rejected == [RejectedDetection(on_deck, "outside_pool")]


def test_pool_margin_one_turns_the_pool_filter_off():
    # The box is farther from the water than the frame's height: off, not just a margin of 1 × the height
    corner = [[(1800, 1000), (1919, 1000), (1919, 1079), (1800, 1079)]]  # water only in the lower right corner
    far = det(0, score=0.9)  # bottom-center (50, 100): about 1970 px from that water

    assert filter_frame([far], corner, 1080, FilteringConfig(pool_margin=1)) == ([far], [])
    assert filter_frame([far], corner, 1080, FilteringConfig(pool_margin=0.99))[0] == []


def test_filter_detections_uses_each_frames_water():
    on_top = det(500, score=0.9)  # bottom at y = 100
    source = video([on_top], [on_top], [on_top], [on_top])

    # Frame 2: no water found; frame 3: missing from the pool's results. Both keep their boxes.
    result = filter_detections(source, pool(LOWER_HALF, WHOLE_FRAME, []), CONFIG)

    assert [f.detections for f in result.frames] == [[], [on_top], [on_top], [on_top]]
    assert [[r.reason for r in f.rejected] for f in result.frames] == [["outside_pool"], [], [], []]


def test_filter_detections_per_frame():
    swimmer, duplicate, weak = det(0, score=0.9), det(10, score=0.7), det(500, score=0.2)
    source = video([], [swimmer], [swimmer, duplicate], [weak])

    result = filter_detections(source, all_water(source), CONFIG)

    assert [f.detections for f in result.frames] == [[], [swimmer], [swimmer], []]
    assert [[r.reason for r in f.rejected] for f in result.frames] == [[], [], ["duplicate"], ["min_score"]]


def test_filter_detections_keeps_video_info_and_frame_timing():
    source = video([det(0, score=0.9)], [], [det(0, score=0.3)])

    result = filter_detections(source, all_water(source), CONFIG)

    assert (result.video_path, result.fps, result.width, result.height, result.model) == (
        source.video_path, source.fps, source.width, source.height, source.model
    )
    assert [(f.frame_idx, f.timestamp) for f in result.frames] == [(f.frame_idx, f.timestamp) for f in source.frames]


def test_filter_detections_does_not_change_its_input():
    source = video([det(0, score=0.9), det(10, score=0.7)], [det(500, score=0.2)])
    before = copy.deepcopy(source)

    filter_detections(source, all_water(source), CONFIG)

    assert source == before


def test_filtered_output_survives_save_and_load(tmp_path):
    swimmer, duplicate, weak = det(0, score=0.9), det(10, score=0.7), det(500, score=0.2)
    source = video([swimmer, duplicate, weak])
    result = filter_detections(source, all_water(source), CONFIG)

    result.to_json(tmp_path / "3_filtered.json")
    loaded = VideoDetections.from_json(tmp_path / "3_filtered.json")

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
