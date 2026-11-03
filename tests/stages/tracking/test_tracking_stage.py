import copy
from dataclasses import replace

import pytest

from swim.pipeline.config import TrackingConfig
from swim.stages.detection.types import BBox, Detection, FrameDetections, VideoDetections
from swim.stages.tracking.stage import track

FPS = 30.0
TRACKERS = ["bytetrack", "ocsort", "sort"]


def swimmer(frame: int, lane_y: float = 100, score: float = 0.9) -> Detection:
    """A swimmer moving 8 px per frame along a lane."""
    x = 10 + 8 * frame
    return Detection(bbox=BBox(x, lane_y, x + 120, lane_y + 60), score=score)


def video(frames: list[list[Detection]]) -> VideoDetections:
    return VideoDetections(
        video_path="data/input/test.MOV", fps=FPS, width=1920, height=1080, model="rfdetr-medium",
        # Timestamps rounded as when saved, so a saved and loaded copy compares equal
        frames=[FrameDetections(frame_idx=i, timestamp=round(i / FPS, 4), detections=dets) for i, dets in enumerate(frames)],
    )


def ids(result: VideoDetections) -> list[list[int]]:
    return [[d.track_id for d in f.detections] for f in result.frames]


def settings(**changes) -> TrackingConfig:
    return replace(TrackingConfig(), **changes)


@pytest.mark.parametrize("tracker", TRACKERS)
def test_one_swimmer_keeps_one_id_from_the_first_frame(tracker):
    result = track(video([[swimmer(i)] for i in range(30)]), settings(tracker=tracker))

    assert ids(result) == [[1]] * 30  # the box that starts the track gets its ID too


@pytest.mark.parametrize("tracker", TRACKERS)
def test_two_swimmers_in_separate_lanes_keep_their_ids(tracker):
    # Lane B's swimmer appears 3 frames later: IDs are numbered in order of first appearance
    frames = [[swimmer(i, lane_y=100)] + ([swimmer(i, lane_y=500)] if i >= 3 else []) for i in range(30)]

    result = track(video(frames), settings(tracker=tracker))

    by_lane = {(d.bbox.y1, d.track_id) for f in result.frames for d in f.detections}
    assert by_lane == {(100, 1), (500, 2)}


@pytest.mark.parametrize("tracker", TRACKERS)
def test_gap_shorter_than_lost_seconds_keeps_the_id(tracker):
    frames = [[swimmer(i)] if not 10 <= i < 20 else [] for i in range(30)]  # 10 frames = 0.33 s without the swimmer

    result = track(video(frames), settings(tracker=tracker, lost_seconds=1.0))

    assert {tid for f in ids(result) for tid in f} == {1}
    assert sum(1 for f in result.frames if f.detections) == 20


def test_gap_longer_than_lost_seconds_starts_a_new_track():
    frames = [[swimmer(i)] if not 10 <= i < 70 else [] for i in range(80)]  # 60 frames = 2 s

    result = track(video(frames), settings(lost_seconds=1.0))

    assert ids(result)[:10] == [[1]] * 10
    assert ids(result)[70:] == [[2]] * 10


def test_bytetrack_weak_box_continues_a_track_but_cannot_start_one():
    # The swimmer's score drops to 0.3 from frame 10 (under water); a weak box in another lane all along
    frames = [[swimmer(i, score=0.9 if i < 10 else 0.3), swimmer(i, lane_y=700, score=0.3)] for i in range(20)]

    result = track(video(frames), settings(tracker="bytetrack", start_score=0.5, high_score=0.5))

    assert ids(result) == [[1]] * 20
    assert all([r.reason for r in f.rejected] == ["untracked"] for f in result.frames)
    assert all(f.rejected[0].detection.bbox.y1 == 700 for f in result.frames)


def test_ocsort_ignores_boxes_below_high_score():
    frames = [[swimmer(i, score=0.9 if i < 10 else 0.3)] for i in range(20)]

    result = track(video(frames), settings(tracker="ocsort", high_score=0.5))

    assert ids(result) == [[1]] * 10 + [[]] * 10


def seen_for(frames: int, start: int = 3) -> list[list[Detection]]:
    """A box seen in `frames` frames in a row, from frame `start`, in a 12-frame video."""
    return [[swimmer(i)] if start <= i < start + frames else [] for i in range(12)]


def test_a_person_must_be_seen_min_frames_in_a_row_to_become_a_track():
    assert ids(track(video(seen_for(1)), settings(min_frames=2)))[3] == []  # e.g. a reflection seen once
    assert ids(track(video(seen_for(2)), settings(min_frames=2)))[3:5] == [[1], [1]]
    assert ids(track(video(seen_for(2)), settings(min_frames=3)))[3:5] == [[], []]
    assert ids(track(video(seen_for(3)), settings(min_frames=3)))[3:6] == [[1], [1], [1]]


def test_output_keeps_boxes_video_info_and_frame_timing():
    source = video([[swimmer(i), swimmer(i, lane_y=700, score=0.2)] for i in range(10)])

    result = track(source, TrackingConfig())

    assert (result.video_path, result.fps, result.width, result.height, result.model) == (
        source.video_path, source.fps, source.width, source.height, source.model
    )
    assert [(f.frame_idx, f.timestamp) for f in result.frames] == [(f.frame_idx, f.timestamp) for f in source.frames]
    for before, after in zip(source.frames, result.frames):
        kept = [replace(d, track_id=None) for d in after.detections] + [r.detection for r in after.rejected]
        assert sorted(kept, key=lambda d: d.bbox.y1) == before.detections  # same boxes and scores, none lost


def test_input_is_not_changed_and_output_survives_save_and_load(tmp_path):
    source = video([[swimmer(i)] for i in range(10)])
    before = copy.deepcopy(source)

    result = track(source, TrackingConfig())
    result.to_json(tmp_path / "3_tracks.json")

    assert source == before
    assert VideoDetections.from_json(tmp_path / "3_tracks.json") == result


def test_unknown_tracker():
    with pytest.raises(ValueError, match="Unknown tracker"):
        track(video([[swimmer(0)]]), settings(tracker="deepsort"))
