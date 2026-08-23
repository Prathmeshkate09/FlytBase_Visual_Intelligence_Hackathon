from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_agent.postprocess import Detection
from traffic_agent.tracker import GroupedBoTSORT


def _detection(class_name: str, class_id: int, x: float) -> Detection:
    return Detection(
        x1=x,
        y1=10.0,
        x2=x + 20.0,
        y2=30.0,
        confidence=0.8,
        class_id=class_id,
        class_name=class_name,
        source="test",
    )


@dataclass
class _FakeLostTrack:
    track_id: int
    xyxy: np.ndarray


class _FakeTracker:
    def __init__(self, native_id: int = 1) -> None:
        self.native_id = native_id
        self.calls: list[int] = []
        self.lost_stracks: list[_FakeLostTrack] = []
        self.removed_stracks: list[_FakeLostTrack] = []

    def update(self, boxes: list[Detection], frame: np.ndarray) -> np.ndarray:
        del frame
        self.calls.append(len(boxes))
        if boxes:
            detection = boxes[0]
            self.lost_stracks = []
            return np.asarray(
                [[
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                    self.native_id,
                    detection.confidence,
                    0,
                    0,
                ]],
                dtype=np.float32,
            )
        self.lost_stracks = [
            _FakeLostTrack(self.native_id, np.asarray([11.0, 10.0, 31.0, 30.0]))
        ]
        return np.empty((0, 8), dtype=np.float32)


def test_association_groups_have_distinct_global_ids() -> None:
    trackers: dict[str, _FakeTracker] = {}

    def tracker_factory(group: str) -> _FakeTracker:
        trackers[group] = _FakeTracker(native_id=1)
        return trackers[group]

    adapter = GroupedBoTSORT(
        tracker_config=None,  # type: ignore[arg-type]
        tracker_factory=tracker_factory,
        boxes_factory=lambda detections, shape: detections,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    rows = adapter.update(
        [_detection("person", 0, 10.0), _detection("car", 2, 50.0)],
        frame,
        0,
    )

    assert len(rows) == 2
    assert set(trackers) == {"cyclist", "motorcycle", "pedestrian", "road_vehicle"}
    assert len({row.track_id for row in rows}) == 2
    assert {row.association_group for row in rows} == {"pedestrian", "road_vehicle"}


def test_empty_group_frame_advances_tracker_and_emits_occlusion() -> None:
    fake_tracker = _FakeTracker()
    adapter = GroupedBoTSORT(
        tracker_config=None,  # type: ignore[arg-type]
        tracker_factory=lambda group: fake_tracker,
        boxes_factory=lambda detections, shape: detections,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    observed = adapter.update([_detection("car", 2, 10.0)], frame, 0)
    occluded = adapter.update([], frame, 1)

    assert fake_tracker.calls == [1, 0]
    assert observed[0].observed is True
    assert occluded[0].track_id == observed[0].track_id
    assert occluded[0].observed is False
    assert occluded[0].state == "occluded"
    assert occluded[0].confidence is None
    assert occluded[0].detection_source == "kalman_prediction"


def test_invalid_tracker_detection_index_fails_loudly() -> None:
    class BadTracker(_FakeTracker):
        def update(self, boxes: list[Detection], frame: np.ndarray) -> np.ndarray:
            row = super().update(boxes, frame)
            row[0, -1] = 99
            return row

    adapter = GroupedBoTSORT(
        tracker_config=None,  # type: ignore[arg-type]
        tracker_factory=lambda group: BadTracker(),
        boxes_factory=lambda detections, shape: detections,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        adapter.update([_detection("car", 2, 10.0)], frame, 0)
    except RuntimeError as error:
        assert "invalid detection index" in str(error)
    else:
        raise AssertionError("invalid tracker detection index was accepted")


def test_old_prediction_is_audited_but_not_rendered() -> None:
    fake_tracker = _FakeTracker()
    adapter = GroupedBoTSORT(
        tracker_config=None,  # type: ignore[arg-type]
        max_prediction_frames=1,
        tracker_factory=lambda group: fake_tracker,
        boxes_factory=lambda detections, shape: detections,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    adapter.update([_detection("car", 2, 10.0)], frame, 0)
    recent_prediction = adapter.update([], frame, 1)
    old_prediction = adapter.update([], frame, 2)

    assert len(recent_prediction) == 1
    assert recent_prediction[0].observed is False
    assert old_prediction == []
    assert adapter.rejected_predictions[-1].reason == "prediction_age_exceeded"


def test_invalid_boundary_prediction_is_rejected_and_audited() -> None:
    class BoundaryLostTracker(_FakeTracker):
        def update(self, boxes: list[Detection], frame: np.ndarray) -> np.ndarray:
            if boxes:
                return super().update(boxes, frame)
            self.calls.append(0)
            self.lost_stracks = [
                _FakeLostTrack(
                    self.native_id, np.asarray([10.0, 100.0, 30.0, 120.0])
                )
            ]
            return np.empty((0, 8), dtype=np.float32)

    fake_tracker = BoundaryLostTracker()
    adapter = GroupedBoTSORT(
        tracker_config=None,  # type: ignore[arg-type]
        tracker_factory=lambda group: fake_tracker,
        boxes_factory=lambda detections, shape: detections,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    adapter.update([_detection("car", 2, 10.0)], frame, 0)
    rows = adapter.update([], frame, 1)

    assert rows == []
    assert len(adapter.rejected_predictions) == 1
    rejection = adapter.rejected_predictions[0]
    assert rejection.track_id == 1
    assert rejection.frame == 1
    assert rejection.reason == "invalid_after_frame_clipping"
    assert rejection.clipped_y1 == rejection.clipped_y2 == 100.0

