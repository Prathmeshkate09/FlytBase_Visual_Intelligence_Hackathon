from __future__ import annotations

from dataclasses import replace

from traffic_agent.lifecycle import LifecycleConfig, LifecycleState, TrackLifecycleManager
from traffic_agent.postprocess import RoadUserROI
from traffic_agent.tracker import TrackObservation


def _observation(frame: int, *, observed: bool = True, class_name: str = "car") -> TrackObservation:
    return TrackObservation(
        frame=frame,
        track_id=1,
        native_track_id=10,
        association_group="road_vehicle",
        x1=10.0,
        y1=10.0,
        x2=30.0,
        y2=30.0,
        confidence=0.8 if observed else None,
        detected_class_id=2 if class_name == "car" else 7,
        detected_class_name=class_name,
        observed=observed,
        state="tracked" if observed else "occluded",
        detection_source="test" if observed else "kalman_prediction",
    )


def _manager(exit_roi: RoadUserROI | None = None) -> TrackLifecycleManager:
    return TrackLifecycleManager(
        LifecycleConfig(confirmation_observations=2),
        exit_roi=exit_roi,
        frame_width=100,
        frame_height=100,
    )


def test_confirm_occlude_recover_and_video_end() -> None:
    manager = _manager()
    manager.update([_observation(0)], frame_number=0)
    manager.update([_observation(1)], frame_number=1)
    manager.update([_observation(2, observed=False)], frame_number=2)
    manager.update([_observation(3)], frame_number=3)
    manager.update([_observation(4)], frame_number=4)
    manager.finalize_video(4)

    assert [event.event for event in manager.events] == [
        "created",
        "confirmed",
        "lost",
        "recovered",
        "video_end",
    ]
    assert manager.summaries()[0].state == LifecycleState.VIDEO_END
    assert manager.summaries()[0].observation_count == 4


def test_removed_track_is_exit_only_inside_configured_exit() -> None:
    exit_roi = RoadUserROI(
        include_polygons=(((0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)),)
    )
    manager = _manager(exit_roi)
    manager.update([_observation(0)], frame_number=0)
    manager.update([], frame_number=1, removed_track_ids=[1])

    assert manager.summaries()[0].state == LifecycleState.EXITED
    assert manager.events[-1].reason == "inside_configured_exit"


def test_internal_removal_is_reported_as_expired() -> None:
    manager = _manager()
    manager.update([_observation(0)], frame_number=0)
    manager.update([], frame_number=1, removed_track_ids=[1])

    assert manager.summaries()[0].state == LifecycleState.EXPIRED
    assert manager.events[-1].reason == "removed_away_from_exit"


def test_final_class_uses_confidence_weighted_temporal_vote() -> None:
    manager = _manager()
    manager.update([replace(_observation(0, class_name="truck"), confidence=0.3)], frame_number=0)
    manager.update([replace(_observation(1, class_name="truck"), confidence=0.3)], frame_number=1)
    manager.update([replace(_observation(2, class_name="car"), confidence=0.9)], frame_number=2)

    summary = manager.summaries()[0]
    assert summary.final_class_name == "car"
    assert summary.final_class_confidence == 0.6


def test_finalized_track_cannot_reappear() -> None:
    manager = _manager()
    manager.update([_observation(0)], frame_number=0)
    manager.update([], frame_number=1, removed_track_ids=[1])

    try:
        manager.update([_observation(2)], frame_number=2)
    except RuntimeError as error:
        assert "Finalized track" in str(error)
    else:
        raise AssertionError("finalized track was allowed to reappear")
