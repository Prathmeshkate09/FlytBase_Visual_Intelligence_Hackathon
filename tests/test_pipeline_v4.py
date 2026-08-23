from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from traffic_agent.lifecycle import LifecycleState
from traffic_agent.pipeline_v4 import (
    PipelineV4Config,
    _apply_confidence_thresholds,
    _confirmed_track_ids,
    _extend_cached_detections,
    _final_summary_rows,
    _load_cached_detections,
    load_detector_config,
)
from traffic_agent.postprocess import Detection


def test_load_detector_config_normalizes_class_ids(tmp_path: Path) -> None:
    path = tmp_path / "detector.json"
    path.write_text(
        json.dumps(
            {
                "model_path": "weights.pt",
                "class_ids": [0, 2, 3],
                "use_sahi": False,
            }
        ),
        encoding="utf-8",
    )

    config = load_detector_config(path)

    assert config.class_ids == (0, 2, 3)


def test_pipeline_config_rejects_missing_inputs(tmp_path: Path) -> None:
    config = PipelineV4Config(
        input_video=tmp_path / "missing.mp4",
        output_dir=tmp_path / "output",
        detection_config=tmp_path / "missing.json",
        tracker_config=tmp_path / "missing.yaml",
    )

    with pytest.raises(FileNotFoundError, match="Video not found"):
        config.validate()


def test_cached_detection_loader_and_class_threshold_filter(tmp_path: Path) -> None:
    path = tmp_path / "detections.csv"
    path.write_text(
        "frame,class_id,class_name,confidence,x1,y1,x2,y2,source,source_classes,merge_count\n"
        "0,0,pedestrian,0.19,1,2,11,22,standard,people,1\n"
        "0,3,motorcycle,0.81,5,8,20,24,standard,motor,1\n",
        encoding="utf-8",
    )
    grouped = _load_cached_detections(path)
    config = load_detector_config(
        _write_detector_config(tmp_path, {"pedestrian": 0.20, "motorcycle": 0.50})
    )

    accepted, rejected = _apply_confidence_thresholds(grouped[0], config)

    assert [item.class_name for item in accepted] == ["motorcycle"]
    assert [item.reason for item in rejected] == [
        "below_class_confidence_threshold"
    ]


def test_cached_rejected_loader_restores_only_roi_rejections(tmp_path: Path) -> None:
    accepted_path = tmp_path / "detections.csv"
    accepted_path.write_text(
        "frame,class_id,class_name,confidence,x1,y1,x2,y2\n"
        "0,2,car,0.80,1,2,11,22\n",
        encoding="utf-8",
    )
    rejected_path = tmp_path / "rejected_detections.csv"
    rejected_path.write_text(
        "frame,class_id,class_name,confidence,x1,y1,x2,y2,reason\n"
        "0,0,pedestrian,0.75,20,20,30,40,outside_road_user_roi\n"
        "0,3,motorcycle,0.90,40,20,55,40,rider_duplicate_suppressed\n"
        "1,1,bicycle,0.70,60,20,75,40,duplicate_suppressed\n",
        encoding="utf-8",
    )

    accepted = _load_cached_detections(accepted_path)
    recovered = _load_cached_detections(
        rejected_path, allowed_reasons={"outside_road_user_roi"}
    )
    _extend_cached_detections(accepted, recovered)

    assert [item.class_name for item in accepted[0]] == ["car", "pedestrian"]
    assert 1 not in accepted


def _write_detector_config(
    directory: Path, thresholds: dict[str, float]
) -> Path:
    path = directory / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "model_path": "weights.pt",
                "class_ids": [0, 3],
                "use_sahi": False,
                "confidence": 0.05,
                "class_confidence_thresholds": thresholds,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_confirmation_filter_retains_complete_history_only_for_stable_tracks() -> None:
    summaries = [
        {"track_id": 10, "observation_count": 2},
        {"track_id": 20, "observation_count": 3},
        {"track_id": 30, "observation_count": 8},
    ]

    assert _confirmed_track_ids(summaries, 3) == {20, 30}


def test_final_summary_aggregates_stitched_source_ids() -> None:
    stitched = pd.DataFrame(
        [
            {
                "frame": 1,
                "track_id": 10,
                "source_track_id": 10,
                "observed": True,
                "confidence": 0.8,
                "detected_class_id": 2,
                "detected_class_name": "car",
            },
            {
                "frame": 5,
                "track_id": 10,
                "source_track_id": 20,
                "observed": True,
                "confidence": 0.7,
                "detected_class_id": 2,
                "detected_class_name": "car",
            },
        ]
    )
    native = {
        10: SimpleNamespace(
            last_frame=3,
            last_observed_frame=3,
            state=LifecycleState.OCCLUDED,
        ),
        20: SimpleNamespace(
            last_frame=5,
            last_observed_frame=5,
            state=LifecycleState.VIDEO_END,
        ),
    }

    rows = _final_summary_rows(stitched, native)

    assert rows == [
        {
            "track_id": 10,
            "state": "video_end",
            "first_frame": 1,
            "last_frame": 5,
            "last_observed_frame": 5,
            "observation_count": 2,
            "final_class_id": 2,
            "final_class_name": "car",
            "final_class_confidence": 1.0,
        }
    ]

