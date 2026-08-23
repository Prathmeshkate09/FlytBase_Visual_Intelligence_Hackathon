from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from traffic_agent.lifecycle import LifecycleState
from traffic_agent.pipeline_v4 import (
    PipelineV4Config,
    _confirmed_track_ids,
    _final_summary_rows,
    load_detector_config,
)


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

