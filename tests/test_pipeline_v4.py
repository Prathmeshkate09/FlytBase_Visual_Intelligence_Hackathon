from __future__ import annotations

import json
from pathlib import Path

import pytest

from traffic_agent.pipeline_v4 import PipelineV4Config, load_detector_config


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
