from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from traffic_agent.evaluation_v4 import (
    QualityGates,
    _trackeval_numpy_compatibility,
    detection_error_diagnostics,
    detection_cache_metrics,
    mode_classification_metrics,
    prepare_predictions,
    prepare_trackeval_sequence,
    quality_gate_report,
    read_cvat_mot_archive,
)


def _mot_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("gt/labels.txt", "car\npedestrian\nignore\n")
        archive.writestr(
            "gt/gt.txt",
            "1,1,10,20,20,30,1,1,1.0\n"
            "2,1,11,20,20,30,1,1,1.0\n"
            "1,2,80,80,10,10,0,3,0.5\n",
        )
    return path


def _tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 7,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 30.0,
                "y2": 50.0,
                "observed": True,
            },
            {
                "frame": 1,
                "track_id": 7,
                "x1": 11.0,
                "y1": 20.0,
                "x2": 31.0,
                "y2": 50.0,
                "observed": False,
            },
        ]
    )


def test_cvat_mot_reader_maps_labels_and_ignored_rows(tmp_path: Path) -> None:
    ground_truth = read_cvat_mot_archive(_mot_archive(tmp_path / "truth.zip"))

    assert list(ground_truth["class_name"]) == ["car", "car", "ignore"]
    assert list(ground_truth["ignored"]) == [False, False, True]
    assert ground_truth.iloc[0]["x2"] == 30.0
    assert ground_truth.iloc[0]["y2"] == 50.0


def test_trackeval_sequence_is_one_based_and_category_agnostic(tmp_path: Path) -> None:
    ground_truth = read_cvat_mot_archive(_mot_archive(tmp_path / "truth.zip"))
    predictions = prepare_predictions(
        _tracks(),
        pd.DataFrame([{"track_id": 7, "final_class_name": "car"}]),
        frame_count=2,
    )

    sequence = prepare_trackeval_sequence(
        ground_truth, predictions, frame_count=2
    )

    assert sequence["num_gt_dets"] == 2
    assert sequence["num_tracker_dets"] == 2
    assert sequence["num_gt_ids"] == 1
    assert sequence["num_tracker_ids"] == 1
    assert sequence["similarity_scores"][0][0, 0] == pytest.approx(1.0)


def test_prepare_predictions_replaces_per_row_class_with_final_track_class() -> None:
    tracks = _tracks().assign(class_name="pedestrian")

    predictions = prepare_predictions(
        tracks,
        pd.DataFrame([{"track_id": 7, "final_class_name": "truck"}]),
        frame_count=2,
    )

    assert predictions["class_name"].tolist() == ["truck", "truck"]
    assert "class_name_x" not in predictions
    assert "class_name_y" not in predictions


def test_mode_accuracy_uses_matched_boxes(tmp_path: Path) -> None:
    ground_truth = read_cvat_mot_archive(_mot_archive(tmp_path / "truth.zip"))
    predictions = prepare_predictions(
        _tracks(),
        pd.DataFrame([{"track_id": 7, "final_class_name": "truck"}]),
        frame_count=2,
    )

    report = mode_classification_metrics(ground_truth, predictions)

    assert report["matched_boxes"] == 2
    assert report["mode_accuracy"] == 0.0
    assert report["confusion"] == {"car": {"truck": 2}}


def test_detection_diagnostics_report_class_grid_and_examples(tmp_path: Path) -> None:
    ground_truth = read_cvat_mot_archive(_mot_archive(tmp_path / "truth.zip"))
    predictions = prepare_predictions(
        _tracks().iloc[[0]].assign(confidence=0.9),
        pd.DataFrame([{"track_id": 7, "final_class_name": "car"}]),
        frame_count=2,
    )

    report = detection_error_diagnostics(
        ground_truth,
        predictions,
        frame_count=2,
        frame_width=100,
        frame_height=100,
        grid_columns=2,
        grid_rows=2,
    )

    assert report["per_class"]["car"] == {"tp": 1, "fn": 1, "fp": 0}
    assert report["spatial_grid"]["cells"]["r1c0"]["tp"] == 1
    assert report["representative_false_negatives"][0]["frame"] == 1
    assert report["representative_false_positives"] == []


def test_detection_cache_metrics_applies_confidence_floor(tmp_path: Path) -> None:
    ground_truth = read_cvat_mot_archive(_mot_archive(tmp_path / "truth.zip"))
    detections = pd.DataFrame(
        [
            {"frame": 0, "class_name": "car", "confidence": 0.9, "x1": 10, "y1": 20, "x2": 30, "y2": 50},
            {"frame": 1, "class_name": "car", "confidence": 0.1, "x1": 11, "y1": 20, "x2": 31, "y2": 50},
        ]
    )

    report = detection_cache_metrics(
        ground_truth,
        detections,
        frame_count=2,
        frame_width=100,
        frame_height=100,
        minimum_confidence=0.5,
    )

    assert report["matched_boxes"] == 1
    assert report["detection_precision"] == 1.0
    assert report["detection_recall"] == 0.5


def test_quality_gate_requires_every_metric() -> None:
    metrics = {
        "detection_precision": 0.95,
        "detection_recall": 0.94,
        "idf1": 0.90,
        "hota": 0.75,
    }
    passing = quality_gate_report(
        metrics,
        {"mode_accuracy": 0.85},
        gates=QualityGates(),
    )
    failing = quality_gate_report(
        {**metrics, "idf1": 0.80},
        {"mode_accuracy": 0.85},
        gates=QualityGates(),
    )

    assert passing["quality_status"] == "passed"
    assert failing["quality_status"] == "failed"
    assert failing["gate_results"]["idf1"] is False


def test_trackeval_numpy_compatibility_is_scoped() -> None:
    import numpy as np

    had_float = "float" in np.__dict__
    had_int = "int" in np.__dict__
    with _trackeval_numpy_compatibility():
        assert np.float is float
        assert np.int is int
    assert ("float" in np.__dict__) is had_float
    assert ("int" in np.__dict__) is had_int
