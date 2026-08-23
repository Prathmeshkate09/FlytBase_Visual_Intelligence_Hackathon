import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from traffic_agent.appearance import classify_bgr_pixels, relative_size_classes
from traffic_agent.calibration import load_ground_calibration
from traffic_agent.level2 import (
    _level2_rows_by_frame,
    calculate_kinematics,
    calculate_object_insights,
    write_evidence_video,
)


def test_colour_family_classification() -> None:
    red = np.full((100, 3), (0, 0, 255), dtype=np.uint8)
    white = np.full((100, 3), (245, 245, 245), dtype=np.uint8)
    assert classify_bgr_pixels(red) == ("red", 1.0)
    assert classify_bgr_pixels(white) == ("white", 1.0)


def test_relative_vehicle_size_is_scene_relative() -> None:
    raw = pd.DataFrame(
        [
            {"track_id": 1, "class_name": "car", "x1": 0, "y1": 0, "x2": 10, "y2": 10},
            {"track_id": 2, "class_name": "car", "x1": 0, "y1": 0, "x2": 20, "y2": 20},
            {"track_id": 3, "class_name": "car", "x1": 0, "y1": 0, "x2": 30, "y2": 30},
            {"track_id": 4, "class_name": "motorcycle", "x1": 0, "y1": 0, "x2": 5, "y2": 10},
        ]
    )
    result = relative_size_classes(raw).set_index("track_id")
    assert result.loc[1, "relative_size_class"] == "small_passenger_vehicle"
    assert result.loc[2, "relative_size_class"] == "medium_passenger_vehicle"
    assert result.loc[3, "relative_size_class"] == "large_passenger_vehicle"
    assert result.loc[4, "relative_size_class"] == "two_wheeler"


def test_metric_kinematics_require_and_use_homography(tmp_path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "reference": "synthetic 10 m square",
                "quality": "surveyed",
                "max_reprojection_error_m": 0.01,
                "image_points_px": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "world_points_m": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }
        ),
        encoding="utf-8",
    )
    calibration = load_ground_calibration(calibration_path)
    trajectories = pd.DataFrame(
        {
            "frame": np.arange(10),
            "timestamp_s": np.arange(10) / 10,
            "track_id": 7,
            "segment_id": "7-0",
            "class_id": 2,
            "class_name": "car",
            "observed": True,
            "ground_x_smooth": np.arange(10, dtype=float),
            "ground_y_smooth": 50.0,
        }
    )
    result = calculate_kinematics(trajectories, calibration=calibration, smoothing_window=5)
    assert np.allclose(result["speed_m_s"], 1.0, atol=1e-6)
    assert result["inside_calibration_region"].all()
    assert np.allclose(result["speed_kmh"], 3.6, atol=1e-6)


def test_uncalibrated_kinematics_do_not_emit_metric_columns() -> None:
    trajectories = pd.DataFrame(
        {
            "frame": np.arange(5),
            "timestamp_s": np.arange(5) / 10,
            "track_id": 7,
            "segment_id": "7-0",
            "class_id": 2,
            "class_name": "car",
            "observed": True,
            "ground_x_smooth": np.arange(5, dtype=float),
            "ground_y_smooth": 50.0,
        }
    )
    result = calculate_kinematics(trajectories, calibration=None, smoothing_window=5)
    assert "speed_px_s" in result.columns
    assert "speed_m_s" not in result.columns
    assert "speed_kmh" not in result.columns


def test_object_insights_keep_one_canonical_class_column() -> None:
    raw = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "track_id": [7, 7, 7],
            "class_id": [2, 2, 2],
            "class_name": ["car", "car", "car"],
            "x1": [0.0, 1.0, 2.0],
            "y1": [0.0, 0.0, 0.0],
            "x2": [10.0, 11.0, 12.0],
            "y2": [10.0, 10.0, 10.0],
        }
    )
    kinematics = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "timestamp_s": [0.0, 0.1, 0.2],
            "track_id": [7, 7, 7],
            "segment_id": ["7-0", "7-0", "7-0"],
            "observed": [True, True, True],
            "position_x_px": [5.0, 6.0, 7.0],
            "position_y_px": [10.0, 10.0, 10.0],
            "speed_px_s": [0.0, 10.0, 10.0],
        }
    )
    appearance = pd.DataFrame(
        {
            "track_id": [7],
            "dominant_color": ["red"],
            "color_confidence": [0.9],
            "color_samples": [3],
        }
    )
    sizes = relative_size_classes(raw)

    result = calculate_object_insights(
        raw,
        kinematics,
        appearance,
        sizes,
        fps=10.0,
        metric_available=False,
    )

    assert result.loc[0, "class_name"] == "car"
    assert "class_name_x" not in result.columns
    assert "class_name_y" not in result.columns


def test_level2_renderer_keeps_every_box_without_object_cap() -> None:
    raw = pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": track_id,
                "x1": float(track_id),
                "y1": 0.0,
                "x2": float(track_id + 5),
                "y2": 10.0,
                "observed": track_id % 2 == 0,
            }
            for track_id in range(1, 31)
        ]
    )

    rows_by_frame = _level2_rows_by_frame(raw)

    assert len(rows_by_frame[0]) == 30
    assert set(rows_by_frame[0]["track_id"]) == set(range(1, 31))


def test_level2_evidence_report_proves_every_box_was_rendered(
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (100, 100)
    )
    assert writer.isOpened()
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    raw = pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 1,
                "class_name": "car",
                "x1": 10.0,
                "y1": 10.0,
                "x2": 30.0,
                "y2": 30.0,
                "observed": True,
            },
            {
                "frame": 1,
                "track_id": 1,
                "class_name": "car",
                "x1": 12.0,
                "y1": 10.0,
                "x2": 32.0,
                "y2": 30.0,
                "observed": False,
            },
        ]
    )
    objects = pd.DataFrame(
        [
            {
                "track_id": 1,
                "duration_s": 0.2,
                "dominant_color": "unknown",
                "color_confidence": 0.0,
                "relative_size_class": "unknown",
            }
        ]
    )
    kinematics = pd.DataFrame(
        [
            {"frame": 0, "track_id": 1, "speed_px_s": 0.0},
            {"frame": 1, "track_id": 1, "speed_px_s": 0.0},
        ]
    )

    report = write_evidence_video(
        video,
        tmp_path / "evidence.mp4",
        raw=raw,
        objects=objects,
        kinematics=kinematics,
        fps=10.0,
        max_width=100,
        trail_length=5,
        max_labels_per_frame=1,
    )

    assert report["input_track_rows"] == 2
    assert report["rendered_boxes"] == 2
    assert report["complete"] is True
