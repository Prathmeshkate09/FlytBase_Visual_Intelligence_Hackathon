from __future__ import annotations

from xml.etree import ElementTree as ET

import pandas as pd
import pytest

from traffic_agent.cvat import VideoMetadata, build_cvat_interpolation_xml


def _tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 10,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 30.0,
                "y2": 50.0,
                "observed": True,
            },
            {
                "frame": 1,
                "track_id": 10,
                "x1": 11.0,
                "y1": 20.0,
                "x2": 31.0,
                "y2": 50.0,
                "observed": False,
            },
            {
                "frame": 2,
                "track_id": 10,
                "x1": 12.0,
                "y1": 20.0,
                "x2": 32.0,
                "y2": 50.0,
                "observed": True,
            },
        ]
    )


def test_cvat_seed_uses_final_class_and_marks_predictions() -> None:
    summaries = pd.DataFrame(
        [{"track_id": 10, "final_class_name": "person"}]
    )

    xml_bytes, report = build_cvat_interpolation_xml(
        _tracks(),
        summaries,
        metadata=VideoMetadata(width=100, height=80, frame_count=5, fps=30.0),
        task_name="held-out",
    )

    root = ET.fromstring(xml_bytes)
    track = root.find("track")
    assert track is not None
    assert track.attrib["label"] == "pedestrian"
    boxes = track.findall("box")
    assert [box.attrib["frame"] for box in boxes] == ["0", "1", "2", "3"]
    assert [box.attrib["outside"] for box in boxes] == ["0", "0", "0", "1"]
    assert [box.attrib["occluded"] for box in boxes[:3]] == ["0", "1", "0"]
    second_attributes = {
        attribute.attrib["name"]: attribute.text
        for attribute in boxes[1].findall("attribute")
    }
    assert second_attributes["v4_track_id"] == "10"
    assert second_attributes["seed_state"] == "kalman_prediction"
    assert second_attributes["review_status"] == "unchecked"
    assert report["ground_truth_status"] == "seed_predictions_only"
    assert report["observed_seed_boxes"] == 2
    assert report["predicted_seed_boxes"] == 1
    assert report["outside_markers"] == 1


def test_cvat_seed_rejects_missing_track_summary() -> None:
    with pytest.raises(ValueError, match="missing final classes"):
        build_cvat_interpolation_xml(
            _tracks(),
            pd.DataFrame([{"track_id": 99, "final_class_name": "car"}]),
            metadata=VideoMetadata(width=100, height=80, frame_count=5, fps=30.0),
            task_name="development",
        )


def test_cvat_seed_rejects_invalid_boxes() -> None:
    tracks = _tracks()
    tracks.loc[0, "x2"] = 150.0

    with pytest.raises(ValueError, match="invalid boxes"):
        build_cvat_interpolation_xml(
            tracks,
            pd.DataFrame([{"track_id": 10, "final_class_name": "car"}]),
            metadata=VideoMetadata(width=100, height=80, frame_count=5, fps=30.0),
            task_name="development",
        )
