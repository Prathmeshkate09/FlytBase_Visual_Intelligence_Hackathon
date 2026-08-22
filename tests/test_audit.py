import pandas as pd
import pytest

from traffic_agent.audit import AuditThresholds, audit_tracks


def _row(
    frame: int,
    track_id: int,
    *,
    x: float,
    confidence: float = 0.9,
    class_id: int = 2,
    class_name: str = "car",
) -> dict[str, object]:
    return {
        "frame": frame,
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "x1": x,
        "y1": 100.0,
        "x2": x + 40.0,
        "y2": 130.0,
        "center_x": x + 20.0,
        "center_y": 115.0,
    }


def test_audit_detects_persistent_duplicates_and_same_frame_handoff() -> None:
    rows = []
    for frame in range(5):
        rows.append(_row(frame, 40, x=100 + frame))
        rows.append(_row(frame, 844, x=100 + frame))
    for frame in range(4, 9):
        rows.append(_row(frame, 1323, x=105 + frame))

    report, failures = audit_tracks(
        pd.DataFrame(rows),
        fps=10.0,
        frame_width=400,
        frame_height=300,
        thresholds=AuditThresholds(
            persistent_duplicate_frames=5,
            boundary_margin_px=20,
            stable_track_seconds=0.2,
        ),
    )

    assert report["audit_status"] == "failed"
    assert report["metrics"]["persistent_duplicate_pairs"] >= 1
    assert report["metrics"]["suspected_id_handoffs"] >= 1
    assert {"persistent_duplicate", "suspected_id_handoff"}.issubset(set(failures["event_type"]))
    duplicate = report["persistent_duplicate_examples"][0]
    assert {duplicate["track_id"], duplicate["other_track_id"]} == {40, 844}


def test_audit_uses_raw_classes_and_reports_evidence_filtering() -> None:
    rows = []
    for frame in range(10):
        rows.append(
            _row(
                frame,
                7,
                x=100 + frame,
                confidence=0.10 if frame < 5 else 0.80,
                class_id=2 if frame < 8 else 7,
                class_name="car" if frame < 8 else "truck",
            )
        )

    report, failures = audit_tracks(
        pd.DataFrame(rows),
        fps=10.0,
        frame_width=400,
        frame_height=300,
        thresholds=AuditThresholds(stable_track_seconds=0.2, display_confidence=0.20),
    )

    assert report["metrics"]["class_changing_tracks"] == 1
    assert report["metrics"]["unstable_class_tracks"] == 1
    assert report["metrics"]["median_raw_class_consistency"] == pytest.approx(0.8)
    assert report["evidence_filter_audit"]["display_share"] == pytest.approx(0.5)
    assert "class_instability" in set(failures["event_type"])


def test_audit_rejects_invalid_boxes() -> None:
    invalid = pd.DataFrame([_row(0, 1, x=100)])
    invalid.loc[0, "x2"] = invalid.loc[0, "x1"]

    with pytest.raises(ValueError, match="invalid bounding boxes"):
        audit_tracks(invalid, fps=10.0, frame_width=400, frame_height=300)

