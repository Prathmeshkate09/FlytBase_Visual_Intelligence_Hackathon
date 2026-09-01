from __future__ import annotations

import pytest

from traffic_agent.tuning_v4 import report_rank, tracker_config_variant


def _report(*, status: str, hota: float, idf1: float, switches: int = 0) -> dict:
    return {
        "quality_status": status,
        "gate_results": {
            "detection_precision": status == "passed",
            "detection_recall": status == "passed",
            "idf1": status == "passed",
            "hota": status == "passed",
            "mode_accuracy": True,
        },
        "ground_truth_metrics": {
            "detection_precision": 0.95 if status == "passed" else 0.8,
            "detection_recall": 0.95 if status == "passed" else 0.8,
            "idf1": idf1,
            "hota": hota,
            "id_switches": switches,
            "fragmentations": 4,
        },
    }


def test_report_rank_prefers_gate_pass_then_hota() -> None:
    assert report_rank(_report(status="passed", hota=0.71, idf1=0.86)) > report_rank(
        _report(status="failed", hota=0.90, idf1=0.95)
    )
    assert report_rank(_report(status="passed", hota=0.75, idf1=0.86)) > report_rank(
        _report(status="passed", hota=0.72, idf1=0.90)
    )


def test_tracker_variant_rejects_invalid_threshold_order() -> None:
    base = {
        "track_low_thresh": 0.05,
        "track_high_thresh": 0.25,
        "new_track_thresh": 0.25,
    }
    with pytest.raises(ValueError, match="cannot exceed"):
        tracker_config_variant(base, {"track_low_thresh": 0.30})
