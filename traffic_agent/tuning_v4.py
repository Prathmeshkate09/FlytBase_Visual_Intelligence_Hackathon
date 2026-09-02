from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DETECTION_THRESHOLD_VARIANTS: dict[str, dict[str, float]] = {
    "current": {
        "pedestrian": 0.15, "bicycle": 0.15, "car": 0.12, "lgv": 0.12,
        "hgv": 0.12, "truck": 0.12, "bus": 0.12, "motorcycle": 0.05,
    },
    "recall": {
        "pedestrian": 0.05, "bicycle": 0.05, "car": 0.05, "lgv": 0.05,
        "hgv": 0.05, "truck": 0.05, "bus": 0.05, "motorcycle": 0.05,
    },
    "balanced": {
        "pedestrian": 0.10, "bicycle": 0.10, "car": 0.08, "lgv": 0.08,
        "hgv": 0.08, "truck": 0.08, "bus": 0.08, "motorcycle": 0.05,
    },
    "precision": {
        "pedestrian": 0.20, "bicycle": 0.20, "car": 0.18, "lgv": 0.18,
        "hgv": 0.18, "truck": 0.18, "bus": 0.18, "motorcycle": 0.10,
    },
    # Scene-refit candidate-cache evaluation found the useful precision/recall
    # frontier above the legacy aerial-checkpoint thresholds. Keep these
    # explicit so replay tuning covers that evidence without rerunning the GPU.
    "uniform030": {
        "pedestrian": 0.30, "bicycle": 0.30, "car": 0.30, "lgv": 0.30,
        "hgv": 0.30, "truck": 0.30, "bus": 0.30, "motorcycle": 0.30,
    },
    "uniform035": {
        "pedestrian": 0.35, "bicycle": 0.35, "car": 0.35, "lgv": 0.35,
        "hgv": 0.35, "truck": 0.35, "bus": 0.35, "motorcycle": 0.35,
    },
    "uniform040": {
        "pedestrian": 0.40, "bicycle": 0.40, "car": 0.40, "lgv": 0.40,
        "hgv": 0.40, "truck": 0.40, "bus": 0.40, "motorcycle": 0.40,
    },
}


TRACKER_VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "high015": {"track_high_thresh": 0.15, "new_track_thresh": 0.15},
    "high020": {"track_high_thresh": 0.20, "new_track_thresh": 0.20},
    "high030": {"track_high_thresh": 0.30, "new_track_thresh": 0.30},
    "match070": {"match_thresh": 0.70},
    "match090": {"match_thresh": 0.90},
    "fuse_score": {"fuse_score": True},
    "no_gmc": {"gmc_method": "none"},
}


def report_rank(report: dict[str, Any]) -> tuple[float, ...]:
    metrics = report["ground_truth_metrics"]
    gates = report["gate_results"]
    passed = sum(bool(value) for value in gates.values())
    is_passing = report["quality_status"] == "passed"
    weakest_gate_ratio = min(
        float(metrics["detection_precision"]) / 0.90,
        float(metrics["detection_recall"]) / 0.90,
        float(metrics["idf1"]) / 0.85,
        float(metrics["hota"]) / 0.70,
    )
    return (
        float(is_passing),
        float(passed),
        2.0 if is_passing else weakest_gate_ratio,
        float(metrics["hota"]),
        float(metrics["idf1"]),
        -float(metrics["id_switches"]),
        -float(metrics["fragmentations"]),
    )


def detector_config_variant(
    base: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, Any]:
    return {**base, "class_confidence_thresholds": dict(thresholds)}


def tracker_config_variant(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    result = {**base, **overrides}
    if float(result["track_low_thresh"]) > float(result["track_high_thresh"]):
        raise ValueError("track_low_thresh cannot exceed track_high_thresh")
    if float(result["new_track_thresh"]) < float(result["track_low_thresh"]):
        raise ValueError("new_track_thresh cannot be below track_low_thresh")
    return result


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    from ultralytics.utils import YAML

    payload = YAML.load(str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def write_json_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
