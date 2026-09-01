from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .cvat import SEED_CLASS_MAP


TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
MOT_COLUMNS = (
    "frame",
    "track_id",
    "x",
    "y",
    "width",
    "height",
    "not_ignored",
    "class_id",
    "visibility",
)
REQUIRED_PREDICTION_COLUMNS = {
    "frame",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "observed",
}


@contextmanager
def _trackeval_numpy_compatibility() -> Any:
    """Temporarily restore aliases required by the pinned official TrackEval.

    TrackEval's latest upstream commit predates NumPy 1.24 and its HOTA and
    Identity implementations still reference ``np.float`` and ``np.int``.
    Keep the compatibility boundary local to the metric call and remove only
    aliases that this function added.
    """
    added: list[str] = []
    for name, value in (("float", float), ("int", int)):
        if name not in np.__dict__:
            setattr(np, name, value)
            added.append(name)
    try:
        yield
    finally:
        for name in added:
            delattr(np, name)


@dataclass(frozen=True)
class QualityGates:
    detection_precision: float = 0.90
    detection_recall: float = 0.90
    idf1: float = 0.85
    hota: float = 0.70
    mode_accuracy: float = 0.80

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} gate must be in [0, 1]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.lower().endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {suffix} in MOT archive, found {len(matches)}"
        )
    return matches[0]


def read_cvat_mot_archive(path: Path) -> pd.DataFrame:
    """Read the official CVAT MOT export structure into validated boxes."""
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        gt_member = _zip_member(archive, "gt/gt.txt")
        labels_member = _zip_member(archive, "gt/labels.txt")
        labels = [
            line.strip()
            for line in archive.read(labels_member).decode("utf-8-sig").splitlines()
            if line.strip()
        ]
        if not labels:
            raise ValueError("MOT labels.txt cannot be empty")
        gt_text = archive.read(gt_member).decode("utf-8-sig")

    ground_truth = pd.read_csv(io.StringIO(gt_text), header=None, comment="#")
    if ground_truth.shape[1] < len(MOT_COLUMNS):
        raise ValueError(
            f"MOT gt.txt needs at least {len(MOT_COLUMNS)} columns, "
            f"found {ground_truth.shape[1]}"
        )
    ground_truth = ground_truth.iloc[:, : len(MOT_COLUMNS)].copy()
    ground_truth.columns = MOT_COLUMNS
    numeric_columns = set(MOT_COLUMNS) - {"visibility"}
    for column in numeric_columns:
        ground_truth[column] = pd.to_numeric(ground_truth[column], errors="raise")
    ground_truth["visibility"] = pd.to_numeric(
        ground_truth["visibility"], errors="coerce"
    )
    ground_truth["frame"] = ground_truth["frame"].astype(int)
    ground_truth["track_id"] = ground_truth["track_id"].astype(int)
    ground_truth["class_id"] = ground_truth["class_id"].astype(int)
    if (ground_truth["frame"] < 1).any():
        raise ValueError("MOT frames must be one-based positive integers")
    if (ground_truth["track_id"] < 1).any():
        raise ValueError("MOT track IDs must be positive integers")
    if (ground_truth["width"] <= 0).any() or (ground_truth["height"] <= 0).any():
        raise ValueError("MOT boxes must have positive width and height")
    if (ground_truth["class_id"] < 1).any() or (
        ground_truth["class_id"] > len(labels)
    ).any():
        raise ValueError("MOT class IDs must reference labels.txt")
    ground_truth["class_name"] = ground_truth["class_id"].map(
        lambda class_id: labels[int(class_id) - 1].strip().lower()
    )
    ground_truth["x1"] = ground_truth["x"].astype(float)
    ground_truth["y1"] = ground_truth["y"].astype(float)
    ground_truth["x2"] = ground_truth["x1"] + ground_truth["width"].astype(float)
    ground_truth["y2"] = ground_truth["y1"] + ground_truth["height"].astype(float)
    ground_truth["ignored"] = ground_truth["not_ignored"].astype(float) <= 0
    return ground_truth


def _normalize_mode(value: object) -> str:
    name = str(value).strip().lower()
    return SEED_CLASS_MAP.get(name, name)


def prepare_predictions(
    tracks: pd.DataFrame, summaries: pd.DataFrame, *, frame_count: int
) -> pd.DataFrame:
    missing = REQUIRED_PREDICTION_COLUMNS - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks is missing required columns: {sorted(missing)}")
    summary_columns = {"track_id", "final_class_name"}
    summary_missing = summary_columns - set(summaries.columns)
    if summary_missing:
        raise ValueError(
            f"track_summary is missing required columns: {sorted(summary_missing)}"
        )
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    predictions = tracks.copy()
    predictions["frame"] = pd.to_numeric(
        predictions["frame"], errors="raise"
    ).astype(int) + 1
    predictions["track_id"] = pd.to_numeric(
        predictions["track_id"], errors="raise"
    ).astype(int)
    for column in ("x1", "y1", "x2", "y2"):
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    if (predictions["frame"] < 1).any() or (
        predictions["frame"] > frame_count
    ).any():
        raise ValueError("Prediction frames lie outside the evaluated segment")
    invalid = (predictions["x2"] <= predictions["x1"]) | (
        predictions["y2"] <= predictions["y1"]
    )
    if invalid.any():
        raise ValueError(f"tracks contains {int(invalid.sum())} invalid boxes")
    if predictions.duplicated(["frame", "track_id"]).any():
        raise ValueError("tracks contains duplicate frame/track_id rows")

    summary = summaries[["track_id", "final_class_name"]].copy()
    summary["track_id"] = pd.to_numeric(summary["track_id"], errors="raise").astype(int)
    summary["summary_class_name"] = summary["final_class_name"].map(_normalize_mode)
    predictions = predictions.merge(
        summary[["track_id", "summary_class_name"]],
        on="track_id",
        how="left",
        validate="many_to_one",
    )
    if predictions["summary_class_name"].isna().any():
        missing_ids = sorted(
            int(value)
            for value in predictions.loc[
                predictions["summary_class_name"].isna(), "track_id"
            ].unique()
        )
        raise ValueError(f"track_summary is missing IDs: {missing_ids[:10]}")
    predictions["class_name"] = predictions.pop("summary_class_name")
    return predictions.sort_values(["frame", "track_id"]).reset_index(drop=True)


def _box_iou_matrix(gt_boxes: np.ndarray, tracker_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=float)
    intersection_x1 = np.maximum(gt_boxes[:, None, 0], tracker_boxes[None, :, 0])
    intersection_y1 = np.maximum(gt_boxes[:, None, 1], tracker_boxes[None, :, 1])
    intersection_x2 = np.minimum(gt_boxes[:, None, 2], tracker_boxes[None, :, 2])
    intersection_y2 = np.minimum(gt_boxes[:, None, 3], tracker_boxes[None, :, 3])
    intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
        0.0, intersection_y2 - intersection_y1
    )
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
        gt_boxes[:, 3] - gt_boxes[:, 1]
    )
    tracker_area = (tracker_boxes[:, 2] - tracker_boxes[:, 0]) * (
        tracker_boxes[:, 3] - tracker_boxes[:, 1]
    )
    union = gt_area[:, None] + tracker_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0,
    )


def prepare_trackeval_sequence(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    frame_count: int,
) -> dict[str, Any]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    gt = ground_truth[
        (~ground_truth["ignored"].astype(bool))
        & (ground_truth["class_name"].map(_normalize_mode) != "ignore")
    ].copy()
    if (gt["frame"] > frame_count).any():
        raise ValueError("Ground-truth frames lie outside the evaluated segment")

    gt_id_map = {
        int(track_id): index
        for index, track_id in enumerate(sorted(gt["track_id"].unique()))
    }
    tracker_id_map = {
        int(track_id): index
        for index, track_id in enumerate(sorted(predictions["track_id"].unique()))
    }
    gt_ids: list[np.ndarray] = []
    tracker_ids: list[np.ndarray] = []
    similarity_scores: list[np.ndarray] = []
    gt_dets: list[np.ndarray] = []
    tracker_dets: list[np.ndarray] = []

    for frame in range(1, frame_count + 1):
        gt_frame = gt[gt["frame"] == frame]
        tracker_frame = predictions[predictions["frame"] == frame]
        current_gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].to_numpy(float)
        current_tracker_boxes = tracker_frame[
            ["x1", "y1", "x2", "y2"]
        ].to_numpy(float)
        gt_dets.append(current_gt_boxes)
        tracker_dets.append(current_tracker_boxes)
        gt_ids.append(
            np.asarray(
                [gt_id_map[int(value)] for value in gt_frame["track_id"]],
                dtype=int,
            )
        )
        tracker_ids.append(
            np.asarray(
                [tracker_id_map[int(value)] for value in tracker_frame["track_id"]],
                dtype=int,
            )
        )
        similarity_scores.append(
            _box_iou_matrix(current_gt_boxes, current_tracker_boxes)
        )

    return {
        "num_timesteps": frame_count,
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(tracker_id_map),
        "num_gt_dets": int(sum(len(values) for values in gt_ids)),
        "num_tracker_dets": int(sum(len(values) for values in tracker_ids)),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "gt_dets": gt_dets,
        "tracker_dets": tracker_dets,
        "similarity_scores": similarity_scores,
    }


def mode_classification_metrics(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    gt = ground_truth[
        (~ground_truth["ignored"].astype(bool))
        & (ground_truth["class_name"].map(_normalize_mode) != "ignore")
    ].copy()
    confusion: dict[str, dict[str, int]] = {}
    matched = 0
    correct = 0
    for frame in sorted(set(gt["frame"]) | set(predictions["frame"])):
        gt_frame = gt[gt["frame"] == frame]
        pred_frame = predictions[predictions["frame"] == frame]
        similarity = _box_iou_matrix(
            gt_frame[["x1", "y1", "x2", "y2"]].to_numpy(float),
            pred_frame[["x1", "y1", "x2", "y2"]].to_numpy(float),
        )
        if similarity.size == 0:
            continue
        rows, columns = linear_sum_assignment(1.0 - similarity)
        for row_index, column_index in zip(rows, columns):
            if similarity[row_index, column_index] < iou_threshold:
                continue
            gt_name = _normalize_mode(gt_frame.iloc[row_index]["class_name"])
            pred_name = _normalize_mode(pred_frame.iloc[column_index]["class_name"])
            predictions_by_name = confusion.setdefault(gt_name, {})
            predictions_by_name[pred_name] = predictions_by_name.get(pred_name, 0) + 1
            matched += 1
            correct += int(gt_name == pred_name)
    return {
        "matched_boxes": matched,
        "correct_mode_boxes": correct,
        "mode_accuracy": correct / matched if matched else 0.0,
        "confusion": {
            gt_name: dict(sorted(predictions_by_name.items()))
            for gt_name, predictions_by_name in sorted(confusion.items())
        },
    }


def detection_error_diagnostics(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    iou_threshold: float = 0.5,
    grid_columns: int = 4,
    grid_rows: int = 4,
    example_limit: int = 25,
) -> dict[str, Any]:
    """Explain category-agnostic detection errors without changing gate metrics."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Diagnostic frame dimensions must be positive")
    if grid_columns < 1 or grid_rows < 1 or example_limit < 0:
        raise ValueError("Diagnostic grid dimensions must be positive")
    gt = ground_truth[
        (~ground_truth["ignored"].astype(bool))
        & (ground_truth["class_name"].map(_normalize_mode) != "ignore")
    ].copy()
    gt["class_name"] = gt["class_name"].map(_normalize_mode)
    predictions = predictions.copy()
    predictions["class_name"] = predictions["class_name"].map(_normalize_mode)
    class_counts: dict[str, dict[str, int]] = {}
    spatial_counts: dict[str, dict[str, int]] = {}
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []

    def increment_class(name: str, field: str) -> None:
        bucket = class_counts.setdefault(name, {"tp": 0, "fn": 0, "fp": 0})
        bucket[field] += 1

    def grid_key(row: pd.Series) -> str:
        center_x = (float(row["x1"]) + float(row["x2"])) / 2.0
        ground_y = float(row["y2"])
        column = min(grid_columns - 1, max(0, int(center_x / frame_width * grid_columns)))
        grid_row = min(grid_rows - 1, max(0, int(ground_y / frame_height * grid_rows)))
        return f"r{grid_row}c{column}"

    def increment_grid(row: pd.Series, field: str) -> None:
        bucket = spatial_counts.setdefault(
            grid_key(row), {"tp": 0, "fn": 0, "fp": 0}
        )
        bucket[field] += 1

    for frame in range(1, frame_count + 1):
        gt_frame = gt[gt["frame"] == frame].reset_index(drop=True)
        pred_frame = predictions[predictions["frame"] == frame].reset_index(drop=True)
        similarity = _box_iou_matrix(
            gt_frame[["x1", "y1", "x2", "y2"]].to_numpy(float),
            pred_frame[["x1", "y1", "x2", "y2"]].to_numpy(float),
        )
        matched_gt: set[int] = set()
        matched_predictions: set[int] = set()
        if similarity.size:
            rows, columns = linear_sum_assignment(-similarity)
            for row_index, column_index in zip(rows, columns):
                if similarity[row_index, column_index] < iou_threshold:
                    continue
                matched_gt.add(int(row_index))
                matched_predictions.add(int(column_index))
                gt_row = gt_frame.iloc[row_index]
                increment_class(str(gt_row["class_name"]), "tp")
                increment_grid(gt_row, "tp")
        for row_index, row in gt_frame.iterrows():
            if int(row_index) in matched_gt:
                continue
            increment_class(str(row["class_name"]), "fn")
            increment_grid(row, "fn")
            if len(false_negatives) < example_limit:
                false_negatives.append(
                    {
                        "frame": frame - 1,
                        "track_id": int(row["track_id"]),
                        "class_name": str(row["class_name"]),
                        "box": [round(float(row[name]), 3) for name in ("x1", "y1", "x2", "y2")],
                    }
                )
        for row_index, row in pred_frame.iterrows():
            if int(row_index) in matched_predictions:
                continue
            increment_class(str(row["class_name"]), "fp")
            increment_grid(row, "fp")
            if len(false_positives) < example_limit:
                confidence = pd.to_numeric(pd.Series([row.get("confidence")]), errors="coerce").iloc[0]
                false_positives.append(
                    {
                        "frame": frame - 1,
                        "track_id": int(row["track_id"]),
                        "class_name": str(row["class_name"]),
                        "confidence": None if pd.isna(confidence) else round(float(confidence), 6),
                        "box": [round(float(row[name]), 3) for name in ("x1", "y1", "x2", "y2")],
                    }
                )
    return {
        "matching_iou": iou_threshold,
        "per_class": dict(sorted(class_counts.items())),
        "spatial_grid": {
            "columns": grid_columns,
            "rows": grid_rows,
            "cells": dict(sorted(spatial_counts.items())),
        },
        "representative_false_negatives": false_negatives,
        "representative_false_positives": false_positives,
    }


def detection_cache_metrics(
    ground_truth: pd.DataFrame,
    detections: pd.DataFrame,
    *,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    minimum_confidence: float = 0.0,
) -> dict[str, Any]:
    required = {"frame", "class_name", "confidence", "x1", "y1", "x2", "y2"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"Detection cache is missing columns: {sorted(missing)}")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    predictions = detections.copy()
    predictions["confidence"] = pd.to_numeric(
        predictions["confidence"], errors="raise"
    )
    predictions = predictions[predictions["confidence"] >= minimum_confidence].copy()
    predictions["frame"] = pd.to_numeric(predictions["frame"], errors="raise").astype(int) + 1
    if (predictions["frame"] < 1).any() or (predictions["frame"] > frame_count).any():
        raise ValueError("Detection cache frames lie outside the evaluated segment")
    predictions["track_id"] = np.arange(1, len(predictions) + 1, dtype=int)
    predictions["observed"] = True
    diagnostics = detection_error_diagnostics(
        ground_truth,
        predictions,
        frame_count=frame_count,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    true_positives = sum(
        int(values["tp"]) for values in diagnostics["per_class"].values()
    )
    gt_boxes = int(
        (
            (~ground_truth["ignored"].astype(bool))
            & (ground_truth["class_name"].map(_normalize_mode) != "ignore")
        ).sum()
    )
    prediction_boxes = len(predictions)
    return {
        "minimum_confidence": minimum_confidence,
        "detection_precision": (
            true_positives / prediction_boxes if prediction_boxes else 0.0
        ),
        "detection_recall": true_positives / gt_boxes if gt_boxes else 0.0,
        "matched_boxes": true_positives,
        "ground_truth_boxes": gt_boxes,
        "prediction_boxes": prediction_boxes,
        "error_diagnostics": diagnostics,
    }


def _pipeline_stage_diagnostics(
    manifest: dict[str, Any], run_manifest_path: Path, tracks: pd.DataFrame
) -> dict[str, Any]:
    outputs = manifest.get("outputs", {})
    run_directory = run_manifest_path.parent

    candidate_relative = outputs.get("candidate_detections")
    expected_candidate_hash = str(manifest.get("candidate_detections_sha256") or "")
    if candidate_relative and expected_candidate_hash:
        candidate_path = run_directory / str(candidate_relative)
        if not candidate_path.exists() or sha256_file(candidate_path) != expected_candidate_hash:
            raise ValueError(
                "candidate_detections.csv hash does not match run_manifest.json"
            )

    def read_output(name: str) -> pd.DataFrame | None:
        relative = outputs.get(name)
        if not relative:
            return None
        path = run_directory / str(relative)
        return pd.read_csv(path) if path.exists() else None

    candidates = read_output("candidate_detections")
    accepted = read_output("detections")
    native = read_output("native_tracks")
    observed_final = int(tracks["observed"].astype(str).str.lower().eq("true").sum())
    observed_native = (
        int(native["observed"].astype(str).str.lower().eq("true").sum())
        if native is not None
        else None
    )
    counts = {
        "candidate_detections": len(candidates) if candidates is not None else None,
        "accepted_detections": len(accepted) if accepted is not None else None,
        "native_observed_rows": observed_native,
        "confirmed_observed_rows": observed_final,
        "final_prediction_rows": len(tracks) - observed_final,
    }
    losses = {
        "candidate_to_accepted": (
            len(candidates) - len(accepted)
            if candidates is not None and accepted is not None
            else None
        ),
        "accepted_to_native_observed": (
            len(accepted) - observed_native
            if accepted is not None and observed_native is not None
            else None
        ),
        "native_to_confirmed_observed": (
            observed_native - observed_final if observed_native is not None else None
        ),
    }
    return {"stage_counts": counts, "stage_losses": losses}


def trackeval_metrics(sequence: dict[str, Any]) -> dict[str, float | int]:
    if sequence["num_gt_dets"] == 0 or sequence["num_tracker_dets"] == 0:
        return {
            "detection_precision": 0.0,
            "detection_recall": 0.0,
            "idf1": 0.0,
            "hota": 0.0,
            "deta": 0.0,
            "assa": 0.0,
            "mota": 0.0,
            "id_switches": 0,
            "fragmentations": 0,
        }
    try:
        import trackeval
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "TrackEval is required for formal metrics. Install requirements-eval.txt."
        ) from error

    with _trackeval_numpy_compatibility():
        hota_result = trackeval.metrics.HOTA().eval_sequence(sequence)
        identity_result = trackeval.metrics.Identity(
            {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
        ).eval_sequence(sequence)
        clear_result = trackeval.metrics.CLEAR(
            {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
        ).eval_sequence(sequence)
    return {
        "detection_precision": float(clear_result["CLR_Pr"]),
        "detection_recall": float(clear_result["CLR_Re"]),
        "idf1": float(identity_result["IDF1"]),
        "hota": float(np.mean(hota_result["HOTA"])),
        "deta": float(np.mean(hota_result["DetA"])),
        "assa": float(np.mean(hota_result["AssA"])),
        "mota": float(clear_result["MOTA"]),
        "id_switches": int(clear_result["IDSW"]),
        "fragmentations": int(clear_result["Frag"]),
    }


def quality_gate_report(
    metrics: dict[str, float | int],
    mode_metrics: dict[str, Any],
    *,
    gates: QualityGates | None = None,
) -> dict[str, Any]:
    active_gates = gates or QualityGates()
    active_gates.validate()
    values = {
        "detection_precision": float(metrics["detection_precision"]),
        "detection_recall": float(metrics["detection_recall"]),
        "idf1": float(metrics["idf1"]),
        "hota": float(metrics["hota"]),
        "mode_accuracy": float(mode_metrics["mode_accuracy"]),
    }
    non_finite = [name for name, value in values.items() if not math.isfinite(value)]
    if non_finite:
        raise RuntimeError(f"Evaluation returned non-finite metrics: {non_finite}")
    thresholds = asdict(active_gates)
    gate_results = {
        name: values[name] >= threshold for name, threshold in thresholds.items()
    }
    passed = all(gate_results.values())
    return {
        "quality_status": "passed" if passed else "failed",
        "gate_thresholds": thresholds,
        "gate_results": gate_results,
        "blocking_reason": None
        if passed
        else "One or more labelled detection/tracking quality gates failed",
    }


def evaluate_v4(
    *,
    mot_archive: Path,
    tracks_path: Path,
    summary_path: Path,
    run_manifest_path: Path,
    output_path: Path,
    diagnostics_output_path: Path | None = None,
    gates: QualityGates | None = None,
) -> dict[str, Any]:
    for path in (mot_archive, tracks_path, summary_path, run_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected_hash = str(manifest.get("tracks_sha256", ""))
    actual_hash = sha256_file(tracks_path)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("tracks.csv hash does not match run_manifest.json")
    frame_count = int(manifest.get("source", {}).get("processed_frames", 0))
    if frame_count <= 0:
        raise ValueError("run_manifest source.processed_frames must be positive")

    ground_truth = read_cvat_mot_archive(mot_archive)
    raw_tracks = pd.read_csv(tracks_path)
    predictions = prepare_predictions(
        raw_tracks,
        pd.read_csv(summary_path),
        frame_count=frame_count,
    )
    sequence = prepare_trackeval_sequence(
        ground_truth, predictions, frame_count=frame_count
    )
    metrics = trackeval_metrics(sequence)
    mode_metrics = mode_classification_metrics(ground_truth, predictions)
    diagnostics = detection_error_diagnostics(
        ground_truth,
        predictions,
        frame_count=frame_count,
        frame_width=int(manifest.get("source", {}).get("width", 0)),
        frame_height=int(manifest.get("source", {}).get("height", 0)),
    )
    diagnostics.update(
        _pipeline_stage_diagnostics(manifest, run_manifest_path, raw_tracks)
    )
    gate_report = quality_gate_report(metrics, mode_metrics, gates=gates)
    report = {
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        **gate_report,
        "ground_truth_metrics": {**metrics, **mode_metrics},
        "counts": {
            "frames": frame_count,
            "ground_truth_tracks": int(sequence["num_gt_ids"]),
            "prediction_tracks": int(predictions["track_id"].nunique()),
            "ground_truth_boxes": int(sequence["num_gt_dets"]),
            "prediction_boxes": int(sequence["num_tracker_dets"]),
        },
        "error_diagnostics": diagnostics,
        "provenance": {
            "mot_archive": str(mot_archive),
            "mot_archive_sha256": sha256_file(mot_archive),
            "tracks": str(tracks_path),
            "tracks_sha256": actual_hash,
            "track_summary": str(summary_path),
            "track_summary_sha256": sha256_file(summary_path),
            "run_manifest": str(run_manifest_path),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "trackeval_commit": TRACKEVAL_COMMIT,
        },
        "evaluation_policy": {
            "matching_iou": 0.5,
            "tracking_scope": "category-agnostic road users",
            "prediction_rows": "observed detections and occluded Kalman predictions",
            "mode_classification": "reported separately after IoU-0.5 matching",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if diagnostics_output_path is not None:
        diagnostics_output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_output_path.write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
    return report
