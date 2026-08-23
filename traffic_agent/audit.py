from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AUDIT_REQUIRED_COLUMNS = {
    "frame",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
}

VEHICLE_CLASS_NAMES = {"vehicle", "car", "motorcycle", "bus", "truck", "lgv", "hgv"}

FAILURE_COLUMNS = [
    "event_type",
    "track_id",
    "other_track_id",
    "start_frame",
    "end_frame",
    "gap_frames",
    "evidence_frames",
    "mean_iou",
    "distance_px",
    "class_name",
    "class_consistency",
    "details",
]


@dataclass(frozen=True)
class AuditThresholds:
    duplicate_iou: float = 0.90
    persistent_duplicate_frames: int = 5
    boundary_margin_px: float = 50.0
    handoff_max_gap_frames: int = 10
    handoff_distance_scale: float = 1.50
    handoff_min_distance_px: float = 35.0
    unstable_class_consistency: float = 0.90
    stable_track_seconds: float = 1.0
    display_confidence: float = 0.20

    def validate(self) -> None:
        if not 0.0 < self.duplicate_iou <= 1.0:
            raise ValueError("duplicate_iou must be in (0, 1]")
        if self.persistent_duplicate_frames < 1:
            raise ValueError("persistent_duplicate_frames must be positive")
        if self.boundary_margin_px < 0:
            raise ValueError("boundary_margin_px cannot be negative")
        if self.handoff_max_gap_frames < 0:
            raise ValueError("handoff_max_gap_frames cannot be negative")
        if self.handoff_distance_scale <= 0 or self.handoff_min_distance_px <= 0:
            raise ValueError("handoff distance thresholds must be positive")
        if not 0.0 < self.unstable_class_consistency <= 1.0:
            raise ValueError("unstable_class_consistency must be in (0, 1]")
        if self.stable_track_seconds <= 0:
            raise ValueError("stable_track_seconds must be positive")
        if not 0.0 <= self.display_confidence <= 1.0:
            raise ValueError("display_confidence must be in [0, 1]")


def _validate_inputs(raw: pd.DataFrame, fps: float, frame_width: int, frame_height: int) -> pd.DataFrame:
    missing = AUDIT_REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing tracking-audit columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("Cannot audit an empty tracking table")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")

    work = raw.copy()
    numeric_columns = [
        "frame",
        "track_id",
        "class_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
    ]
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="raise")
        if not np.isfinite(work[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {column} contains non-finite values")

    if "observed" not in work.columns:
        work["observed"] = True
    elif not pd.api.types.is_bool_dtype(work["observed"]):
        normalized = work["observed"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if normalized.isna().any():
            invalid = sorted(work.loc[normalized.isna(), "observed"].astype(str).unique())
            raise ValueError(f"Unsupported observed values: {invalid}")
        work["observed"] = normalized.astype(bool)

    work["confidence"] = pd.to_numeric(work["confidence"], errors="raise")
    observed_confidence = work.loc[work["observed"], "confidence"].to_numpy(dtype=float)
    if not np.isfinite(observed_confidence).all():
        raise ValueError("Observed confidence contains non-finite values")
    predicted_confidence = work.loc[~work["observed"], "confidence"].dropna().to_numpy(dtype=float)
    if not np.isfinite(predicted_confidence).all():
        raise ValueError("Predicted confidence contains non-finite non-null values")

    if work["class_name"].isna().any():
        raise ValueError("class_name contains null values")
    if ((work["x2"] <= work["x1"]) | (work["y2"] <= work["y1"])).any():
        raise ValueError("Tracking table contains invalid bounding boxes")
    if (work["frame"] < 0).any():
        raise ValueError("frame values cannot be negative")

    work["frame"] = work["frame"].astype(int)
    work["track_id"] = work["track_id"].astype(int)
    work["class_id"] = work["class_id"].astype(int)
    work["class_name"] = work["class_name"].astype(str).str.lower()
    return work.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def _box_iou(first: pd.Series, second: pd.Series) -> float:
    intersection_x1 = max(float(first["x1"]), float(second["x1"]))
    intersection_y1 = max(float(first["y1"]), float(second["y1"]))
    intersection_x2 = min(float(first["x2"]), float(second["x2"]))
    intersection_y2 = min(float(first["y2"]), float(second["y2"]))
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(
        0.0, intersection_y2 - intersection_y1
    )
    first_area = (float(first["x2"]) - float(first["x1"])) * (
        float(first["y2"]) - float(first["y1"])
    )
    second_area = (float(second["x2"]) - float(second["x1"])) * (
        float(second["y2"]) - float(second["y1"])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _mode_with_consistency(group: pd.DataFrame) -> tuple[int, str, float]:
    class_id = int(group["class_id"].mode().iloc[0])
    names = group.loc[group["class_id"] == class_id, "class_name"]
    class_name = str(names.mode().iloc[0])
    consistency = float((group["class_id"] == class_id).mean())
    return class_id, class_name, consistency


def _track_descriptors(raw: pd.DataFrame, fps: float) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for track_id, group in raw.groupby("track_id", sort=False):
        ordered = group.sort_values("frame", kind="stable").drop_duplicates("frame", keep="last")
        class_id, class_name, consistency = _mode_with_consistency(group)
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        start_frame = int(first["frame"])
        end_frame = int(last["frame"])
        observations = int(ordered["frame"].nunique())
        descriptors.append(
            {
                "track_id": int(track_id),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "observations": observations,
                "span_frames": end_frame - start_frame + 1,
                "duration_s": (end_frame - start_frame + 1) / fps,
                "observed_ratio": observations / (end_frame - start_frame + 1),
                "class_id": class_id,
                "class_name": class_name,
                "class_consistency": consistency,
                "first": first,
                "last": last,
            }
        )
    return descriptors


def _inside_boundary(row: pd.Series, width: int, height: int, margin: float) -> bool:
    return bool(
        float(row["x1"]) > margin
        and float(row["y1"]) > margin
        and float(row["x2"]) < width - margin
        and float(row["y2"]) < height - margin
    )


def _duplicate_pairs(
    raw: pd.DataFrame, thresholds: AuditThresholds
) -> tuple[list[dict[str, Any]], int, int]:
    pair_evidence: dict[tuple[int, int], dict[str, Any]] = {}
    overlapping_instances = 0
    overlapping_frames: set[int] = set()

    for frame, group in raw.groupby("frame", sort=False):
        records = [row for _, row in group.iterrows()]
        for first_index, first in enumerate(records):
            for second in records[first_index + 1 :]:
                first_id = int(first["track_id"])
                second_id = int(second["track_id"])
                if first_id == second_id:
                    continue
                overlap = _box_iou(first, second)
                if overlap < thresholds.duplicate_iou:
                    continue
                overlapping_instances += 1
                overlapping_frames.add(int(frame))
                pair = tuple(sorted((first_id, second_id)))
                evidence = pair_evidence.setdefault(
                    pair,
                    {
                        "track_id": pair[0],
                        "other_track_id": pair[1],
                        "start_frame": int(frame),
                        "end_frame": int(frame),
                        "evidence_frames": 0,
                        "iou_sum": 0.0,
                        "max_iou": 0.0,
                    },
                )
                evidence["start_frame"] = min(evidence["start_frame"], int(frame))
                evidence["end_frame"] = max(evidence["end_frame"], int(frame))
                evidence["evidence_frames"] += 1
                evidence["iou_sum"] += overlap
                evidence["max_iou"] = max(evidence["max_iou"], overlap)

    pairs: list[dict[str, Any]] = []
    for evidence in pair_evidence.values():
        evidence_frames = int(evidence["evidence_frames"])
        pairs.append(
            {
                "track_id": int(evidence["track_id"]),
                "other_track_id": int(evidence["other_track_id"]),
                "start_frame": int(evidence["start_frame"]),
                "end_frame": int(evidence["end_frame"]),
                "evidence_frames": evidence_frames,
                "mean_iou": float(evidence["iou_sum"] / evidence_frames),
                "max_iou": float(evidence["max_iou"]),
            }
        )
    pairs.sort(key=lambda item: (-item["evidence_frames"], -item["mean_iou"]))
    return pairs, overlapping_instances, len(overlapping_frames)


def _handoff_candidates(
    descriptors: list[dict[str, Any]], thresholds: AuditThresholds
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for old in descriptors:
        for new in descriptors:
            if old["track_id"] == new["track_id"]:
                continue
            if old["start_frame"] >= new["start_frame"]:
                continue
            gap = int(new["start_frame"] - old["end_frame"])
            if gap < 0 or gap > thresholds.handoff_max_gap_frames:
                continue
            if old["class_name"] not in VEHICLE_CLASS_NAMES or new["class_name"] not in VEHICLE_CLASS_NAMES:
                continue

            old_last = old["last"]
            new_first = new["first"]
            distance = math.hypot(
                float(old_last["center_x"]) - float(new_first["center_x"]),
                float(old_last["center_y"]) - float(new_first["center_y"]),
            )
            old_diagonal = math.hypot(
                float(old_last["x2"] - old_last["x1"]),
                float(old_last["y2"] - old_last["y1"]),
            )
            new_diagonal = math.hypot(
                float(new_first["x2"] - new_first["x1"]),
                float(new_first["y2"] - new_first["y1"]),
            )
            distance_limit = thresholds.handoff_distance_scale * max(
                thresholds.handoff_min_distance_px, old_diagonal, new_diagonal
            )
            if distance > distance_limit:
                continue
            candidates.append(
                {
                    "track_id": int(old["track_id"]),
                    "other_track_id": int(new["track_id"]),
                    "start_frame": int(old["end_frame"]),
                    "end_frame": int(new["start_frame"]),
                    "gap_frames": gap,
                    "distance_px": distance,
                    "old_class_name": str(old["class_name"]),
                    "new_class_name": str(new["class_name"]),
                }
            )
    candidates.sort(key=lambda item: (item["gap_frames"], item["distance_px"]))
    return candidates


def audit_tracks(
    raw: pd.DataFrame,
    *,
    fps: float,
    frame_width: int,
    frame_height: int,
    thresholds: AuditThresholds | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit tracker output without claiming ground-truth accuracy."""
    active_thresholds = thresholds or AuditThresholds()
    active_thresholds.validate()
    work = _validate_inputs(raw, fps, frame_width, frame_height)
    observed_work = work[work["observed"]].copy()
    if observed_work.empty:
        raise ValueError("Cannot audit a tracking table without observed detections")
    descriptors = _track_descriptors(observed_work, fps)
    first_frame = int(observed_work["frame"].min())
    last_frame = int(observed_work["frame"].max())

    duplicate_observations = int(
        observed_work.duplicated(["frame", "track_id"], keep=False).sum()
    )
    duplicate_pairs, overlapping_instances, overlapping_frame_count = _duplicate_pairs(
        observed_work, active_thresholds
    )
    persistent_pairs = [
        pair
        for pair in duplicate_pairs
        if pair["evidence_frames"] >= active_thresholds.persistent_duplicate_frames
    ]
    handoffs = _handoff_candidates(descriptors, active_thresholds)

    later_starts = [item for item in descriptors if item["start_frame"] > first_frame + 2]
    early_ends = [item for item in descriptors if item["end_frame"] < last_frame - 2]
    internal_starts = [
        item
        for item in later_starts
        if _inside_boundary(item["first"], frame_width, frame_height, active_thresholds.boundary_margin_px)
    ]
    internal_ends = [
        item
        for item in early_ends
        if _inside_boundary(item["last"], frame_width, frame_height, active_thresholds.boundary_margin_px)
    ]
    unstable_tracks = [
        item
        for item in descriptors
        if item["class_consistency"] < active_thresholds.unstable_class_consistency
    ]
    class_changing_tracks = [item for item in descriptors if item["class_consistency"] < 1.0]

    stable_observations = max(1, int(math.ceil(active_thresholds.stable_track_seconds * fps)))
    stable_ids = {
        int(item["track_id"]) for item in descriptors if item["observations"] >= stable_observations
    }
    displayed = observed_work[
        observed_work["track_id"].isin(stable_ids)
        & (observed_work["confidence"] >= active_thresholds.display_confidence)
    ]
    all_frames = pd.Index(range(first_frame, last_frame + 1), name="frame")
    raw_counts = observed_work.groupby("frame").size().reindex(all_frames, fill_value=0)
    displayed_counts = displayed.groupby("frame").size().reindex(all_frames, fill_value=0)

    class_counts: dict[str, int] = {}
    for item in descriptors:
        class_name = str(item["class_name"])
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    failure_rows: list[dict[str, Any]] = []
    for pair in persistent_pairs:
        failure_rows.append(
            {
                "event_type": "persistent_duplicate",
                **pair,
                "details": "Different IDs occupy nearly the same box across multiple frames",
            }
        )
    for candidate in handoffs:
        failure_rows.append(
            {
                "event_type": "suspected_id_handoff",
                **candidate,
                "details": f"{candidate['old_class_name']} -> {candidate['new_class_name']}",
            }
        )
    for item in internal_starts:
        failure_rows.append(
            {
                "event_type": "internal_track_start",
                "track_id": item["track_id"],
                "start_frame": item["start_frame"],
                "end_frame": item["start_frame"],
                "class_name": item["class_name"],
                "details": "Track begins away from the frame boundary",
            }
        )
    for item in internal_ends:
        failure_rows.append(
            {
                "event_type": "internal_track_end",
                "track_id": item["track_id"],
                "start_frame": item["end_frame"],
                "end_frame": item["end_frame"],
                "class_name": item["class_name"],
                "details": "Track ends away from the frame boundary",
            }
        )
    for item in unstable_tracks:
        failure_rows.append(
            {
                "event_type": "class_instability",
                "track_id": item["track_id"],
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "class_name": item["class_name"],
                "class_consistency": item["class_consistency"],
                "details": "Raw per-frame class agreement is below the configured threshold",
            }
        )
    failures = pd.DataFrame(failure_rows).reindex(columns=FAILURE_COLUMNS)

    critical_flags = {
        "duplicate_track_observations": duplicate_observations > 0,
        "persistent_duplicate_tracks": bool(persistent_pairs),
        "suspected_id_handoffs": bool(handoffs),
    }
    review_flags = {
        "internal_track_starts": bool(internal_starts),
        "internal_track_ends": bool(internal_ends),
        "unstable_classes": bool(unstable_tracks),
        "evidence_hides_observations": len(displayed) < len(observed_work),
    }
    if any(critical_flags.values()):
        audit_status = "failed"
    elif any(review_flags.values()):
        audit_status = "needs_review"
    else:
        audit_status = "clear"

    durations = np.asarray([item["duration_s"] for item in descriptors], dtype=float)
    consistencies = np.asarray([item["class_consistency"] for item in descriptors], dtype=float)
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_status": audit_status,
        "verification_status": "not_ground_truth_verified",
        "source": {
            "fps": float(fps),
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "first_frame": first_frame,
            "last_frame": last_frame,
            "audited_frames": last_frame - first_frame + 1,
        },
        "thresholds": asdict(active_thresholds),
        "metrics": {
            "input_rows": int(len(work)),
            "observed_rows": int(len(observed_work)),
            "predicted_rows": int((~work["observed"]).sum()),
            "raw_rows": int(len(observed_work)),
            "raw_unique_ids": int(len(descriptors)),
            "unique_ids_by_modal_class": dict(sorted(class_counts.items())),
            "median_track_duration_s": float(np.median(durations)),
            "tracks_under_1s": int((durations < 1.0).sum()),
            "tracks_under_2s": int((durations < 2.0).sum()),
            "median_raw_class_consistency": float(np.median(consistencies)),
            "class_changing_tracks": int(len(class_changing_tracks)),
            "unstable_class_tracks": int(len(unstable_tracks)),
            "duplicate_track_frame_rows": duplicate_observations,
            "overlapping_different_id_instances": overlapping_instances,
            "frames_with_overlapping_ids": overlapping_frame_count,
            "persistent_duplicate_pairs": int(len(persistent_pairs)),
            "suspected_id_handoffs": int(len(handoffs)),
            "internal_track_starts": int(len(internal_starts)),
            "internal_track_ends": int(len(internal_ends)),
            "internal_start_ratio": float(len(internal_starts) / len(later_starts)) if later_starts else 0.0,
            "internal_end_ratio": float(len(internal_ends) / len(early_ends)) if early_ends else 0.0,
        },
        "evidence_filter_audit": {
            "stable_track_ids": int(len(stable_ids)),
            "displayed_rows": int(len(displayed)),
            "display_share": float(len(displayed) / len(observed_work)),
            "median_raw_objects_per_frame": float(raw_counts.median()),
            "median_displayed_objects_per_frame": float(displayed_counts.median()),
        },
        "critical_flags": critical_flags,
        "review_flags": review_flags,
        "persistent_duplicate_examples": persistent_pairs[:20],
        "handoff_examples": handoffs[:20],
        "notes": [
            "This audit detects internal inconsistencies and does not replace labelled detection/tracking evaluation.",
            "A passing ground-truth gate must report detection precision/recall plus IDF1 and HOTA.",
        ],
    }
    return report, failures


def write_tracking_audit(
    raw_csv: Path,
    output_dir: Path,
    *,
    fps: float,
    frame_width: int,
    frame_height: int,
    thresholds: AuditThresholds | None = None,
) -> dict[str, Any]:
    if not raw_csv.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {raw_csv}")
    raw = pd.read_csv(raw_csv)
    report, failures = audit_tracks(
        raw,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
        thresholds=thresholds,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "tracking_audit.json"
    failures_path = output_dir / "tracking_failures.csv"
    report["outputs"] = {
        "tracking_audit": report_path.name,
        "tracking_failures": failures_path.name,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failures.to_csv(failures_path, index=False)
    return report
