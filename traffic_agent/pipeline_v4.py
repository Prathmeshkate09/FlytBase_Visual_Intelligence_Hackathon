from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .detection import DetectorConfig, RoadUserDetector
from .lifecycle import LifecycleConfig, TrackLifecycleManager
from .postprocess import (
    Detection,
    PostprocessConfig,
    RejectedDetection,
    RoadUserROI,
    postprocess_detections,
)
from .stitching import StitchingConfig, stitch_tracks
from .tracker import GroupedBoTSORT, TrackObservation


@dataclass(frozen=True)
class PipelineV4Config:
    input_video: Path
    output_dir: Path
    detection_config: Path = Path("config/detection_v4.json")
    tracker_config: Path = Path("config/botsort_drone_v4.yaml")
    road_user_roi: Path | None = None
    exit_roi: Path | None = None
    cached_detections: Path | None = None
    cached_rejected_detections: Path | None = None
    max_seconds: float | None = None
    confirmation_observations: int = 3
    max_prediction_frames: int = 15
    enable_offline_stitching: bool = True

    def validate(self) -> None:
        if not self.input_video.exists():
            raise FileNotFoundError(f"Video not found: {self.input_video}")
        if not self.detection_config.exists():
            raise FileNotFoundError(f"Detection configuration not found: {self.detection_config}")
        if not self.tracker_config.exists():
            raise FileNotFoundError(f"Tracker configuration not found: {self.tracker_config}")
        if self.road_user_roi is not None and not self.road_user_roi.exists():
            raise FileNotFoundError(f"Road-user ROI not found: {self.road_user_roi}")
        if self.exit_roi is not None and not self.exit_roi.exists():
            raise FileNotFoundError(f"Exit ROI not found: {self.exit_roi}")
        if self.cached_detections is not None and not self.cached_detections.exists():
            raise FileNotFoundError(
                f"Cached detections file not found: {self.cached_detections}"
            )
        if (
            self.cached_rejected_detections is not None
            and not self.cached_rejected_detections.exists()
        ):
            raise FileNotFoundError(
                "Cached rejected detections file not found: "
                f"{self.cached_rejected_detections}"
            )
        if (
            self.cached_rejected_detections is not None
            and self.cached_detections is None
        ):
            raise ValueError("cached_rejected_detections requires cached_detections")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if self.max_prediction_frames < 0:
            raise ValueError("max_prediction_frames cannot be negative")
        LifecycleConfig(self.confirmation_observations).validate()


DETECTION_FIELDS = [
    "frame",
    "timestamp_s",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "ground_x",
    "ground_y",
    "association_group",
    "source",
    "source_classes",
    "merge_count",
]

TRACK_FIELDS = [
    "frame",
    "timestamp_s",
    "track_id",
    "source_track_id",
    "native_track_id",
    "association_group",
    "class_id",
    "class_name",
    "detected_class_id",
    "detected_class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "ground_x",
    "ground_y",
    "observed",
    "state",
    "detection_source",
]

REJECTED_PREDICTION_FIELDS = [
    "frame",
    "track_id",
    "native_track_id",
    "association_group",
    "raw_x1",
    "raw_y1",
    "raw_x2",
    "raw_y2",
    "clipped_x1",
    "clipped_y1",
    "clipped_x2",
    "clipped_y2",
    "reason",
]

STITCH_FIELDS = [
    "old_track_id",
    "new_track_id",
    "canonical_track_id",
    "gap_frames",
    "endpoint_distance_px",
    "predicted_distance_px",
    "distance_limit_px",
    "area_ratio",
    "direction_change_degrees",
    "score",
]


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def load_detector_config(path: Path) -> DetectorConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "class_ids" in payload:
        payload["class_ids"] = tuple(int(value) for value in payload["class_ids"])
    config = DetectorConfig(**payload)
    config.validate()
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _detection_row(detection: Detection, frame: int, fps: float) -> dict[str, Any]:
    ground_x, ground_y = detection.ground_point
    return {
        "frame": frame,
        "timestamp_s": round(frame / fps, 6),
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": round(detection.confidence, 6),
        "x1": round(detection.x1, 3),
        "y1": round(detection.y1, 3),
        "x2": round(detection.x2, 3),
        "y2": round(detection.y2, 3),
        "ground_x": round(ground_x, 3),
        "ground_y": round(ground_y, 3),
        "association_group": detection.association_group,
        "source": detection.source,
        "source_classes": "+".join(detection.source_classes),
        "merge_count": detection.merge_count,
    }


def _track_row(observation: TrackObservation, fps: float) -> dict[str, Any]:
    ground_x, ground_y = observation.ground_point
    return {
        "frame": observation.frame,
        "timestamp_s": round(observation.frame / fps, 6),
        "track_id": observation.track_id,
        "source_track_id": observation.track_id,
        "native_track_id": observation.native_track_id,
        "association_group": observation.association_group,
        # Compatibility aliases retain the original per-frame detector vote.
        # Final stabilized class is written separately to track_summary.csv.
        "class_id": observation.detected_class_id,
        "class_name": observation.detected_class_name,
        "detected_class_id": observation.detected_class_id,
        "detected_class_name": observation.detected_class_name,
        "confidence": "" if observation.confidence is None else round(observation.confidence, 6),
        "x1": round(observation.x1, 3),
        "y1": round(observation.y1, 3),
        "x2": round(observation.x2, 3),
        "y2": round(observation.y2, 3),
        "center_x": round((observation.x1 + observation.x2) / 2.0, 3),
        "center_y": round((observation.y1 + observation.y2) / 2.0, 3),
        "ground_x": round(ground_x, 3),
        "ground_y": round(ground_y, 3),
        "observed": observation.observed,
        "state": observation.state,
        "detection_source": observation.detection_source,
    }


def _rejected_row(
    rejected: RejectedDetection, frame: int, fps: float
) -> dict[str, Any]:
    row = _detection_row(rejected.detection, frame, fps)
    row["reason"] = rejected.reason
    return row


def _load_cached_detections(
    path: Path, *, allowed_reasons: set[str] | None = None
) -> dict[int, list[Detection]]:
    required = {
        "frame",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    grouped: dict[int, list[Detection]] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Cached detections are missing columns: {sorted(missing)}"
            )
        for row in reader:
            if allowed_reasons is not None:
                reason = str(row.get("reason") or "")
                if reason not in allowed_reasons:
                    continue
            source_classes = tuple(
                value for value in str(row.get("source_classes", "")).split("+") if value
            )
            detection = Detection(
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                confidence=float(row["confidence"]),
                class_id=int(row["class_id"]),
                class_name=str(row["class_name"]),
                source=str(row.get("source") or "cached"),
                source_classes=source_classes,
                merge_count=int(row.get("merge_count") or 1),
            )
            detection.validate()
            grouped.setdefault(int(row["frame"]), []).append(detection)
    if not grouped and allowed_reasons is None:
        raise ValueError(f"Cached detections file is empty: {path}")
    return grouped


def _extend_cached_detections(
    destination: dict[int, list[Detection]],
    additions: dict[int, list[Detection]],
) -> None:
    for frame, detections in additions.items():
        destination.setdefault(frame, []).extend(detections)


def _apply_confidence_thresholds(
    detections: Iterable[Detection], detector_config: DetectorConfig
) -> tuple[list[Detection], list[RejectedDetection]]:
    accepted: list[Detection] = []
    rejected: list[RejectedDetection] = []
    for detection in detections:
        threshold = detector_config.confidence_threshold_for(detection.class_name)
        if detection.confidence < threshold:
            rejected.append(
                RejectedDetection(detection, "below_class_confidence_threshold")
            )
        else:
            accepted.append(detection)
    return accepted, rejected


def _write_rows(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _final_summary_rows(
    stitched: Any, native_summaries: dict[int, Any]
) -> list[dict[str, Any]]:
    import pandas as pd

    summaries: list[dict[str, Any]] = []
    for final_track_id, group in stitched.groupby("track_id", sort=True):
        observed = group[group["observed"].astype(bool)].copy()
        source_ids = sorted(int(value) for value in group["source_track_id"].unique())
        final_source_id = max(
            source_ids,
            key=lambda value: (
                native_summaries[value].last_frame,
                native_summaries[value].last_observed_frame,
            ),
        )
        if observed.empty:
            final_class_id, final_class_name, final_class_confidence = -1, "unknown", 0.0
            last_observed_frame = -1
        else:
            observed["_class_score"] = pd.to_numeric(
                observed["confidence"], errors="coerce"
            ).fillna(0.0)
            scores = (
                observed.groupby(["detected_class_id", "detected_class_name"])[
                    "_class_score"
                ]
                .sum()
                .sort_values(ascending=False)
            )
            final_class_id, final_class_name = scores.index[0]
            total_score = float(scores.sum())
            final_class_confidence = (
                float(scores.iloc[0] / total_score) if total_score > 0 else 0.0
            )
            last_observed_frame = int(observed["frame"].max())
        summaries.append(
            {
                "track_id": int(final_track_id),
                "state": native_summaries[final_source_id].state.value,
                "first_frame": int(group["frame"].min()),
                "last_frame": int(group["frame"].max()),
                "last_observed_frame": last_observed_frame,
                "observation_count": int(observed["frame"].nunique()),
                "final_class_id": int(final_class_id),
                "final_class_name": str(final_class_name),
                "final_class_confidence": final_class_confidence,
            }
        )
    return summaries


def _confirmed_track_ids(
    summaries: Iterable[dict[str, Any]], minimum_observations: int
) -> set[int]:
    if minimum_observations < 1:
        raise ValueError("minimum_observations must be positive")
    return {
        int(summary["track_id"])
        for summary in summaries
        if int(summary["observation_count"]) >= minimum_observations
    }


def run_pipeline_v4(config: PipelineV4Config) -> dict[str, Any]:
    config = PipelineV4Config(
        **{
            **asdict(config),
            "detection_config": _resolve_project_path(config.detection_config),
            "tracker_config": _resolve_project_path(config.tracker_config),
            "road_user_roi": (
                _resolve_project_path(config.road_user_roi) if config.road_user_roi else None
            ),
            "exit_roi": _resolve_project_path(config.exit_roi) if config.exit_roi else None,
            "cached_detections": (
                _resolve_project_path(config.cached_detections)
                if config.cached_detections
                else None
            ),
            "cached_rejected_detections": (
                _resolve_project_path(config.cached_rejected_detections)
                if config.cached_rejected_detections
                else None
            ),
        }
    )
    config.validate()
    import cv2
    from tqdm import tqdm

    config.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(config.input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {config.input_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video metadata contains invalid FPS or dimensions")
    maximum_frames = source_frames
    if config.max_seconds is not None:
        maximum_frames = min(source_frames, int(config.max_seconds * fps))

    detector_config = load_detector_config(config.detection_config)
    detector = (
        None if config.cached_detections else RoadUserDetector(detector_config)
    )
    cached_detections = (
        _load_cached_detections(config.cached_detections)
        if config.cached_detections
        else None
    )
    if config.cached_rejected_detections is not None:
        assert cached_detections is not None
        recovered = _load_cached_detections(
            config.cached_rejected_detections,
            allowed_reasons={"outside_road_user_roi"},
        )
        _extend_cached_detections(cached_detections, recovered)
    tracker = GroupedBoTSORT(
        config.tracker_config,
        device=detector_config.device,
        max_prediction_frames=config.max_prediction_frames,
    )
    road_roi = RoadUserROI.from_json(config.road_user_roi) if config.road_user_roi else None
    exit_roi = RoadUserROI.from_json(config.exit_roi) if config.exit_roi else None
    lifecycle = TrackLifecycleManager(
        LifecycleConfig(config.confirmation_observations),
        exit_roi=exit_roi,
        frame_width=width,
        frame_height=height,
    )

    detection_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    prior_removed_ids: set[int] = set()
    processed_frames = 0
    final_frame = -1

    progress = tqdm(total=maximum_frames, desc="V4 detect + track", unit="frame")
    try:
        for frame_number in range(maximum_frames):
            success, frame = capture.read()
            progress.update(1)
            if not success:
                break
            final_frame = frame_number
            raw_detections = (
                cached_detections.get(frame_number, [])
                if cached_detections is not None
                else detector.predict(frame)
            )
            thresholded_detections, threshold_rejections = _apply_confidence_thresholds(
                raw_detections, detector_config
            )
            detections, rejected = postprocess_detections(
                thresholded_detections,
                frame_width=width,
                frame_height=height,
                config=PostprocessConfig(),
                roi=road_roi,
            )
            observations = tracker.update(detections, frame, frame_number)
            removed_ids = tracker.removed_global_ids
            newly_removed = removed_ids - prior_removed_ids
            prior_removed_ids = removed_ids
            lifecycle.update(
                observations,
                frame_number=frame_number,
                removed_track_ids=newly_removed,
            )

            detection_rows.extend(_detection_row(item, frame_number, fps) for item in detections)
            rejected_rows.extend(
                _rejected_row(item, frame_number, fps)
                for item in threshold_rejections
            )
            rejected_rows.extend(_rejected_row(item, frame_number, fps) for item in rejected)
            rows = [_track_row(item, fps) for item in observations]
            track_rows.extend(rows)
            processed_frames += 1
    finally:
        progress.close()
        capture.release()

    lifecycle.finalize_video(max(final_frame, 0))
    native_summaries = {
        summary.track_id: summary for summary in lifecycle.summaries()
    }
    import pandas as pd

    native_frame = pd.DataFrame(track_rows, columns=TRACK_FIELDS)
    if native_frame.empty:
        stitched_frame_table = native_frame.copy()
        stitch_table = pd.DataFrame(columns=STITCH_FIELDS)
    elif config.enable_offline_stitching:
        stitched_frame_table, stitch_table = stitch_tracks(
            native_frame, StitchingConfig()
        )
    else:
        stitched_frame_table = native_frame.copy()
        stitch_table = pd.DataFrame(columns=STITCH_FIELDS)
    all_summary_rows = _final_summary_rows(stitched_frame_table, native_summaries)
    confirmed_final_ids = _confirmed_track_ids(
        all_summary_rows, config.confirmation_observations
    )
    final_frame_table = stitched_frame_table[
        stitched_frame_table["track_id"].isin(confirmed_final_ids)
    ].copy()
    final_track_rows = final_frame_table.to_dict("records")
    final_observed_rows = final_frame_table[
        final_frame_table["observed"].astype(bool)
    ].to_dict("records")
    final_summary_rows = [
        summary
        for summary in all_summary_rows
        if int(summary["track_id"]) in confirmed_final_ids
    ]
    source_to_final = {
        int(source): int(final_track)
        for source, final_track in final_frame_table[
            ["source_track_id", "track_id"]
        ].drop_duplicates().itertuples(index=False, name=None)
    }

    native_event_rows = [
        {**asdict(event), "state": event.state.value}
        for event in lifecycle.events
    ]
    final_event_rows = [
        {
            **row,
            "track_id": source_to_final[int(row["track_id"])],
            "source_track_id": int(row["track_id"]),
        }
        for row in native_event_rows
        if int(row["track_id"]) in source_to_final
    ]
    for stitch in stitch_table.to_dict("records"):
        new_track_id = int(stitch["new_track_id"])
        if new_track_id not in source_to_final:
            continue
        start_frame = int(
            final_frame_table.loc[
                final_frame_table["source_track_id"] == new_track_id, "frame"
            ].min()
        )
        final_event_rows.append(
            {
                "frame": start_frame,
                "track_id": int(stitch["canonical_track_id"]),
                "source_track_id": new_track_id,
                "event": "stitched",
                "state": "recovered",
                "reason": "offline_motion_consistent_tracklet_stitch",
            }
        )
    final_event_rows.sort(
        key=lambda row: (int(row["frame"]), int(row["track_id"]), str(row["event"]))
    )

    _write_rows(config.output_dir / "detections.csv", DETECTION_FIELDS, detection_rows)
    _write_rows(
        config.output_dir / "rejected_detections.csv",
        DETECTION_FIELDS + ["reason"],
        rejected_rows,
    )
    _write_rows(config.output_dir / "tracks_native.csv", TRACK_FIELDS, track_rows)
    _write_rows(config.output_dir / "tracks.csv", TRACK_FIELDS, final_track_rows)
    _write_rows(config.output_dir / "raw_tracks.csv", TRACK_FIELDS, final_observed_rows)
    _write_rows(
        config.output_dir / "rejected_track_predictions.csv",
        REJECTED_PREDICTION_FIELDS,
        [asdict(item) for item in tracker.rejected_predictions],
    )
    _write_rows(
        config.output_dir / "track_events_native.csv",
        ["frame", "track_id", "event", "state", "reason"],
        native_event_rows,
    )
    _write_rows(
        config.output_dir / "track_events.csv",
        ["frame", "track_id", "source_track_id", "event", "state", "reason"],
        final_event_rows,
    )
    _write_rows(
        config.output_dir / "track_summary_native.csv",
        [
            "track_id",
            "state",
            "first_frame",
            "last_frame",
            "last_observed_frame",
            "observation_count",
            "final_class_id",
            "final_class_name",
            "final_class_confidence",
        ],
        [
            {**asdict(summary), "state": summary.state.value}
            for summary in native_summaries.values()
        ],
    )
    _write_rows(
        config.output_dir / "track_summary.csv",
        [
            "track_id",
            "state",
            "first_frame",
            "last_frame",
            "last_observed_frame",
            "observation_count",
            "final_class_id",
            "final_class_name",
            "final_class_confidence",
        ],
        final_summary_rows,
    )
    _write_rows(
        config.output_dir / "track_stitches.csv",
        STITCH_FIELDS,
        stitch_table.to_dict("records"),
    )

    run_id = str(uuid.uuid4())
    input_hash = _sha256(config.input_video)
    trajectory_hash = _sha256(config.output_dir / "tracks.csv")
    quality_report = {
        "schema_version": 1,
        "run_id": run_id,
        "quality_status": "not_evaluated",
        "ground_truth_metrics": None,
        "blocking_reason": "COCO and MOT ground-truth evaluation has not been run",
        "counts": {
            "processed_frames": processed_frames,
            "merged_detections": len(detection_rows),
            "rejected_detections": len(rejected_rows),
            "observed_track_rows": len(final_observed_rows),
            "predicted_track_rows": len(final_track_rows) - len(final_observed_rows),
            "rejected_track_predictions": len(tracker.rejected_predictions),
            "native_unique_tracks": len(native_summaries),
            "stitched_tracklets": len(stitch_table),
            "unique_tracks": len(final_summary_rows),
            "discarded_tentative_tracks": len(all_summary_rows)
            - len(final_summary_rows),
            "internal_expirations": sum(
                summary["state"] == "expired" for summary in final_summary_rows
            ),
        },
    }
    (config.output_dir / "quality_report.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "quality_status": "not_evaluated",
        "input_video_sha256": input_hash,
        "cached_detections_sha256": (
            _sha256(config.cached_detections) if config.cached_detections else None
        ),
        "cached_rejected_detections_sha256": (
            _sha256(config.cached_rejected_detections)
            if config.cached_rejected_detections
            else None
        ),
        "tracks_sha256": trajectory_hash,
        "source": {
            "video": str(config.input_video),
            "fps": fps,
            "width": width,
            "height": height,
            "frames": source_frames,
            "processed_frames": processed_frames,
        },
        "config": {
            **asdict(config),
            "input_video": str(config.input_video),
            "output_dir": str(config.output_dir),
            "detection_config": str(config.detection_config),
            "tracker_config": str(config.tracker_config),
            "road_user_roi": str(config.road_user_roi) if config.road_user_roi else None,
            "exit_roi": str(config.exit_roi) if config.exit_roi else None,
            "cached_detections": (
                str(config.cached_detections) if config.cached_detections else None
            ),
            "cached_rejected_detections": (
                str(config.cached_rejected_detections)
                if config.cached_rejected_detections
                else None
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "ultralytics": _package_version("ultralytics"),
            "sahi": _package_version("sahi"),
            "opencv": _package_version("opencv-python"),
            "torch": _package_version("torch"),
            "lap": _package_version("lap"),
        },
        "outputs": {
            "detections": "detections.csv",
            "rejected_detections": "rejected_detections.csv",
            "tracks": "tracks.csv",
            "native_tracks": "tracks_native.csv",
            "observed_tracks_compatibility": "raw_tracks.csv",
            "rejected_track_predictions": "rejected_track_predictions.csv",
            "track_events": "track_events.csv",
            "native_track_events": "track_events_native.csv",
            "track_summary": "track_summary.csv",
            "native_track_summary": "track_summary_native.csv",
            "track_stitches": "track_stitches.csv",
            "quality_report": "quality_report.json",
        },
    }
    (config.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return {"manifest": manifest, "quality_report": quality_report}


