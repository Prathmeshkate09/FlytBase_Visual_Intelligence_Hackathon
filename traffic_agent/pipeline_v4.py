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
from .tracker import GroupedBoTSORT, TrackObservation


@dataclass(frozen=True)
class PipelineV4Config:
    input_video: Path
    output_dir: Path
    detection_config: Path = Path("config/detection_v4.json")
    tracker_config: Path = Path("config/botsort_drone_v4.yaml")
    road_user_roi: Path | None = None
    exit_roi: Path | None = None
    max_seconds: float | None = None
    confirmation_observations: int = 3

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
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
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


def _write_rows(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    detector = RoadUserDetector(detector_config)
    tracker = GroupedBoTSORT(
        config.tracker_config, device=detector_config.device
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
    observed_track_rows: list[dict[str, Any]] = []
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
            raw_detections = detector.predict(frame)
            detections, rejected = postprocess_detections(
                raw_detections,
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
            rejected_rows.extend(_rejected_row(item, frame_number, fps) for item in rejected)
            rows = [_track_row(item, fps) for item in observations]
            track_rows.extend(rows)
            observed_track_rows.extend(row for row in rows if row["observed"])
            processed_frames += 1
    finally:
        progress.close()
        capture.release()

    lifecycle.finalize_video(max(final_frame, 0))
    _write_rows(config.output_dir / "detections.csv", DETECTION_FIELDS, detection_rows)
    _write_rows(
        config.output_dir / "rejected_detections.csv",
        DETECTION_FIELDS + ["reason"],
        rejected_rows,
    )
    _write_rows(config.output_dir / "tracks.csv", TRACK_FIELDS, track_rows)
    _write_rows(config.output_dir / "raw_tracks.csv", TRACK_FIELDS, observed_track_rows)
    _write_rows(
        config.output_dir / "track_events.csv",
        ["frame", "track_id", "event", "state", "reason"],
        [
            {
                **asdict(event),
                "state": event.state.value,
            }
            for event in lifecycle.events
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
        [
            {**asdict(summary), "state": summary.state.value}
            for summary in lifecycle.summaries()
        ],
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
            "observed_track_rows": len(observed_track_rows),
            "predicted_track_rows": len(track_rows) - len(observed_track_rows),
            "unique_tracks": len(lifecycle.summaries()),
            "internal_expirations": sum(
                summary.state.value == "expired" for summary in lifecycle.summaries()
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
        },
        "versions": {
            "python": platform.python_version(),
            "ultralytics": _package_version("ultralytics"),
            "sahi": _package_version("sahi"),
            "opencv": _package_version("opencv-python-headless"),
        },
        "outputs": {
            "detections": "detections.csv",
            "rejected_detections": "rejected_detections.csv",
            "tracks": "tracks.csv",
            "observed_tracks_compatibility": "raw_tracks.csv",
            "track_events": "track_events.csv",
            "track_summary": "track_summary.csv",
            "quality_report": "quality_report.json",
        },
    }
    (config.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return {"manifest": manifest, "quality_report": quality_report}
