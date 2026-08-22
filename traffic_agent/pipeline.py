from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from .analytics import write_analysis_outputs


@dataclass(frozen=True)
class PipelineConfig:
    input_video: Path
    output_dir: Path
    model: str = "yolo26l.pt"
    tracker: str = "config/botsort_drone.yaml"
    imgsz: int = 1280
    confidence: float = 0.10
    iou: float = 0.50
    device: str = "0"
    classes: tuple[int, ...] = (0, 1, 2, 3, 5, 7)
    frame_stride: int = 1
    max_seconds: float | None = None
    save_video: bool = True
    trail_length: int = 45


CSV_FIELDS = [
    "frame",
    "timestamp_s",
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
    "ground_x",
    "ground_y",
]


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(value) for value in rng.integers(70, 255, size=3))


def _draw_trails(frame: np.ndarray, trails: dict[int, deque[tuple[int, int]]]) -> None:
    for track_id, points in trails.items():
        if len(points) < 2:
            continue
        polyline = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [polyline], False, _color(track_id), 2, cv2.LINE_AA)


def _rows_from_result(result, frame_number: int, fps: float) -> Iterable[dict[str, object]]:
    boxes = result.boxes
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    ids = boxes.id.detach().cpu().numpy().astype(int)
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    names = result.names

    rows: list[dict[str, object]] = []
    for bounds, track_id, class_id, confidence in zip(xyxy, ids, classes, confidences):
        x1, y1, x2, y2 = (float(value) for value in bounds)
        rows.append(
            {
                "frame": frame_number,
                "timestamp_s": round(frame_number / fps, 6),
                "track_id": int(track_id),
                "class_id": int(class_id),
                "class_name": str(names[int(class_id)]),
                "confidence": round(float(confidence), 6),
                "x1": round(x1, 3),
                "y1": round(y1, 3),
                "x2": round(x2, 3),
                "y2": round(y2, 3),
                "center_x": round((x1 + x2) / 2.0, 3),
                "center_y": round((y1 + y2) / 2.0, 3),
                "ground_x": round((x1 + x2) / 2.0, 3),
                "ground_y": round(y2, 3),
            }
        )
    return rows


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    if not config.input_video.exists():
        raise FileNotFoundError(f"Video not found: {config.input_video}")
    if config.frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(config.input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {config.input_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = total_frames
    if config.max_seconds is not None:
        max_frame = min(total_frames, int(config.max_seconds * fps))

    model = YOLO(config.model)
    tracker_path = Path(config.tracker)
    if not tracker_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root / tracker_path
        if candidate.exists():
            tracker_path = candidate

    video_writer = None
    annotated_path = config.output_dir / "annotated_tracking.mp4"
    if config.save_video:
        output_fps = max(1.0, fps / config.frame_stride)
        video_writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not video_writer.isOpened():
            raise RuntimeError("Could not create annotated output video")

    raw_csv = config.output_dir / "raw_tracks.csv"
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=config.trail_length))
    processed_frames = 0
    tracked_detections = 0

    try:
        with raw_csv.open("w", newline="", encoding="utf-8") as csv_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS)
            writer.writeheader()

            progress = tqdm(total=max_frame, desc="Detecting and tracking", unit="frame")
            frame_number = -1
            while frame_number + 1 < max_frame:
                success, frame = capture.read()
                frame_number += 1
                progress.update(1)
                if not success:
                    break
                if frame_number % config.frame_stride != 0:
                    continue

                result = model.track(
                    source=frame,
                    persist=True,
                    tracker=str(tracker_path),
                    imgsz=config.imgsz,
                    conf=config.confidence,
                    iou=config.iou,
                    classes=list(config.classes),
                    device=config.device,
                    half=config.device != "cpu",
                    verbose=False,
                )[0]
                rows = list(_rows_from_result(result, frame_number, fps))
                writer.writerows(rows)
                tracked_detections += len(rows)
                processed_frames += 1

                if video_writer is not None:
                    annotated = result.plot()
                    for row in rows:
                        track_id = int(row["track_id"])
                        trails[track_id].append((int(float(row["ground_x"])), int(float(row["ground_y"]))))
                    _draw_trails(annotated, trails)
                    video_writer.write(annotated)
            progress.close()
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()

    metadata = {
        "input_video": str(config.input_video),
        "source_fps": fps,
        "frame_width": width,
        "frame_height": height,
        "source_frames": total_frames,
        "processed_frames": processed_frames,
        "tracked_detections": tracked_detections,
        "config": {**asdict(config), "input_video": str(config.input_video), "output_dir": str(config.output_dir)},
    }
    (config.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if tracked_detections == 0:
        raise RuntimeError(
            "No tracked detections were produced. Try --conf 0.05, --imgsz 1600, "
            "or verify that the selected COCO classes are visible."
        )

    summary = write_analysis_outputs(
        raw_csv=raw_csv,
        output_dir=config.output_dir,
        fps=fps,
        frame_width=width,
        frame_height=height,
    )
    return {"metadata": metadata, "summary": summary}

