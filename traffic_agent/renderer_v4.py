from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


REQUIRED_TRACK_COLUMNS = {
    "frame",
    "track_id",
    "association_group",
    "x1",
    "y1",
    "x2",
    "y2",
    "observed",
}

CLASS_COLOURS = {
    "person": (255, 90, 220),
    "pedestrian": (255, 90, 220),
    "bicycle": (255, 220, 50),
    "cyclist": (255, 220, 50),
    "motorcycle": (40, 220, 255),
    "car": (255, 170, 50),
    "bus": (210, 80, 255),
    "truck": (70, 220, 90),
    "vehicle": (255, 170, 50),
}


@dataclass(frozen=True)
class RendererV4Config:
    max_width: int = 1920
    trail_length: int = 20
    box_thickness: int = 2
    label_scale: float = 0.42

    def validate(self) -> None:
        if self.max_width < 320:
            raise ValueError("max_width must be at least 320")
        if self.trail_length < 1:
            raise ValueError("trail_length must be positive")
        if self.box_thickness < 1:
            raise ValueError("box_thickness must be positive")
        if self.label_scale <= 0:
            raise ValueError("label_scale must be positive")


def _normalize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_TRACK_COLUMNS - set(tracks.columns)
    if missing:
        raise ValueError(f"Missing renderer columns: {sorted(missing)}")
    if tracks.empty:
        raise ValueError("Cannot render an empty tracking table")

    work = tracks.copy()
    for column in ["frame", "track_id", "x1", "y1", "x2", "y2"]:
        work[column] = pd.to_numeric(work[column], errors="raise")
        if not np.isfinite(work[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {column} contains non-finite values")
    if (work["frame"] < 0).any():
        raise ValueError("frame values cannot be negative")
    if ((work["x2"] <= work["x1"]) | (work["y2"] <= work["y1"])).any():
        raise ValueError("Renderer input contains invalid bounding boxes")
    work["frame"] = work["frame"].astype(int)
    work["track_id"] = work["track_id"].astype(int)
    if work.duplicated(["frame", "track_id"]).any():
        raise ValueError("Renderer input contains duplicate frame/track_id rows")
    if work["observed"].dtype != bool:
        normalized = work["observed"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if normalized.isna().any():
            raise ValueError("observed contains values that are not boolean")
        work["observed"] = normalized.astype(bool)
    return work.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def _class_lookup(summary: pd.DataFrame) -> dict[int, str]:
    required = {"track_id", "final_class_name"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing track-summary columns: {sorted(missing)}")
    if summary["track_id"].duplicated().any():
        raise ValueError("track_summary contains duplicate track IDs")
    return {
        int(row.track_id): str(row.final_class_name)
        for row in summary.itertuples(index=False)
    }


def _dashed_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    colour: tuple[int, int, int],
    thickness: int,
    dash: int = 8,
) -> None:
    vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(vector))
    if length == 0:
        return
    direction = vector / length
    for offset in range(0, int(length), dash * 2):
        segment_start = np.asarray(start, dtype=float) + direction * offset
        segment_end = np.asarray(start, dtype=float) + direction * min(offset + dash, length)
        cv2.line(
            frame,
            tuple(np.rint(segment_start).astype(int)),
            tuple(np.rint(segment_end).astype(int)),
            colour,
            thickness,
            cv2.LINE_AA,
        )


def _dashed_rectangle(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = bounds
    _dashed_line(frame, (x1, y1), (x2, y1), colour, thickness)
    _dashed_line(frame, (x2, y1), (x2, y2), colour, thickness)
    _dashed_line(frame, (x2, y2), (x1, y2), colour, thickness)
    _dashed_line(frame, (x1, y2), (x1, y1), colour, thickness)


def _overlaps(
    candidate: tuple[int, int, int, int],
    occupied: list[tuple[int, int, int, int]],
) -> bool:
    x1, y1, x2, y2 = candidate
    return any(
        x1 < other_x2 and x2 > other_x1 and y1 < other_y2 and y2 > other_y1
        for other_x1, other_y1, other_x2, other_y2 in occupied
    )


def _draw_label(
    frame: np.ndarray,
    text: str,
    bounds: tuple[int, int, int, int],
    colour: tuple[int, int, int],
    occupied: list[tuple[int, int, int, int]],
    scale: float,
) -> bool:
    x1, y1, x2, y2 = bounds
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    frame_height, frame_width = frame.shape[:2]
    candidates = [
        (x1, y1 - 5),
        (x1, y2 + text_height + baseline + 5),
        (x2 + 4, y1 + text_height + 2),
        (x1 - text_width - 4, y1 + text_height + 2),
    ]
    for text_x, text_baseline in candidates:
        background = (
            max(0, text_x - 2),
            max(34, text_baseline - text_height - 3),
            min(frame_width - 1, text_x + text_width + 3),
            min(frame_height - 1, text_baseline + baseline + 2),
        )
        if (
            background[2] <= background[0]
            or background[3] <= background[1]
            or _overlaps(background, occupied)
        ):
            continue
        cv2.rectangle(
            frame,
            (background[0], background[1]),
            (background[2], background[3]),
            (18, 18, 18),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (background[0] + 2, background[3] - baseline - 1),
            font,
            scale,
            colour,
            thickness,
            cv2.LINE_AA,
        )
        occupied.append(background)
        return True
    return False


def render_tracking_video(
    *,
    video_path: Path,
    tracks_path: Path,
    summary_path: Path,
    output_path: Path,
    report_path: Path | None = None,
    config: RendererV4Config | None = None,
) -> dict[str, Any]:
    active = config or RendererV4Config()
    active.validate()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks not found: {tracks_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Track summary not found: {summary_path}")

    tracks = _normalize_tracks(pd.read_csv(tracks_path))
    classes = _class_lookup(pd.read_csv(summary_path))
    unknown_ids = set(tracks["track_id"].unique()) - set(classes)
    if unknown_ids:
        raise ValueError(f"Track summary is missing IDs: {sorted(unknown_ids)[:10]}")
    rows_by_frame = {
        int(frame): group for frame, group in tracks.groupby("frame", sort=False)
    }

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or source_width <= 0 or source_height <= 0:
        capture.release()
        raise RuntimeError("Source video metadata is invalid")
    output_width = min(source_width, active.max_width)
    output_height = int(round(source_height * output_width / source_width))
    output_height -= output_height % 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create evidence video: {output_path}")

    maximum_frame = int(tracks["frame"].max())
    scale_x = output_width / source_width
    scale_y = output_height / source_height
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=active.trail_length)
    )
    rendered_rows = 0
    labels_drawn = 0
    labels_omitted = 0
    rendered_frames = 0

    try:
        for frame_number in range(maximum_frame + 1):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(
                    f"Source video ended before tracked frame {frame_number}"
                )
            frame = cv2.resize(frame, (output_width, output_height))
            occupied: list[tuple[int, int, int, int]] = []
            frame_rows = rows_by_frame.get(frame_number)
            observed_count = 0
            occluded_count = 0
            if frame_rows is not None:
                for row in frame_rows.itertuples(index=False):
                    track_id = int(row.track_id)
                    class_name = classes[track_id]
                    colour = CLASS_COLOURS.get(class_name.lower(), (255, 255, 255))
                    bounds = (
                        int(round(float(row.x1) * scale_x)),
                        int(round(float(row.y1) * scale_y)),
                        int(round(float(row.x2) * scale_x)),
                        int(round(float(row.y2) * scale_y)),
                    )
                    observed = bool(row.observed)
                    if observed:
                        cv2.rectangle(
                            frame,
                            (bounds[0], bounds[1]),
                            (bounds[2], bounds[3]),
                            colour,
                            active.box_thickness,
                            cv2.LINE_AA,
                        )
                        observed_count += 1
                        trails[track_id].append(
                            ((bounds[0] + bounds[2]) // 2, bounds[3])
                        )
                    else:
                        _dashed_rectangle(
                            frame, bounds, colour, active.box_thickness
                        )
                        occluded_count += 1
                    points = list(trails[track_id])
                    for start, end in zip(points[:-1], points[1:]):
                        cv2.line(frame, start, end, colour, 1, cv2.LINE_AA)
                    state = "obs" if observed else "occ"
                    if _draw_label(
                        frame,
                        f"#{track_id} {class_name} {state}",
                        bounds,
                        colour,
                        occupied,
                        active.label_scale,
                    ):
                        labels_drawn += 1
                    else:
                        labels_omitted += 1
                    rendered_rows += 1

            active_count = observed_count + occluded_count
            cv2.rectangle(frame, (0, 0), (output_width, 34), (18, 18, 18), -1)
            cv2.putText(
                frame,
                (
                    f"V4 tracking verification | frame {frame_number} | "
                    f"active {active_count} | observed {observed_count} | occluded {occluded_count}"
                ),
                (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            rendered_frames += 1
    finally:
        capture.release()
        writer.release()

    if rendered_rows != len(tracks):
        raise RuntimeError(
            f"Renderer completeness failure: wrote {rendered_rows} boxes for {len(tracks)} rows"
        )
    report = {
        "schema_version": 1,
        "video": str(video_path),
        "tracks": str(tracks_path),
        "output": str(output_path),
        "config": asdict(active),
        "rendered_frames": rendered_frames,
        "track_rows": len(tracks),
        "rendered_box_rows": rendered_rows,
        "labels_drawn": labels_drawn,
        "labels_omitted_for_collision": labels_omitted,
        "completeness_passed": rendered_rows == len(tracks),
    }
    destination = report_path or output_path.with_suffix(".json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

