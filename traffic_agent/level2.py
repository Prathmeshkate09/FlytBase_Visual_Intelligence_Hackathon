from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .analytics import clean_trajectories
from .appearance import relative_size_classes, sample_track_colours
from .calibration import GroundCalibration, load_ground_calibration
from .telemetry import TelemetryGroundProjector, parse_dji_srt


LEVEL2_REQUIRED_COLUMNS = {
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
    "ground_x",
    "ground_y",
}


@dataclass(frozen=True)
class Level2Config:
    raw_tracks: Path
    video: Path
    output_dir: Path
    calibration: Path | None = None
    srt: Path | None = None
    srt_frame_offset: int = 0
    srt_segment_index: int | None = None
    srt_max_interpolation_gap_frames: int = 2
    min_track_seconds: float = 1.0
    max_gap_frames: int = 10
    trajectory_smoothing_window: int = 31
    max_color_samples: int = 7
    appearance_confidence: float = 0.20
    stop_speed_m_s: float = 0.50
    display_confidence: float = 0.25
    save_video: bool = True
    evidence_max_width: int = 1280
    evidence_trail_length: int = 20
    evidence_max_objects_per_frame: int = 12


def _video_properties(video_path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Video reports an invalid frame rate: {fps}")
    if width <= 0 or height <= 0 or frames <= 0:
        raise RuntimeError("Video reports invalid dimensions or frame count")
    return fps, width, height, frames


def _smooth(values: np.ndarray, requested_window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 5:
        return values.copy()
    window = min(max(5, requested_window), len(values))
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=2, mode="interp")


def _derivative(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing within a segment")
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, timestamps, edge_order=edge_order)


def _append_motion_columns(
    group: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    prefix: str,
    smoothing_window: int,
) -> pd.DataFrame:
    output = group.copy()
    timestamps = output["timestamp_s"].to_numpy(dtype=np.float64)
    x_smooth = _smooth(x, smoothing_window)
    y_smooth = _smooth(y, smoothing_window)
    vx = _derivative(x_smooth, timestamps)
    vy = _derivative(y_smooth, timestamps)
    speed = np.hypot(vx, vy)
    ax = _derivative(vx, timestamps)
    ay = _derivative(vy, timestamps)
    output[f"position_x_{prefix}"] = x_smooth
    output[f"position_y_{prefix}"] = y_smooth
    output[f"velocity_x_{prefix}_s"] = vx
    output[f"velocity_y_{prefix}_s"] = vy
    output[f"speed_{prefix}_s"] = speed
    output[f"acceleration_x_{prefix}_s2"] = ax
    output[f"acceleration_y_{prefix}_s2"] = ay
    output[f"acceleration_magnitude_{prefix}_s2"] = np.hypot(ax, ay)
    output[f"tangential_acceleration_{prefix}_s2"] = _derivative(speed, timestamps)
    output["heading_deg"] = (np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0
    return output


def calculate_kinematics(
    trajectories: pd.DataFrame,
    calibration: GroundCalibration | None = None,
    telemetry: TelemetryGroundProjector | None = None,
    smoothing_window: int = 15,
    stop_speed_m_s: float = 0.50,
) -> pd.DataFrame:
    """Calculate per-frame motion, emitting metric values only with a valid homography."""
    if trajectories.empty:
        return trajectories.copy()
    required = {"track_id", "segment_id", "timestamp_s", "ground_x_smooth", "ground_y_smooth"}
    missing = required - set(trajectories.columns)
    if missing:
        raise ValueError(f"Missing kinematics columns: {sorted(missing)}")
    if calibration is not None and telemetry is not None:
        raise ValueError("Use either a static calibration or SRT telemetry, not both")

    groups: list[pd.DataFrame] = []
    for _, group in trajectories.groupby(["track_id", "segment_id"], sort=False):
        group = group.sort_values("timestamp_s").drop_duplicates("timestamp_s", keep="last")
        points_px = group[["ground_x_smooth", "ground_y_smooth"]].to_numpy(dtype=np.float64)
        motion = _append_motion_columns(
            group,
            points_px[:, 0],
            points_px[:, 1],
            prefix="px",
            smoothing_window=smoothing_window,
        )
        if calibration is not None:
            points_m = calibration.transform(points_px)
            metric_reliable = calibration.inside_control_region(points_px)
            metric_quality = calibration.quality
        elif telemetry is not None:
            points_m, metric_reliable = telemetry.project(
                group["frame"].to_numpy(dtype=int),
                points_px,
            )
            metric_quality = "telemetry-derived"
        else:
            points_m = None

        if points_m is not None:
            if not np.isfinite(points_m).all():
                raise ValueError("Metric projection produced non-finite ground points")
            motion = _append_motion_columns(
                motion,
                points_m[:, 0],
                points_m[:, 1],
                prefix="m",
                smoothing_window=smoothing_window,
            )
            motion["speed_kmh"] = motion["speed_m_s"] * 3.6
            motion["inside_calibration_region"] = metric_reliable
            motion["metric_calibration_quality"] = metric_quality
            motion["stopped"] = motion["speed_m_s"] < stop_speed_m_s
        else:
            motion["inside_calibration_region"] = False
            motion["metric_calibration_quality"] = "unavailable"
            motion["stopped"] = False
        groups.append(motion)
    return pd.concat(groups, ignore_index=True).sort_values(["frame", "track_id"])


def _class_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for track_id, group in raw.groupby("track_id", sort=False):
        modal_id = int(group["class_id"].mode().iloc[0])
        modal_name = str(group.loc[group["class_id"] == modal_id, "class_name"].mode().iloc[0])
        consistency = float((group["class_id"] == modal_id).mean())
        box_width = (group["x2"] - group["x1"]).clip(lower=0)
        box_height = (group["y2"] - group["y1"]).clip(lower=0)
        estimated_plate_height = float(np.minimum(box_width, box_height).median() * 0.06)
        plate_status = (
            "insufficient_resolution_for_validated_ocr"
            if estimated_plate_height < 12.0
            else "specialized_plate_detector_and_ocr_not_run"
        )
        rows.append(
            {
                "track_id": int(track_id),
                "class_id": modal_id,
                "class_name": modal_name,
                "raw_class_consistency": consistency,
                "median_box_width_px": float(box_width.median()),
                "median_box_height_px": float(box_height.median()),
                "estimated_plate_height_upper_bound_px": estimated_plate_height,
                "plate_ocr_status": plate_status,
            }
        )
    return pd.DataFrame(rows)


def calculate_object_insights(
    raw: pd.DataFrame,
    kinematics: pd.DataFrame,
    appearance: pd.DataFrame,
    sizes: pd.DataFrame,
    fps: float,
    metric_available: bool,
) -> pd.DataFrame:
    class_stats = _class_statistics(raw)
    rows: list[dict[str, object]] = []
    frame_dt = 1.0 / fps
    for (track_id, segment_id), group in kinematics.groupby(["track_id", "segment_id"], sort=False):
        group = group.sort_values("timestamp_s")
        duration = float(group["timestamp_s"].iloc[-1] - group["timestamp_s"].iloc[0] + frame_dt)
        row: dict[str, object] = {
            "track_id": int(track_id),
            "segment_id": str(segment_id),
            "start_s": float(group["timestamp_s"].iloc[0]),
            "end_s": float(group["timestamp_s"].iloc[-1]),
            "duration_s": duration,
            "observed_ratio": float(group["observed"].mean()) if "observed" in group else 1.0,
            "mean_speed_px_s": float(group["speed_px_s"].mean()),
            "p95_speed_px_s": float(group["speed_px_s"].quantile(0.95)),
            "displacement_px": float(
                np.hypot(
                    group["position_x_px"].iloc[-1] - group["position_x_px"].iloc[0],
                    group["position_y_px"].iloc[-1] - group["position_y_px"].iloc[0],
                )
            ),
        }
        if metric_available:
            reliable_rows = group[group["inside_calibration_region"].astype(bool)]
            row["metric_coverage_ratio"] = float(len(reliable_rows) / len(group))
            row["metric_kinematics_reliable"] = bool(
                duration >= 2.0
                and row["observed_ratio"] >= 0.70
                and row["metric_coverage_ratio"] >= 0.80
            )
            if reliable_rows.empty:
                row.update(
                    {
                        "mean_speed_m_s": np.nan,
                        "p95_speed_m_s": np.nan,
                        "mean_speed_kmh": np.nan,
                        "p95_speed_kmh": np.nan,
                        "p95_acceleration_m_s2": np.nan,
                        "stopped_duration_s": 0.0,
                    }
                )
            else:
                row.update(
                    {
                        "mean_speed_m_s": float(reliable_rows["speed_m_s"].mean()),
                        "p95_speed_m_s": float(reliable_rows["speed_m_s"].quantile(0.95)),
                        "mean_speed_kmh": float(reliable_rows["speed_kmh"].mean()),
                        "p95_speed_kmh": float(reliable_rows["speed_kmh"].quantile(0.95)),
                        "p95_acceleration_m_s2": float(
                            reliable_rows["acceleration_magnitude_m_s2"].quantile(0.95)
                        ),
                        "stopped_duration_s": float(reliable_rows["stopped"].astype(bool).sum() * frame_dt),
                    }
                )
        else:
            row["metric_coverage_ratio"] = 0.0
            row["metric_kinematics_reliable"] = False
        rows.append(row)

    objects = pd.DataFrame(rows)
    objects = objects.merge(class_stats, on="track_id", how="left")
    objects = objects.merge(appearance, on="track_id", how="left")
    objects = objects.merge(sizes.drop(columns=["class_name"], errors="ignore"), on="track_id", how="left")
    return objects


def _box_colour(label: str) -> tuple[int, int, int]:
    return {
        "black": (80, 80, 80),
        "white": (245, 245, 245),
        "silver_gray": (180, 180, 180),
        "red": (40, 40, 230),
        "orange": (40, 150, 245),
        "yellow": (40, 225, 245),
        "green": (70, 200, 70),
        "blue": (230, 130, 40),
        "purple": (210, 80, 200),
    }.get(label, (255, 255, 255))


def _size_short(label: str) -> str:
    return {
        "small_passenger_vehicle": "S",
        "medium_passenger_vehicle": "M",
        "large_passenger_vehicle": "L",
        "two_wheeler": "2W",
        "heavy_vehicle": "HGV",
    }.get(label, label)


def _colour_short(label: str) -> str:
    return "gray" if label == "silver_gray" else label


def _overlaps(candidate: tuple[int, int, int, int], occupied: list[tuple[int, int, int, int]]) -> bool:
    x1, y1, x2, y2 = candidate
    return any(x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1 for ox1, oy1, ox2, oy2 in occupied)


def _draw_packed_label(
    frame: np.ndarray,
    text: str,
    bounds: tuple[int, int, int, int],
    colour: tuple[int, int, int],
    occupied: list[tuple[int, int, int, int]],
) -> None:
    x1, y1, x2, y2 = bounds
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    frame_height, frame_width = frame.shape[:2]
    candidates = [
        (x1, y1 - text_height - baseline - 7),
        (x1, y2 + text_height + baseline + 7),
        (x2 + 3, y1 + text_height + 3),
    ]
    for text_x, text_baseline in candidates:
        background = (
            max(0, text_x - 2),
            max(0, text_baseline - text_height - 3),
            min(frame_width - 1, text_x + text_width + 3),
            min(frame_height - 1, text_baseline + baseline + 2),
        )
        if background[2] <= background[0] or background[3] <= background[1] or _overlaps(background, occupied):
            continue
        cv2.rectangle(frame, (background[0], background[1]), (background[2], background[3]), (20, 20, 20), -1)
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
        return


def write_evidence_video(
    video_path: Path,
    output_path: Path,
    raw: pd.DataFrame,
    objects: pd.DataFrame,
    kinematics: pd.DataFrame,
    fps: float,
    display_confidence: float,
    max_width: int,
    trail_length: int,
    max_objects_per_frame: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = min(source_width, max_width)
    output_height = int(round(source_height * output_width / source_width))
    output_height -= output_height % 2
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create evidence video: {output_path}")

    per_track = objects.sort_values("duration_s", ascending=False).drop_duplicates("track_id").set_index("track_id")
    reliable_display = per_track[
        (per_track["duration_s"] >= 2.0)
        & (per_track["raw_class_consistency"] >= 0.80)
        & (per_track["color_confidence"] >= 0.55)
        & (per_track["displacement_px"] >= 12.0)
    ]
    stable_ids = set(reliable_display.index.astype(int))
    display = raw[(raw["track_id"].isin(stable_ids)) & (raw["confidence"] >= display_confidence)].copy()
    display["box_area_px2"] = (display["x2"] - display["x1"]).clip(lower=0) * (
        display["y2"] - display["y1"]
    ).clip(lower=0)
    display["display_score"] = display["confidence"] * np.sqrt(display["box_area_px2"].clip(lower=1))
    rows_by_frame = {
        int(frame): group.nlargest(max_objects_per_frame, "display_score")
        for frame, group in display.groupby("frame")
    }
    motion_columns = ["frame", "track_id", "speed_px_s"]
    if "speed_kmh" in kinematics:
        motion_columns += ["speed_kmh", "inside_calibration_region"]
    motion_lookup = kinematics[motion_columns].drop_duplicates(["frame", "track_id"]).set_index(["frame", "track_id"])
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_length))
    maximum_frame = int(raw["frame"].max())
    scale_x = output_width / source_width
    scale_y = output_height / source_height

    try:
        frame_number = 0
        while frame_number <= maximum_frame:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.resize(frame, (output_width, output_height))
            occupied_labels: list[tuple[int, int, int, int]] = []
            for _, detection in rows_by_frame.get(frame_number, pd.DataFrame()).iterrows():
                track_id = int(detection["track_id"])
                if track_id not in per_track.index:
                    continue
                insight = per_track.loc[track_id]
                colour_name = str(insight.get("dominant_color", "unknown"))
                box_colour = _box_colour(colour_name)
                x1 = int(float(detection["x1"]) * scale_x)
                y1 = int(float(detection["y1"]) * scale_y)
                x2 = int(float(detection["x2"]) * scale_x)
                y2 = int(float(detection["y2"]) * scale_y)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)
                trail_point = ((x1 + x2) // 2, y2)
                trails[track_id].append(trail_point)
                points = list(trails[track_id])
                for start, end in zip(points[:-1], points[1:]):
                    cv2.line(frame, start, end, box_colour, 2, cv2.LINE_AA)

                size_label = _size_short(str(insight.get("relative_size_class", "unknown")))
                label = f"#{track_id} {_colour_short(colour_name)} {size_label}"
                motion_key = (frame_number, track_id)
                if motion_key in motion_lookup.index:
                    motion = motion_lookup.loc[motion_key]
                    if "speed_kmh" in motion and bool(motion.get("inside_calibration_region", False)):
                        label += f" ~{float(motion['speed_kmh']):.0f}km/h"
                _draw_packed_label(frame, label, (x1, y1, x2, y2), box_colour, occupied_labels)
            cv2.rectangle(frame, (0, 0), (output_width, 32), (18, 18, 18), -1)
            cv2.putText(
                frame,
                "Level 2 evidence | moving, stable tracks | colour + relative size",
                (14, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_number += 1
    finally:
        capture.release()
        writer.release()


def save_speed_plot(kinematics: pd.DataFrame, objects: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    top_segments = objects.sort_values("duration_s", ascending=False).head(12)[["track_id", "segment_id"]]
    metric = "speed_kmh" in kinematics
    speed_column = "speed_kmh" if metric else "speed_px_s"
    plotted = 0
    for row in top_segments.itertuples(index=False):
        group = kinematics[
            (kinematics["track_id"] == row.track_id) & (kinematics["segment_id"] == row.segment_id)
        ].sort_values("timestamp_s")
        if metric:
            group = group[group["inside_calibration_region"].astype(bool)]
        if not group.empty:
            plt.plot(group["timestamp_s"], group[speed_column], linewidth=1.4, label=f"ID {row.track_id}")
            plotted += 1
    plt.xlabel("Time (s)")
    plt.ylabel("Estimated speed (km/h)" if metric else "Image-plane speed (px/s)")
    plt.title("Per-object speed profiles")
    if plotted:
        plt.legend(ncol=3, fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _build_summary(
    raw: pd.DataFrame,
    objects: pd.DataFrame,
    metric_report: dict[str, object] | None,
) -> dict[str, object]:
    unique_tracks = int(objects["track_id"].nunique()) if not objects.empty else 0
    color_counts = (
        objects.drop_duplicates("track_id")["dominant_color"].fillna("unknown").value_counts().to_dict()
        if not objects.empty
        else {}
    )
    size_counts = (
        objects.drop_duplicates("track_id")["relative_size_class"].fillna("unknown").value_counts().to_dict()
        if not objects.empty
        else {}
    )
    plate_counts = (
        objects.drop_duplicates("track_id")["plate_ocr_status"].value_counts().to_dict()
        if not objects.empty
        else {}
    )
    if metric_report is None:
        kinematics_summary: dict[str, object] = {
            "metric_status": "unavailable_without_ground_plane_calibration",
            "metric_reliable_objects": 0,
            "pixel_kinematics_available": True,
        }
    else:
        reliable = objects[objects["metric_kinematics_reliable"].astype(bool)]
        metric_quality = str(metric_report.get("quality", "unknown"))
        kinematics_summary = {
            "metric_status": (
                "measured" if metric_quality == "surveyed" else "estimated_from_camera_or_non_survey_calibration"
            ),
            "projection_quality": metric_quality,
            "metric_reliable_objects": int(reliable["track_id"].nunique()),
            "median_mean_speed_kmh": (
                round(float(reliable["mean_speed_kmh"].median()), 3) if not reliable.empty else None
            ),
            "median_p95_speed_kmh": (
                round(float(reliable["p95_speed_kmh"].median()), 3) if not reliable.empty else None
            ),
            "median_p95_acceleration_m_s2": (
                round(float(reliable["p95_acceleration_m_s2"].median()), 3) if not reliable.empty else None
            ),
            "plausibility_check": (
                "pass"
                if not reliable.empty
                and float(reliable["p95_speed_kmh"].quantile(0.99)) < 80.0
                and float(reliable["p95_acceleration_m_s2"].quantile(0.95)) < 8.0
                else "review_required"
            ),
        }
    metric_limitation = (
        "Metric kinematics are unavailable until a validated homography or frame-aligned SRT is supplied."
        if metric_report is None
        else "Metric kinematics are estimates subject to the stated camera model, telemetry, and ground-plane assumptions."
    )
    return {
        "level": "Level 2 - Object-Level Insight",
        "source_tracks": int(raw["track_id"].nunique()),
        "analysed_tracks": unique_tracks,
        "appearance": {
            "colour_method": "central vehicle-crop HSV family voting across sampled observations",
            "colour_distribution": color_counts,
            "size_method": "scene-relative median detector-box area; not make/model identification",
            "relative_size_distribution": size_counts,
        },
        "kinematics": kinematics_summary,
        "licence_plate_assessment": {
            "status_distribution": plate_counts,
            "validated_plate_reads": 0,
            "policy": "No plate text is emitted without a specialized detector, sufficient plate pixels, and OCR validation.",
        },
        "metric_projection": metric_report,
        "limitations": [
            "Coarse colour can be affected by shadows, glare, compression, and detector-box background.",
            "Vehicle size is relative within this scene; it is not a make/model or certified body-type prediction.",
            metric_limitation,
            "No licence-plate text is claimed from this top-down aerial sample.",
        ],
        "outputs": {
            "object_insights": "level2_object_insights.csv",
            "frame_kinematics": "level2_kinematics.csv",
            "appearance": "track_appearance.csv",
            "speed_plot": "level2_speed_profiles.png",
            "evidence_video": "level2_evidence.mp4",
            "metric_projection_report": "metric_projection_report.json" if metric_report is not None else None,
        },
    }


def run_level2(config: Level2Config) -> dict[str, object]:
    if not config.raw_tracks.exists():
        raise FileNotFoundError(f"Level 1 raw tracks not found: {config.raw_tracks}")
    if not config.video.exists():
        raise FileNotFoundError(f"Source video not found: {config.video}")
    if config.min_track_seconds <= 0:
        raise ValueError("min_track_seconds must be positive")
    if config.trajectory_smoothing_window < 1:
        raise ValueError("trajectory_smoothing_window must be positive")
    if config.calibration is not None and config.srt is not None:
        raise ValueError("Use either calibration JSON or SRT telemetry, not both")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(config.raw_tracks)
    missing = LEVEL2_REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing Level 1 columns: {sorted(missing)}")
    if raw.empty:
        raise RuntimeError("Level 1 raw tracks are empty")
    fps, width, height, source_frames = _video_properties(config.video)
    calibration = load_ground_calibration(config.calibration) if config.calibration is not None else None
    telemetry = (
        TelemetryGroundProjector(
            parse_dji_srt(config.srt, segment_index=config.srt_segment_index),
            frame_width=width,
            frame_height=height,
            frame_offset=config.srt_frame_offset,
            segment_index=config.srt_segment_index,
            max_interpolation_gap_frames=config.srt_max_interpolation_gap_frames,
        )
        if config.srt is not None
        else None
    )
    metric_report = (
        calibration.to_report()
        if calibration is not None
        else telemetry.to_report() if telemetry is not None else None
    )

    trajectories = clean_trajectories(
        raw,
        fps=fps,
        min_track_seconds=config.min_track_seconds,
        max_gap_frames=config.max_gap_frames,
        smoothing_window=5,
    )
    if trajectories.empty:
        raise RuntimeError("No tracks survived Level 2 stability filtering")
    appearance = sample_track_colours(
        raw,
        video_path=config.video,
        max_samples_per_track=config.max_color_samples,
        minimum_confidence=config.appearance_confidence,
    )
    sizes = relative_size_classes(raw)
    kinematics = calculate_kinematics(
        trajectories,
        calibration=calibration,
        telemetry=telemetry,
        smoothing_window=config.trajectory_smoothing_window,
        stop_speed_m_s=config.stop_speed_m_s,
    )
    if telemetry is not None:
        metric_report = telemetry.to_report()
    objects = calculate_object_insights(
        raw,
        kinematics=kinematics,
        appearance=appearance,
        sizes=sizes,
        fps=fps,
        metric_available=metric_report is not None,
    )

    appearance.to_csv(config.output_dir / "track_appearance.csv", index=False)
    kinematics.to_csv(config.output_dir / "level2_kinematics.csv", index=False)
    objects.to_csv(config.output_dir / "level2_object_insights.csv", index=False)
    save_speed_plot(kinematics, objects, config.output_dir / "level2_speed_profiles.png")
    if metric_report is not None:
        (config.output_dir / "metric_projection_report.json").write_text(
            json.dumps(metric_report, indent=2), encoding="utf-8"
        )
    if config.save_video:
        write_evidence_video(
            config.video,
            config.output_dir / "level2_evidence.mp4",
            raw=raw,
            objects=objects,
            kinematics=kinematics,
            fps=fps,
            display_confidence=config.display_confidence,
            max_width=config.evidence_max_width,
            trail_length=config.evidence_trail_length,
            max_objects_per_frame=config.evidence_max_objects_per_frame,
        )

    summary = _build_summary(raw, objects, metric_report)
    (config.output_dir / "level2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata = {
        "video": str(config.video),
        "raw_tracks": str(config.raw_tracks),
        "source_fps": fps,
        "frame_width": width,
        "frame_height": height,
        "source_frames": source_frames,
        "config": {
            **asdict(config),
            "raw_tracks": str(config.raw_tracks),
            "video": str(config.video),
            "output_dir": str(config.output_dir),
            "calibration": str(config.calibration) if config.calibration is not None else None,
            "srt": str(config.srt) if config.srt is not None else None,
        },
    }
    (config.output_dir / "level2_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return summary
