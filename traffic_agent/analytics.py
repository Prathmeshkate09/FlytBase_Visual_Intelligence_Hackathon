from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "frame",
    "timestamp_s",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "ground_x",
    "ground_y",
}


def _rolling_median(values: pd.Series, window: int) -> pd.Series:
    safe_window = max(1, min(window, len(values)))
    return values.rolling(safe_window, center=True, min_periods=1).median()


def _parse_observed(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    parsed = normalized.map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        invalid = sorted(normalized[parsed.isna()].unique())
        raise ValueError(f"Unsupported observed values: {invalid}")
    return parsed.astype(bool)


def clean_trajectories(
    raw: pd.DataFrame,
    fps: float,
    min_track_seconds: float = 0.5,
    max_gap_frames: int = 10,
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """Stabilize class labels, interpolate brief gaps, and smooth track points."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing trajectory columns: {sorted(missing)}")
    if raw.empty:
        return raw.copy()

    minimum_observations = max(2, int(round(min_track_seconds * fps)))
    cleaned_groups: list[pd.DataFrame] = []

    for track_id, group in raw.groupby("track_id", sort=False):
        group = group.sort_values("frame").drop_duplicates("frame", keep="last")
        if len(group) < minimum_observations:
            continue

        class_id = int(group["class_id"].mode().iloc[0])
        class_name = str(group.loc[group["class_id"] == class_id, "class_name"].mode().iloc[0])

        chunks: list[pd.DataFrame] = []
        chunk_start = 0
        frames = group["frame"].to_numpy(dtype=int)
        for index, gap in enumerate(np.diff(frames), start=1):
            if gap > max_gap_frames + 1:
                chunks.append(group.iloc[chunk_start:index])
                chunk_start = index
        chunks.append(group.iloc[chunk_start:])

        for chunk_index, chunk in enumerate(chunks):
            if len(chunk) < minimum_observations:
                continue
            frame_index = np.arange(int(chunk["frame"].min()), int(chunk["frame"].max()) + 1)
            source_observed = (
                pd.Series(
                    _parse_observed(chunk["observed"]).to_numpy(),
                    index=chunk["frame"].to_numpy(dtype=int),
                )
                if "observed" in chunk
                else None
            )
            expanded = chunk.set_index("frame").reindex(frame_index)
            expanded.index.name = "frame"
            expanded["observed"] = (
                source_observed.reindex(frame_index).fillna(False).astype(bool)
                if source_observed is not None
                else expanded["track_id"].notna()
            )
            expanded["track_id"] = int(track_id)
            expanded["segment_id"] = f"{int(track_id)}-{chunk_index}"
            expanded["class_id"] = class_id
            expanded["class_name"] = class_name
            expanded["timestamp_s"] = expanded.index.to_numpy(dtype=float) / fps

            numeric = [
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
            for column in numeric:
                if column in expanded:
                    expanded[column] = expanded[column].interpolate(limit=max_gap_frames, limit_direction="both")

            expanded["ground_x_smooth"] = _rolling_median(expanded["ground_x"], smoothing_window)
            expanded["ground_y_smooth"] = _rolling_median(expanded["ground_y"], smoothing_window)
            cleaned_groups.append(expanded.reset_index())

    if not cleaned_groups:
        return pd.DataFrame(columns=list(raw.columns) + ["observed", "segment_id", "ground_x_smooth", "ground_y_smooth"])
    return pd.concat(cleaned_groups, ignore_index=True).sort_values(["frame", "track_id"])


def calculate_object_metrics(
    trajectories: pd.DataFrame,
    fps: float,
    raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per track and frame-level motion values in pixel units."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    if trajectories.empty:
        return pd.DataFrame(), trajectories.copy()

    consistency_by_track: dict[int, float] = {}
    if raw is not None:
        required = {"track_id", "class_id"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Missing raw class-consistency columns: {sorted(missing)}")
        for raw_track_id, raw_group in raw.groupby("track_id", sort=False):
            modal_class = raw_group["class_id"].mode().iloc[0]
            consistency_by_track[int(raw_track_id)] = float((raw_group["class_id"] == modal_class).mean())

    motion_groups: list[pd.DataFrame] = []
    object_rows: list[dict[str, float | int | str]] = []
    dt = 1.0 / fps

    for (track_id, segment_id), group in trajectories.groupby(["track_id", "segment_id"], sort=False):
        group = group.sort_values("frame").copy()
        dx = group["ground_x_smooth"].diff()
        dy = group["ground_y_smooth"].diff()
        step_px = np.hypot(dx, dy).fillna(0.0)
        group["speed_px_s"] = step_px / dt
        group["acceleration_px_s2"] = group["speed_px_s"].diff().fillna(0.0) / dt
        motion_groups.append(group)

        displacement = float(
            np.hypot(
                group["ground_x_smooth"].iloc[-1] - group["ground_x_smooth"].iloc[0],
                group["ground_y_smooth"].iloc[-1] - group["ground_y_smooth"].iloc[0],
            )
        )
        path_length = float(step_px.sum())
        duration = float(group["timestamp_s"].iloc[-1] - group["timestamp_s"].iloc[0] + dt)
        observed_ratio = float(group["observed"].mean()) if "observed" in group else 1.0
        class_consistency = consistency_by_track.get(
            int(track_id),
            float(
                trajectories.loc[trajectories["track_id"] == track_id, "class_id"]
                .value_counts(normalize=True)
                .iloc[0]
            ),
        )
        object_rows.append(
            {
                "track_id": int(track_id),
                "segment_id": str(segment_id),
                "class_id": int(group["class_id"].iloc[0]),
                "class_name": str(group["class_name"].iloc[0]),
                "start_s": float(group["timestamp_s"].iloc[0]),
                "end_s": float(group["timestamp_s"].iloc[-1]),
                "duration_s": duration,
                "observed_ratio": observed_ratio,
                "class_consistency": class_consistency,
                "path_length_px": path_length,
                "displacement_px": displacement,
                "straightness": displacement / path_length if path_length > 0 else 0.0,
                "mean_speed_px_s": float(group["speed_px_s"].mean()),
                "p95_speed_px_s": float(group["speed_px_s"].quantile(0.95)),
                "max_speed_px_s": float(group["speed_px_s"].max()),
            }
        )

    return pd.DataFrame(object_rows), pd.concat(motion_groups, ignore_index=True)


def calculate_timeseries(trajectories: pd.DataFrame) -> pd.DataFrame:
    if trajectories.empty:
        return pd.DataFrame()
    observed_mask = (
        trajectories["observed"].astype(bool)
        if "observed" in trajectories.columns
        else pd.Series(True, index=trajectories.index)
    )
    observed = trajectories[observed_mask].copy()
    observed["second"] = np.floor(observed["timestamp_s"]).astype(int)
    counts = (
        observed.groupby(["second", "class_name"])["track_id"]
        .nunique()
        .rename("visible_objects")
        .reset_index()
    )
    totals = observed.groupby("second")["track_id"].nunique().rename("all_visible_objects").reset_index()
    return counts.merge(totals, on="second", how="left")


def tracking_quality(object_metrics: pd.DataFrame, raw: pd.DataFrame, fps: float) -> dict[str, float | int]:
    if object_metrics.empty:
        return {
            "unique_tracks": 0,
            "median_track_duration_s": 0.0,
            "long_track_ratio": 0.0,
            "median_observed_ratio": 0.0,
            "median_class_consistency": 0.0,
        }
    durations = object_metrics["duration_s"]
    return {
        "unique_tracks": int(object_metrics["track_id"].nunique()),
        "tracked_detections": int(len(raw)),
        "median_track_duration_s": round(float(durations.median()), 3),
        "long_track_ratio": round(float((durations >= 2.0).mean()), 4),
        "median_observed_ratio": round(float(object_metrics["observed_ratio"].median()), 4),
        "median_class_consistency": round(float(object_metrics["class_consistency"].median()), 4),
        "fps": round(float(fps), 3),
    }


def save_density_heatmap(trajectories: pd.DataFrame, frame_width: int, frame_height: int, output_path: Path) -> None:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    if trajectories.empty:
        plt.text(0.5, 0.5, "No trajectories", ha="center", va="center")
    else:
        heatmap, x_edges, y_edges = np.histogram2d(
            trajectories["ground_x_smooth"],
            trajectories["ground_y_smooth"],
            bins=(max(32, frame_width // 40), max(24, frame_height // 40)),
            range=((0, frame_width), (0, frame_height)),
        )
        heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (0, 0), sigmaX=2.0)
        plt.imshow(heatmap.T, origin="upper", cmap="turbo", extent=(0, frame_width, frame_height, 0), aspect="auto")
        plt.colorbar(label="Trajectory density")
    plt.title("Road-user trajectory density")
    plt.xlabel("Image x (pixels)")
    plt.ylabel("Image y (pixels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_analysis_outputs(
    raw_csv: Path,
    output_dir: Path,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> dict[str, object]:
    raw = pd.read_csv(raw_csv)
    trajectories = clean_trajectories(raw, fps=fps)
    object_metrics, motion = calculate_object_metrics(trajectories, fps=fps, raw=raw)
    timeseries = calculate_timeseries(trajectories)

    trajectories.to_csv(output_dir / "trajectories.csv", index=False)
    motion.to_csv(output_dir / "trajectory_motion.csv", index=False)
    object_metrics.to_csv(output_dir / "object_metrics.csv", index=False)
    timeseries.to_csv(output_dir / "timeseries_counts.csv", index=False)
    save_density_heatmap(trajectories, frame_width, frame_height, output_dir / "density_heatmap.png")

    quality = tracking_quality(object_metrics, raw, fps)
    class_counts = (
        object_metrics.groupby("class_name")["track_id"].nunique().sort_values(ascending=False).astype(int).to_dict()
        if not object_metrics.empty
        else {}
    )
    summary: dict[str, object] = {
        "tracking_quality_proxies": quality,
        "unique_tracks_by_class": class_counts,
        "coordinate_status": "pixel-space; add a ground-plane homography before claiming metres or km/h",
        "outputs": {
            "raw_tracks": "raw_tracks.csv",
            "cleaned_trajectories": "trajectories.csv",
            "object_metrics": "object_metrics.csv",
            "timeseries_counts": "timeseries_counts.csv",
            "density_heatmap": "density_heatmap.png",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
