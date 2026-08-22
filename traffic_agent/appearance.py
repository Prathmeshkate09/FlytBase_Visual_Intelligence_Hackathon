from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


COLOR_ORDER = (
    "black",
    "white",
    "silver_gray",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
)


def classify_bgr_pixels(pixels_bgr: np.ndarray) -> tuple[str, float]:
    """Classify a vehicle crop into a coarse, auditable colour family."""
    pixels = np.asarray(pixels_bgr, dtype=np.uint8).reshape(-1, 1, 3)
    if len(pixels) == 0:
        return "unknown", 0.0
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]

    labels = np.full(len(hsv), "silver_gray", dtype=object)
    labels[value < 48] = "black"
    neutral = saturation < 45
    labels[neutral & (value >= 205)] = "white"
    labels[neutral & (value >= 48) & (value < 205)] = "silver_gray"

    chromatic = ~neutral & (value >= 48)
    labels[chromatic & ((hue < 10) | (hue >= 170))] = "red"
    labels[chromatic & (hue >= 10) & (hue < 22)] = "orange"
    labels[chromatic & (hue >= 22) & (hue < 35)] = "yellow"
    labels[chromatic & (hue >= 35) & (hue < 85)] = "green"
    labels[chromatic & (hue >= 85) & (hue < 130)] = "blue"
    labels[chromatic & (hue >= 130) & (hue < 170)] = "purple"

    counts = {label: int(np.count_nonzero(labels == label)) for label in COLOR_ORDER}
    winner = max(COLOR_ORDER, key=lambda label: counts[label])
    return winner, counts[winner] / len(labels)


def classify_vehicle_crop(frame: np.ndarray, bounds: tuple[float, float, float, float]) -> tuple[str, float]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bounds
    x1 = int(np.clip(round(x1), 0, width - 1))
    x2 = int(np.clip(round(x2), x1 + 1, width))
    y1 = int(np.clip(round(y1), 0, height - 1))
    y2 = int(np.clip(round(y2), y1 + 1, height))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 4 or box_height < 4:
        return "unknown", 0.0

    inset_x = max(1, int(round(box_width * 0.22)))
    inset_y = max(1, int(round(box_height * 0.22)))
    crop = frame[y1 + inset_y : y2 - inset_y, x1 + inset_x : x2 - inset_x]
    if crop.size == 0:
        return "unknown", 0.0

    crop_h, crop_w = crop.shape[:2]
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (crop_w // 2, crop_h // 2),
        (max(1, crop_w // 2 - 1), max(1, crop_h // 2 - 1)),
        0,
        0,
        360,
        255,
        -1,
    )
    return classify_bgr_pixels(crop[mask.astype(bool)])


def relative_size_classes(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"track_id", "class_name", "x1", "y1", "x2", "y2"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing appearance columns: {sorted(missing)}")
    work = raw.copy()
    work["box_area_px2"] = (work["x2"] - work["x1"]).clip(lower=0) * (
        work["y2"] - work["y1"]
    ).clip(lower=0)
    rows = (
        work.groupby("track_id", sort=False)
        .agg(class_name=("class_name", lambda values: str(values.mode().iloc[0])), median_box_area_px2=("box_area_px2", "median"))
        .reset_index()
    )
    car_mask = rows["class_name"].eq("car")
    car_areas = rows.loc[car_mask, "median_box_area_px2"]
    if len(car_areas) >= 3:
        lower, upper = car_areas.quantile([0.33, 0.67]).tolist()
    elif len(car_areas):
        lower = upper = float(car_areas.median())
    else:
        lower = upper = 0.0

    def classify(row: pd.Series) -> str:
        class_name = str(row["class_name"])
        if class_name == "motorcycle":
            return "two_wheeler"
        if class_name == "bus":
            return "bus"
        if class_name == "truck":
            return "heavy_vehicle"
        if class_name != "car":
            return class_name
        area = float(row["median_box_area_px2"])
        if area < lower:
            return "small_passenger_vehicle"
        if area > upper:
            return "large_passenger_vehicle"
        return "medium_passenger_vehicle"

    rows["relative_size_class"] = rows.apply(classify, axis=1)
    rows["size_method"] = "scene-relative median detector-box area"
    return rows


def sample_track_colours(
    raw: pd.DataFrame,
    video_path: Path,
    max_samples_per_track: int = 7,
    minimum_confidence: float = 0.20,
) -> pd.DataFrame:
    required = {"frame", "track_id", "confidence", "x1", "y1", "x2", "y2"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing colour-sampling columns: {sorted(missing)}")
    if max_samples_per_track < 1:
        raise ValueError("max_samples_per_track must be at least 1")

    eligible = raw[raw["confidence"] >= minimum_confidence].copy()
    selected: list[pd.DataFrame] = []
    for _, group in eligible.groupby("track_id", sort=False):
        group = group.sort_values("frame")
        indices = np.unique(np.linspace(0, len(group) - 1, min(max_samples_per_track, len(group))).round().astype(int))
        selected.append(group.iloc[indices])
    if not selected:
        return pd.DataFrame(columns=["track_id", "dominant_color", "color_confidence", "color_samples"])
    samples = pd.concat(selected, ignore_index=True)
    samples_by_frame = {int(frame): group for frame, group in samples.groupby("frame")}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for appearance sampling: {video_path}")
    votes: dict[int, list[tuple[str, float]]] = defaultdict(list)
    try:
        last_frame = int(samples["frame"].max())
        frame_number = 0
        while frame_number <= last_frame:
            success, frame = capture.read()
            if not success:
                break
            for _, row in samples_by_frame.get(frame_number, pd.DataFrame()).iterrows():
                label, confidence = classify_vehicle_crop(
                    frame,
                    (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                )
                if label != "unknown":
                    votes[int(row["track_id"])].append((label, confidence))
            frame_number += 1
    finally:
        capture.release()

    rows: list[dict[str, object]] = []
    for track_id in sorted(raw["track_id"].astype(int).unique()):
        track_votes = votes.get(track_id, [])
        if not track_votes:
            rows.append(
                {"track_id": track_id, "dominant_color": "unknown", "color_confidence": 0.0, "color_samples": 0}
            )
            continue
        totals: dict[str, float] = defaultdict(float)
        for label, confidence in track_votes:
            totals[label] += max(0.05, float(confidence))
        winner = max(totals, key=totals.get)
        total_weight = sum(totals.values())
        rows.append(
            {
                "track_id": track_id,
                "dominant_color": winner,
                "color_confidence": round(totals[winner] / total_weight, 4),
                "color_samples": len(track_votes),
            }
        )
    return pd.DataFrame(rows)
