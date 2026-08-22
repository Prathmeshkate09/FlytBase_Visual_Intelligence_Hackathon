from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_QUALITIES = {"surveyed", "map-derived", "assumption-based"}


@dataclass(frozen=True)
class GroundCalibration:
    """Validated image-to-ground homography with auditable provenance."""

    matrix: np.ndarray
    image_points_px: np.ndarray
    world_points_m: np.ndarray
    reference: str
    quality: str
    rms_reprojection_error_m: float
    max_reprojection_error_m: float
    max_allowed_error_m: float

    def transform(self, points_px: np.ndarray) -> np.ndarray:
        points = np.asarray(points_px, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_px must have shape (N, 2)")
        transformed = cv2.perspectiveTransform(
            points.astype(np.float32).reshape(-1, 1, 2),
            self.matrix.astype(np.float64),
        )
        return transformed.reshape(-1, 2).astype(np.float64)

    def inside_control_region(self, points_px: np.ndarray) -> np.ndarray:
        """Flag points inside the convex hull of calibration control points."""
        points = np.asarray(points_px, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_px must have shape (N, 2)")
        hull = cv2.convexHull(self.image_points_px.astype(np.float32))
        return np.asarray(
            [cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in points],
            dtype=bool,
        )

    def to_report(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "quality": self.quality,
            "image_points_px": self.image_points_px.round(3).tolist(),
            "world_points_m": self.world_points_m.round(3).tolist(),
            "homography": self.matrix.tolist(),
            "rms_reprojection_error_m": round(self.rms_reprojection_error_m, 6),
            "max_reprojection_error_m": round(self.max_reprojection_error_m, 6),
            "max_allowed_error_m": self.max_allowed_error_m,
            "metric_claim": (
                "measured metric coordinates" if self.quality == "surveyed" else "estimated metric coordinates"
            ),
        }


def _validate_control_points(image_points: np.ndarray, world_points: np.ndarray) -> None:
    if image_points.shape != world_points.shape or image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("image_points_px and world_points_m must both have shape (N, 2)")
    if len(image_points) < 4:
        raise ValueError("At least four image/world control-point pairs are required")
    if not np.isfinite(image_points).all() or not np.isfinite(world_points).all():
        raise ValueError("Calibration control points must be finite")
    if len(np.unique(image_points, axis=0)) < 4 or len(np.unique(world_points, axis=0)) < 4:
        raise ValueError("Calibration requires at least four unique image and world points")
    image_area = float(cv2.contourArea(cv2.convexHull(image_points.astype(np.float32))))
    world_area = float(cv2.contourArea(cv2.convexHull(world_points.astype(np.float32))))
    if image_area < 100.0:
        raise ValueError("Image control points are degenerate or cover too little area")
    if world_area < 0.1:
        raise ValueError("World control points are degenerate or cover too little area")


def load_ground_calibration(path: Path) -> GroundCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_points = np.asarray(payload.get("image_points_px"), dtype=np.float64)
    world_points = np.asarray(payload.get("world_points_m"), dtype=np.float64)
    _validate_control_points(image_points, world_points)

    reference = str(payload.get("reference", "")).strip()
    if not reference:
        raise ValueError("Calibration must state its measurement source in 'reference'")
    quality = str(payload.get("quality", "")).strip().lower()
    if quality not in SUPPORTED_QUALITIES:
        raise ValueError(f"quality must be one of {sorted(SUPPORTED_QUALITIES)}")
    max_allowed_error = float(payload.get("max_reprojection_error_m", 0.75))
    if not np.isfinite(max_allowed_error) or max_allowed_error <= 0:
        raise ValueError("max_reprojection_error_m must be a positive finite number")

    matrix, _ = cv2.findHomography(image_points, world_points, method=0)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Could not solve a finite homography from the supplied control points")
    projected = cv2.perspectiveTransform(
        image_points.astype(np.float32).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - world_points, axis=1)
    rms_error = float(np.sqrt(np.mean(np.square(errors))))
    maximum_error = float(errors.max())
    if maximum_error > max_allowed_error:
        raise ValueError(
            f"Calibration maximum point error {maximum_error:.3f} m exceeds the allowed "
            f"{max_allowed_error:.3f} m"
        )

    return GroundCalibration(
        matrix=matrix,
        image_points_px=image_points,
        world_points_m=world_points,
        reference=reference,
        quality=quality,
        rms_reprojection_error_m=rms_error,
        max_reprojection_error_m=maximum_error,
        max_allowed_error_m=max_allowed_error,
    )
