from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class TelemetrySample:
    frame: int
    latitude: float
    longitude: float
    relative_altitude_m: float
    gimbal_yaw_deg: float
    gimbal_pitch_deg: float
    gimbal_roll_deg: float
    focal_length_35mm: float


def _required_float(block: str, pattern: str, field: str) -> float:
    match = re.search(pattern, block)
    if match is None:
        raise ValueError(f"SRT telemetry block is missing {field}")
    value = float(match.group(1))
    if not np.isfinite(value):
        raise ValueError(f"SRT telemetry field {field} is not finite")
    return value


def parse_dji_srt(path: Path, segment_index: int | None = None) -> list[TelemetrySample]:
    """Parse frame-aligned DJI-style SRT telemetry without silently dropping blocks."""
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    samples: list[TelemetrySample] = []
    for block in re.split(r"\r?\n\s*\r?\n", text):
        frame_match = re.search(r"FrameCnt:\s*(\d+)", block)
        if frame_match is None:
            continue
        frame = int(frame_match.group(1))
        samples.append(
            TelemetrySample(
                frame=frame,
                latitude=_required_float(block, r"\[latitude:\s*([-+\d.]+)\]", "latitude"),
                longitude=_required_float(block, r"\[longitude:\s*([-+\d.]+)\]", "longitude"),
                relative_altitude_m=_required_float(block, r"\[rel_alt:\s*([-+\d.]+)", "relative altitude"),
                gimbal_yaw_deg=_required_float(block, r"\[gb_yaw:\s*([-+\d.]+)", "gimbal yaw"),
                gimbal_pitch_deg=_required_float(block, r"gb_pitch:\s*([-+\d.]+)", "gimbal pitch"),
                gimbal_roll_deg=_required_float(block, r"gb_roll:\s*([-+\d.]+)", "gimbal roll"),
                focal_length_35mm=_required_float(block, r"\[focal_len:\s*([-+\d.]+)\]", "focal length"),
            )
        )
    if not samples:
        raise ValueError(f"No frame telemetry found in {path}")
    segments: list[list[TelemetrySample]] = [[]]
    for sample in samples:
        if segments[-1] and sample.frame <= segments[-1][-1].frame:
            segments.append([])
        segments[-1].append(sample)
    if len(segments) > 1 and segment_index is None:
        ranges = [(segment[0].frame, segment[-1].frame) for segment in segments]
        raise ValueError(
            f"SRT contains {len(segments)} frame-count segments {ranges}; "
            "select one explicitly with --srt-segment-index"
        )
    selected_index = 0 if segment_index is None else segment_index
    if selected_index < 0 or selected_index >= len(segments):
        raise ValueError(f"srt_segment_index must be between 0 and {len(segments) - 1}")
    selected = segments[selected_index]
    frames = [sample.frame for sample in selected]
    if len(frames) != len(set(frames)) or any(
        current <= previous for previous, current in zip(frames[:-1], frames[1:])
    ):
        raise ValueError("Selected SRT segment does not have strictly increasing unique FrameCnt values")
    return selected


class TelemetryGroundProjector:
    """Project image rays onto a locally flat ground plane using DJI telemetry."""

    def __init__(
        self,
        samples: list[TelemetrySample],
        frame_width: int,
        frame_height: int,
        frame_offset: int = 0,
        segment_index: int | None = None,
        max_interpolation_gap_frames: int = 2,
    ) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if frame_offset < 0:
            raise ValueError("frame_offset must be zero or greater")
        if max_interpolation_gap_frames < 0:
            raise ValueError("max_interpolation_gap_frames must be zero or greater")
        self.samples = {sample.frame: sample for sample in samples}
        self.sample_frames = np.asarray(sorted(self.samples), dtype=int)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_offset = frame_offset
        self.segment_index = segment_index
        self.max_interpolation_gap_frames = max_interpolation_gap_frames
        self.interpolated_frames: set[int] = set()
        first = samples[0]
        self.origin_latitude = first.latitude
        self.origin_longitude = first.longitude

    def _drone_offset_m(self, sample: TelemetrySample) -> tuple[float, float]:
        latitude_radians = np.radians(self.origin_latitude)
        east = (
            np.radians(sample.longitude - self.origin_longitude)
            * EARTH_RADIUS_M
            * np.cos(latitude_radians)
        )
        north = np.radians(sample.latitude - self.origin_latitude) * EARTH_RADIUS_M
        return float(east), float(north)

    @staticmethod
    def _interpolate_angle(start: float, end: float, fraction: float) -> float:
        delta = (end - start + 180.0) % 360.0 - 180.0
        return start + fraction * delta

    def _sample_for_frame(self, frame: int) -> TelemetrySample | None:
        exact = self.samples.get(frame)
        if exact is not None:
            return exact
        insertion = int(np.searchsorted(self.sample_frames, frame))
        if insertion == 0 or insertion == len(self.sample_frames):
            return None
        previous_frame = int(self.sample_frames[insertion - 1])
        next_frame = int(self.sample_frames[insertion])
        missing_count = next_frame - previous_frame - 1
        if missing_count > self.max_interpolation_gap_frames:
            return None
        previous = self.samples[previous_frame]
        following = self.samples[next_frame]
        fraction = (frame - previous_frame) / (next_frame - previous_frame)
        self.interpolated_frames.add(frame)
        return TelemetrySample(
            frame=frame,
            latitude=previous.latitude + fraction * (following.latitude - previous.latitude),
            longitude=previous.longitude + fraction * (following.longitude - previous.longitude),
            relative_altitude_m=previous.relative_altitude_m
            + fraction * (following.relative_altitude_m - previous.relative_altitude_m),
            gimbal_yaw_deg=self._interpolate_angle(
                previous.gimbal_yaw_deg, following.gimbal_yaw_deg, fraction
            ),
            gimbal_pitch_deg=previous.gimbal_pitch_deg
            + fraction * (following.gimbal_pitch_deg - previous.gimbal_pitch_deg),
            gimbal_roll_deg=self._interpolate_angle(
                previous.gimbal_roll_deg, following.gimbal_roll_deg, fraction
            ),
            focal_length_35mm=previous.focal_length_35mm
            + fraction * (following.focal_length_35mm - previous.focal_length_35mm),
        )

    def _project_one(self, sample: TelemetrySample, point_px: np.ndarray) -> tuple[np.ndarray, bool]:
        sensor_width_mm = 36.0
        sensor_height_mm = sensor_width_mm * self.frame_height / self.frame_width
        focal_x = self.frame_width * sample.focal_length_35mm / sensor_width_mm
        focal_y = self.frame_height * sample.focal_length_35mm / sensor_height_mm
        center_x = (self.frame_width - 1) / 2.0
        center_y = (self.frame_height - 1) / 2.0
        ray_x = (float(point_px[0]) - center_x) / focal_x
        ray_y = (float(point_px[1]) - center_y) / focal_y

        yaw = np.radians(sample.gimbal_yaw_deg)
        pitch = np.radians(sample.gimbal_pitch_deg)
        roll = np.radians(sample.gimbal_roll_deg)
        forward = np.asarray(
            [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)],
            dtype=np.float64,
        )
        right = np.asarray([np.cos(yaw), -np.sin(yaw), 0.0], dtype=np.float64)
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        rolled_right = right * np.cos(roll) + down * np.sin(roll)
        rolled_down = -right * np.sin(roll) + down * np.cos(roll)
        ray_world = forward + ray_x * rolled_right + ray_y * rolled_down

        altitude = sample.relative_altitude_m
        ray_z = float(ray_world[2])
        if altitude <= 0 or ray_z >= -0.05:
            return np.asarray([np.nan, np.nan]), False
        distance = -altitude / ray_z
        drone_east, drone_north = self._drone_offset_m(sample)
        ground = np.asarray(
            [drone_east + distance * ray_world[0], drone_north + distance * ray_world[1]],
            dtype=np.float64,
        )
        reliable = bool(
            sample.gimbal_pitch_deg <= -20.0
            and 5.0 <= altitude <= 500.0
            and 10.0 <= sample.focal_length_35mm <= 200.0
            and distance <= altitude * 10.0
            and np.isfinite(ground).all()
        )
        return ground, reliable

    def project(self, frame_numbers: np.ndarray, points_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frames = np.asarray(frame_numbers, dtype=int)
        points = np.asarray(points_px, dtype=np.float64)
        if frames.ndim != 1 or points.ndim != 2 or points.shape != (len(frames), 2):
            raise ValueError("frame_numbers must be (N,) and points_px must be (N, 2)")
        ground = np.full((len(frames), 2), np.nan, dtype=np.float64)
        reliable = np.zeros(len(frames), dtype=bool)
        missing_frames: list[int] = []
        for index, (frame, point) in enumerate(zip(frames, points)):
            telemetry_frame = int(frame) + self.frame_offset
            sample = self._sample_for_frame(telemetry_frame)
            if sample is None:
                missing_frames.append(telemetry_frame)
                continue
            ground[index], reliable[index] = self._project_one(sample, point)
        if missing_frames:
            preview = sorted(set(missing_frames))[:10]
            raise ValueError(f"SRT does not contain required telemetry frames: {preview}")
        return ground, reliable

    def to_report(self) -> dict[str, object]:
        ordered = sorted(self.samples.values(), key=lambda sample: sample.frame)
        altitudes = np.asarray([sample.relative_altitude_m for sample in ordered])
        pitches = np.asarray([sample.gimbal_pitch_deg for sample in ordered])
        yaws = np.asarray([sample.gimbal_yaw_deg for sample in ordered])
        focal_lengths = np.asarray([sample.focal_length_35mm for sample in ordered])
        return {
            "source": "DJI-style frame-aligned SRT telemetry",
            "quality": "telemetry-derived",
            "metric_claim": "estimated metric coordinates on a locally flat ground plane",
            "frame_offset": self.frame_offset,
            "segment_index": self.segment_index,
            "max_interpolation_gap_frames": self.max_interpolation_gap_frames,
            "interpolated_frame_count": len(self.interpolated_frames),
            "interpolated_frames": sorted(self.interpolated_frames),
            "telemetry_frame_range": [ordered[0].frame, ordered[-1].frame],
            "origin_latitude": self.origin_latitude,
            "origin_longitude": self.origin_longitude,
            "relative_altitude_m_range": [float(altitudes.min()), float(altitudes.max())],
            "gimbal_pitch_deg_range": [float(pitches.min()), float(pitches.max())],
            "gimbal_yaw_deg_range": [float(yaws.min()), float(yaws.max())],
            "focal_length_35mm_range": [float(focal_lengths.min()), float(focal_lengths.max())],
            "camera_model": (
                "Pinhole model using 35 mm-equivalent focal length, centred principal point, "
                "zero lens distortion, ENU local coordinates, and flat ground at takeoff elevation"
            ),
        }
