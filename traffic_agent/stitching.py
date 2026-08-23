from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "frame",
    "track_id",
    "association_group",
    "x1",
    "y1",
    "x2",
    "y2",
    "observed",
}


@dataclass(frozen=True)
class StitchingConfig:
    """Conservative offline recovery for short tracker ID breaks."""

    max_gap_frames: int = 12
    velocity_window: int = 5
    minimum_observations: int = 3
    minimum_distance_px: float = 35.0
    distance_diagonal_scale: float = 1.75
    maximum_area_ratio: float = 2.5
    maximum_direction_change_degrees: float = 60.0
    minimum_motion_px_per_frame: float = 1.0
    ambiguity_margin: float = 0.12

    def validate(self) -> None:
        if self.max_gap_frames < 1:
            raise ValueError("max_gap_frames must be positive")
        if self.velocity_window < 2:
            raise ValueError("velocity_window must be at least two")
        if self.minimum_observations < 2:
            raise ValueError("minimum_observations must be at least two")
        if self.minimum_distance_px <= 0 or self.distance_diagonal_scale <= 0:
            raise ValueError("distance thresholds must be positive")
        if self.maximum_area_ratio <= 1:
            raise ValueError("maximum_area_ratio must be greater than one")
        if not 0 < self.maximum_direction_change_degrees <= 180:
            raise ValueError("maximum_direction_change_degrees must be in (0, 180]")
        if self.minimum_motion_px_per_frame < 0:
            raise ValueError("minimum_motion_px_per_frame cannot be negative")
        if self.ambiguity_margin < 0:
            raise ValueError("ambiguity_margin cannot be negative")


@dataclass(frozen=True)
class StitchDecision:
    old_track_id: int
    new_track_id: int
    canonical_track_id: int
    gap_frames: int
    endpoint_distance_px: float
    predicted_distance_px: float
    distance_limit_px: float
    area_ratio: float
    direction_change_degrees: float | None
    score: float


@dataclass(frozen=True)
class _Tracklet:
    track_id: int
    association_group: str
    first_frame: int
    last_frame: int
    first_point: np.ndarray
    last_point: np.ndarray
    start_velocity: np.ndarray
    end_velocity: np.ndarray
    median_area: float
    maximum_diagonal: float
    observations: int


@dataclass(frozen=True)
class _Candidate:
    old_track_id: int
    new_track_id: int
    gap_frames: int
    endpoint_distance_px: float
    predicted_distance_px: float
    distance_limit_px: float
    area_ratio: float
    direction_change_degrees: float | None
    score: float


def _validate_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(tracks.columns)
    if missing:
        raise ValueError(f"Missing stitching columns: {sorted(missing)}")
    if tracks.empty:
        raise ValueError("Cannot stitch an empty tracking table")

    work = tracks.copy()
    numeric_columns = ["frame", "track_id", "x1", "y1", "x2", "y2"]
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="raise")
        if not np.isfinite(work[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {column} contains non-finite values")
    if ((work["x2"] <= work["x1"]) | (work["y2"] <= work["y1"])).any():
        raise ValueError("Tracking table contains invalid bounding boxes")
    if work["association_group"].isna().any():
        raise ValueError("association_group contains null values")

    work["frame"] = work["frame"].astype(int)
    work["track_id"] = work["track_id"].astype(int)
    work["association_group"] = work["association_group"].astype(str)
    if work["observed"].dtype != bool:
        normalized = work["observed"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if normalized.isna().any():
            raise ValueError("observed contains values that are not boolean")
        work["observed"] = normalized.astype(bool)
    return work.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)


def _velocity(points: np.ndarray, frames: np.ndarray) -> np.ndarray:
    if len(points) < 2 or float(np.ptp(frames)) == 0.0:
        return np.zeros(2, dtype=float)
    centered = frames - frames.mean()
    denominator = float(np.dot(centered, centered))
    return np.asarray(
        [float(np.dot(centered, points[:, axis] - points[:, axis].mean()) / denominator) for axis in range(2)]
    )


def _tracklets(observed: pd.DataFrame, config: StitchingConfig) -> list[_Tracklet]:
    descriptors: list[_Tracklet] = []
    for track_id, group in observed.groupby("track_id", sort=False):
        ordered = group.sort_values("frame", kind="stable").drop_duplicates("frame", keep="last")
        if len(ordered) < config.minimum_observations:
            continue
        association_groups = ordered["association_group"].unique()
        if len(association_groups) != 1:
            raise ValueError(f"Track {int(track_id)} changes association_group")

        points = np.column_stack(
            (
                (ordered["x1"].to_numpy() + ordered["x2"].to_numpy()) / 2.0,
                ordered["y2"].to_numpy(),
            )
        )
        frames = ordered["frame"].to_numpy(dtype=float)
        window = min(config.velocity_window, len(ordered))
        start_velocity = _velocity(points[:window], frames[:window])
        end_velocity = _velocity(points[-window:], frames[-window:])
        widths = ordered["x2"].to_numpy() - ordered["x1"].to_numpy()
        heights = ordered["y2"].to_numpy() - ordered["y1"].to_numpy()
        areas = widths * heights
        diagonals = np.hypot(widths, heights)
        descriptors.append(
            _Tracklet(
                track_id=int(track_id),
                association_group=str(association_groups[0]),
                first_frame=int(frames[0]),
                last_frame=int(frames[-1]),
                first_point=points[0],
                last_point=points[-1],
                start_velocity=start_velocity,
                end_velocity=end_velocity,
                median_area=float(np.median(areas)),
                maximum_diagonal=float(np.max(diagonals)),
                observations=len(ordered),
            )
        )
    return descriptors


def _direction_change(first: np.ndarray, second: np.ndarray, minimum_motion: float) -> float | None:
    first_speed = float(np.linalg.norm(first))
    second_speed = float(np.linalg.norm(second))
    if first_speed < minimum_motion or second_speed < minimum_motion:
        return None
    cosine = float(np.dot(first, second) / (first_speed * second_speed))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _candidates(tracklets: list[_Tracklet], config: StitchingConfig) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for old in tracklets:
        for new in tracklets:
            if old.track_id == new.track_id or old.association_group != new.association_group:
                continue
            gap = new.first_frame - old.last_frame
            if gap < 1 or gap > config.max_gap_frames:
                continue

            area_ratio = max(old.median_area, new.median_area) / min(old.median_area, new.median_area)
            if area_ratio > config.maximum_area_ratio:
                continue
            direction_change = _direction_change(
                old.end_velocity, new.start_velocity, config.minimum_motion_px_per_frame
            )
            if (
                direction_change is not None
                and direction_change > config.maximum_direction_change_degrees
            ):
                continue

            endpoint_distance = float(np.linalg.norm(new.first_point - old.last_point))
            predicted = old.last_point + old.end_velocity * gap
            predicted_distance = float(np.linalg.norm(new.first_point - predicted))
            distance_limit = max(
                config.minimum_distance_px,
                config.distance_diagonal_scale * old.maximum_diagonal,
                config.distance_diagonal_scale * new.maximum_diagonal,
            )
            if predicted_distance > distance_limit:
                continue

            area_penalty = math.log(area_ratio) / math.log(config.maximum_area_ratio)
            score = (
                predicted_distance / distance_limit
                + 0.20 * area_penalty
                + 0.10 * gap / config.max_gap_frames
            )
            candidates.append(
                _Candidate(
                    old_track_id=old.track_id,
                    new_track_id=new.track_id,
                    gap_frames=gap,
                    endpoint_distance_px=endpoint_distance,
                    predicted_distance_px=predicted_distance,
                    distance_limit_px=distance_limit,
                    area_ratio=area_ratio,
                    direction_change_degrees=direction_change,
                    score=score,
                )
            )
    return candidates


def _unambiguous_candidates(
    candidates: list[_Candidate], config: StitchingConfig
) -> list[_Candidate]:
    by_old: dict[int, list[_Candidate]] = {}
    by_new: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        by_old.setdefault(candidate.old_track_id, []).append(candidate)
        by_new.setdefault(candidate.new_track_id, []).append(candidate)

    def is_clear(candidate: _Candidate, options: list[_Candidate]) -> bool:
        ordered = sorted(options, key=lambda item: item.score)
        if ordered[0] != candidate:
            return False
        return len(ordered) == 1 or ordered[1].score - candidate.score >= config.ambiguity_margin

    accepted = [
        candidate
        for candidate in candidates
        if is_clear(candidate, by_old[candidate.old_track_id])
        and is_clear(candidate, by_new[candidate.new_track_id])
    ]
    return sorted(accepted, key=lambda item: (item.gap_frames, item.score))


def stitch_tracks(
    tracks: pd.DataFrame, config: StitchingConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return final IDs plus an audit table of every accepted stitch.

    Only observed rows select tracklets. Kalman predictions are retained in the
    returned table, but when two source IDs map to the same frame the observed
    row wins. The original global tracker ID is always preserved as
    ``source_track_id``.
    """

    active = config or StitchingConfig()
    active.validate()
    work = _validate_tracks(tracks)
    observed = work[work["observed"]]
    tracklets = _tracklets(observed, active)
    accepted = _unambiguous_candidates(_candidates(tracklets, active), active)

    canonical: dict[int, int] = {int(value): int(value) for value in work["track_id"].unique()}

    def resolve(track_id: int) -> int:
        while canonical[track_id] != track_id:
            canonical[track_id] = canonical[canonical[track_id]]
            track_id = canonical[track_id]
        return track_id

    decisions: list[StitchDecision] = []
    used_incoming: set[int] = set()
    used_outgoing: set[int] = set()
    for candidate in accepted:
        if candidate.new_track_id in used_incoming or candidate.old_track_id in used_outgoing:
            continue
        old_canonical = resolve(candidate.old_track_id)
        new_canonical = resolve(candidate.new_track_id)
        if old_canonical == new_canonical:
            continue
        canonical[new_canonical] = old_canonical
        used_incoming.add(candidate.new_track_id)
        used_outgoing.add(candidate.old_track_id)
        decisions.append(
            StitchDecision(
                canonical_track_id=old_canonical,
                **asdict(candidate),
            )
        )

    stitched = work.copy()
    stitched["source_track_id"] = stitched["track_id"].astype(int)
    stitched["track_id"] = stitched["track_id"].map(lambda value: resolve(int(value)))
    stitched["_observed_rank"] = stitched["observed"].astype(int)
    if "confidence" in stitched.columns:
        stitched["_confidence_rank"] = pd.to_numeric(
            stitched["confidence"], errors="coerce"
        ).fillna(-1.0)
    else:
        stitched["_confidence_rank"] = -1.0
    stitched = (
        stitched.sort_values(
            ["frame", "track_id", "_observed_rank", "_confidence_rank"],
            ascending=[True, True, False, False],
            kind="stable",
        )
        .drop_duplicates(["frame", "track_id"], keep="first")
        .drop(columns=["_observed_rank", "_confidence_rank"])
        .sort_values(["frame", "track_id"], kind="stable")
        .reset_index(drop=True)
    )
    decision_rows = pd.DataFrame(
        [asdict(decision) for decision in decisions],
        columns=[field.name for field in StitchDecision.__dataclass_fields__.values()],
    )
    return stitched, decision_rows

