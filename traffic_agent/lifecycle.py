from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

from .postprocess import RoadUserROI
from .tracker import TrackObservation


class LifecycleState(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    OCCLUDED = "occluded"
    RECOVERED = "recovered"
    EXITED = "exited"
    EXPIRED = "expired"
    VIDEO_END = "video_end"


@dataclass(frozen=True)
class LifecycleConfig:
    confirmation_observations: int = 3

    def validate(self) -> None:
        if self.confirmation_observations < 1:
            raise ValueError("confirmation_observations must be positive")


@dataclass(frozen=True)
class TrackEvent:
    frame: int
    track_id: int
    event: str
    state: LifecycleState
    reason: str


@dataclass
class _TrackRecord:
    state: LifecycleState = LifecycleState.TENTATIVE
    first_frame: int = -1
    last_frame: int = -1
    last_observed_frame: int = -1
    observation_count: int = 0
    last_observation: TrackObservation | None = None
    class_scores: dict[tuple[int, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    finalized: bool = False


@dataclass(frozen=True)
class TrackSummary:
    track_id: int
    state: LifecycleState
    first_frame: int
    last_frame: int
    last_observed_frame: int
    observation_count: int
    final_class_id: int
    final_class_name: str
    final_class_confidence: float


class TrackLifecycleManager:
    """Audit every track transition without treating predictions as detections."""

    def __init__(
        self,
        config: LifecycleConfig | None = None,
        *,
        exit_roi: RoadUserROI | None = None,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self.config = config or LifecycleConfig()
        self.config.validate()
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        self.exit_roi = exit_roi
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._records: dict[int, _TrackRecord] = {}
        self._events: list[TrackEvent] = []

    def _emit(
        self, frame: int, track_id: int, event: str, state: LifecycleState, reason: str
    ) -> None:
        self._events.append(TrackEvent(frame, track_id, event, state, reason))

    def update(
        self,
        observations: Iterable[TrackObservation],
        *,
        frame_number: int,
        removed_track_ids: Iterable[int] = (),
    ) -> list[TrackObservation]:
        if frame_number < 0:
            raise ValueError("frame_number cannot be negative")
        rows = list(observations)
        seen_ids: set[int] = set()

        for observation in rows:
            if observation.frame != frame_number:
                raise ValueError("Observation frame does not match lifecycle frame")
            if observation.track_id in seen_ids:
                raise ValueError(f"Duplicate lifecycle observation for track {observation.track_id}")
            seen_ids.add(observation.track_id)
            record = self._records.get(observation.track_id)
            if record is None:
                record = _TrackRecord(first_frame=frame_number)
                self._records[observation.track_id] = record
                self._emit(
                    frame_number,
                    observation.track_id,
                    "created",
                    LifecycleState.TENTATIVE,
                    "first_tracker_output",
                )
            if record.finalized:
                raise RuntimeError(f"Finalized track {observation.track_id} reappeared")

            prior_state = record.state
            record.last_frame = frame_number
            record.last_observation = observation
            if observation.observed:
                record.last_observed_frame = frame_number
                record.observation_count += 1
                record.class_scores[
                    (observation.detected_class_id, observation.detected_class_name)
                ] += float(observation.confidence or 0.0)
                if prior_state == LifecycleState.OCCLUDED:
                    record.state = LifecycleState.RECOVERED
                    self._emit(
                        frame_number,
                        observation.track_id,
                        "recovered",
                        record.state,
                        "matched_after_occlusion",
                    )
                elif (
                    prior_state == LifecycleState.TENTATIVE
                    and record.observation_count >= self.config.confirmation_observations
                ):
                    record.state = LifecycleState.CONFIRMED
                    self._emit(
                        frame_number,
                        observation.track_id,
                        "confirmed",
                        record.state,
                        "minimum_observations_reached",
                    )
                elif prior_state == LifecycleState.RECOVERED:
                    record.state = LifecycleState.CONFIRMED
            elif prior_state not in {LifecycleState.TENTATIVE, LifecycleState.OCCLUDED}:
                record.state = LifecycleState.OCCLUDED
                self._emit(
                    frame_number,
                    observation.track_id,
                    "lost",
                    record.state,
                    "kalman_prediction_without_detection",
                )

        for track_id in set(int(value) for value in removed_track_ids):
            record = self._records.get(track_id)
            if record is None or record.finalized:
                continue
            last = record.last_observation
            is_exit = bool(
                last is not None
                and self.exit_roi is not None
                and self.exit_roi.contains(
                    last.ground_point, self.frame_width, self.frame_height
                )
            )
            record.state = LifecycleState.EXITED if is_exit else LifecycleState.EXPIRED
            record.finalized = True
            self._emit(
                frame_number,
                track_id,
                "exited" if is_exit else "expired",
                record.state,
                "inside_configured_exit" if is_exit else "removed_away_from_exit",
            )
        return rows

    def finalize_video(self, final_frame: int) -> None:
        for track_id, record in self._records.items():
            if record.finalized:
                continue
            record.state = LifecycleState.VIDEO_END
            record.finalized = True
            self._emit(
                final_frame,
                track_id,
                "video_end",
                record.state,
                "source_clip_ended",
            )

    @property
    def events(self) -> tuple[TrackEvent, ...]:
        return tuple(self._events)

    def summaries(self) -> list[TrackSummary]:
        summaries: list[TrackSummary] = []
        for track_id, record in sorted(self._records.items()):
            if record.class_scores:
                final_key, winning_score = max(
                    record.class_scores.items(), key=lambda item: (item[1], item[0][0])
                )
                total_score = sum(record.class_scores.values())
                class_confidence = winning_score / total_score if total_score > 0 else 0.0
                final_class_id, final_class_name = final_key
            else:
                final_class_id, final_class_name, class_confidence = -1, "unknown", 0.0
            summaries.append(
                TrackSummary(
                    track_id=track_id,
                    state=record.state,
                    first_frame=record.first_frame,
                    last_frame=record.last_frame,
                    last_observed_frame=record.last_observed_frame,
                    observation_count=record.observation_count,
                    final_class_id=final_class_id,
                    final_class_name=final_class_name,
                    final_class_confidence=class_confidence,
                )
            )
        return summaries
