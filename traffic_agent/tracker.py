from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .postprocess import ASSOCIATION_GROUPS, Detection


TRACKING_GROUPS = tuple(sorted(set(ASSOCIATION_GROUPS.values())))


@dataclass(frozen=True)
class TrackObservation:
    """One observed or Kalman-predicted state from the tracker."""

    frame: int
    track_id: int
    native_track_id: int
    association_group: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None
    detected_class_id: int
    detected_class_name: str
    observed: bool
    state: str
    detection_source: str

    @property
    def ground_point(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(frozen=True)
class RejectedTrackPrediction:
    """A Kalman prediction that became invalid after frame clipping."""

    frame: int
    track_id: int
    native_track_id: int
    association_group: str
    raw_x1: float
    raw_y1: float
    raw_x2: float
    raw_y2: float
    clipped_x1: float
    clipped_y1: float
    clipped_x2: float
    clipped_y2: float
    reason: str


TrackerFactory = Callable[[str], Any]
BoxesFactory = Callable[[list[Detection], tuple[int, int]], Any]


class GroupedBoTSORT:
    """Run independent BoT-SORT association pools for incompatible road users.

    Ultralytics BoT-SORT associates detections spatially and does not guarantee
    that a pedestrian and road vehicle cannot exchange identities.  Separate
    association pools prevent that failure while keeping fine-grained
    car/bus/truck predictions in one road-vehicle pool.  Native tracker IDs are
    remapped into a monotonic, never-reused global namespace.
    """

    def __init__(
        self,
        tracker_config: Path,
        *,
        device: str = "0",
        max_prediction_frames: int = 15,
        tracker_factory: TrackerFactory | None = None,
        boxes_factory: BoxesFactory | None = None,
    ) -> None:
        if max_prediction_frames < 0:
            raise ValueError("max_prediction_frames cannot be negative")
        self.tracker_config = tracker_config
        self.device = device
        self.max_prediction_frames = max_prediction_frames
        self._tracker_factory = tracker_factory or self._default_tracker_factory
        self._boxes_factory = boxes_factory or self._default_boxes_factory
        # Ultralytics' native track counter is shared. Construct every pool now
        # so no later tracker construction can reset that counter mid-run.
        self._trackers: dict[str, Any] = {
            group: self._tracker_factory(group) for group in TRACKING_GROUPS
        }
        self._started_groups: set[str] = set()
        self._global_ids: dict[tuple[str, int], int] = {}
        self._last_detection: dict[int, Detection] = {}
        self._last_observed_frame: dict[int, int] = {}
        self._next_global_id = 1
        self._rejected_predictions: list[RejectedTrackPrediction] = []

    def _default_tracker_factory(self, association_group: str) -> Any:
        del association_group
        if not self.tracker_config.exists():
            raise FileNotFoundError(f"BoT-SORT configuration not found: {self.tracker_config}")
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import YAML, IterableSimpleNamespace

        payload = YAML.load(str(self.tracker_config))
        if str(payload.get("tracker_type", "")).lower() != "botsort":
            raise ValueError("GroupedBoTSORT requires tracker_type: botsort")
        payload["device"] = self.device
        return BOTSORT(args=IterableSimpleNamespace(**payload))

    @staticmethod
    def _default_boxes_factory(
        detections: list[Detection], frame_shape: tuple[int, int]
    ) -> Any:
        from ultralytics.engine.results import Boxes

        matrix = np.asarray(
            [
                [
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                    detection.confidence,
                    0.0,
                ]
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape((-1, 6))
        return Boxes(matrix, frame_shape).cpu().numpy()

    def _tracker_for(self, group: str) -> Any:
        if group not in self._trackers:
            raise ValueError(f"Unsupported road-user association group: {group}")
        return self._trackers[group]

    def _global_id(self, group: str, native_id: int) -> int:
        key = (group, native_id)
        if key not in self._global_ids:
            self._global_ids[key] = self._next_global_id
            self._next_global_id += 1
        return self._global_ids[key]

    @staticmethod
    def _clip_bounds(
        bounds: Iterable[float], frame_width: int, frame_height: int
    ) -> tuple[float, float, float, float]:
        values = tuple(float(value) for value in bounds)
        if len(values) != 4:
            raise ValueError("Tracker bounding boxes must contain four coordinates")
        x1, y1, x2, y2 = values
        return (
            max(0.0, min(float(frame_width), x1)),
            max(0.0, min(float(frame_height), y1)),
            max(0.0, min(float(frame_width), x2)),
            max(0.0, min(float(frame_height), y2)),
        )

    def update(
        self, detections: Iterable[Detection], frame: np.ndarray, frame_number: int
    ) -> list[TrackObservation]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Tracker input must be a BGR image with shape HxWx3")
        if frame_number < 0:
            raise ValueError("frame_number cannot be negative")

        grouped: dict[str, list[Detection]] = defaultdict(list)
        for detection in detections:
            detection.validate()
            grouped[detection.association_group].append(detection)

        unknown_groups = set(grouped) - set(self._trackers)
        if unknown_groups:
            raise ValueError(f"Unsupported road-user association groups: {sorted(unknown_groups)}")
        self._started_groups.update(grouped)
        # Once a pool has seen an object it must also receive empty frames so
        # lost/removed timers continue advancing.
        groups = sorted(self._started_groups)
        frame_height, frame_width = frame.shape[:2]
        observations: list[TrackObservation] = []

        for group in groups:
            group_detections = grouped.get(group, [])
            tracker = self._tracker_for(group)
            tracker_boxes = self._boxes_factory(group_detections, (frame_height, frame_width))
            tracks = np.asarray(tracker.update(tracker_boxes, frame), dtype=np.float32)
            if tracks.size == 0:
                tracks = np.empty((0, 8), dtype=np.float32)
            elif tracks.ndim != 2 or tracks.shape[1] < 8:
                raise RuntimeError(
                    "Unsupported Ultralytics tracker output; expected rows ending in detection index"
                )

            observed_native_ids: set[int] = set()
            for row in tracks:
                native_id = int(row[4])
                detection_index = int(row[-1])
                if not 0 <= detection_index < len(group_detections):
                    raise RuntimeError(
                        f"BoT-SORT returned invalid detection index {detection_index} "
                        f"for {len(group_detections)} {group} detections"
                    )
                detection = group_detections[detection_index]
                global_id = self._global_id(group, native_id)
                self._last_detection[global_id] = detection
                self._last_observed_frame[global_id] = frame_number
                observed_native_ids.add(native_id)
                x1, y1, x2, y2 = self._clip_bounds(
                    row[:4], frame_width, frame_height
                )
                if x2 <= x1 or y2 <= y1:
                    raise RuntimeError(
                        f"BoT-SORT emitted invalid observed bounds for track {global_id}: "
                        f"{(x1, y1, x2, y2)}"
                    )
                observations.append(
                    TrackObservation(
                        frame=frame_number,
                        track_id=global_id,
                        native_track_id=native_id,
                        association_group=group,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(detection.confidence),
                        detected_class_id=detection.class_id,
                        detected_class_name=detection.class_name,
                        observed=True,
                        state="tracked",
                        detection_source=detection.source,
                    )
                )

            for lost_track in getattr(tracker, "lost_stracks", []):
                native_id = int(lost_track.track_id)
                if native_id in observed_native_ids:
                    continue
                key = (group, native_id)
                if key not in self._global_ids:
                    continue
                global_id = self._global_ids[key]
                last_detection = self._last_detection[global_id]
                bounds = getattr(lost_track, "xyxy", None)
                if bounds is None:
                    bounds = getattr(lost_track, "tlbr", None)
                if bounds is None:
                    raise RuntimeError("Lost BoT-SORT track exposes no predicted bounding box")
                raw_bounds = tuple(float(value) for value in bounds)
                x1, y1, x2, y2 = self._clip_bounds(
                    raw_bounds, frame_width, frame_height
                )
                prediction_age = frame_number - self._last_observed_frame[global_id]
                if prediction_age > self.max_prediction_frames:
                    self._rejected_predictions.append(
                        RejectedTrackPrediction(
                            frame=frame_number,
                            track_id=global_id,
                            native_track_id=native_id,
                            association_group=group,
                            raw_x1=raw_bounds[0],
                            raw_y1=raw_bounds[1],
                            raw_x2=raw_bounds[2],
                            raw_y2=raw_bounds[3],
                            clipped_x1=x1,
                            clipped_y1=y1,
                            clipped_x2=x2,
                            clipped_y2=y2,
                            reason="prediction_age_exceeded",
                        )
                    )
                    continue
                if x2 <= x1 or y2 <= y1:
                    self._rejected_predictions.append(
                        RejectedTrackPrediction(
                            frame=frame_number,
                            track_id=global_id,
                            native_track_id=native_id,
                            association_group=group,
                            raw_x1=raw_bounds[0],
                            raw_y1=raw_bounds[1],
                            raw_x2=raw_bounds[2],
                            raw_y2=raw_bounds[3],
                            clipped_x1=x1,
                            clipped_y1=y1,
                            clipped_x2=x2,
                            clipped_y2=y2,
                            reason="invalid_after_frame_clipping",
                        )
                    )
                    continue
                observations.append(
                    TrackObservation(
                        frame=frame_number,
                        track_id=global_id,
                        native_track_id=native_id,
                        association_group=group,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=None,
                        detected_class_id=last_detection.class_id,
                        detected_class_name=last_detection.class_name,
                        observed=False,
                        state="occluded",
                        detection_source="kalman_prediction",
                    )
                )

        return sorted(observations, key=lambda item: item.track_id)

    @property
    def removed_global_ids(self) -> set[int]:
        removed: set[int] = set()
        for group, tracker in self._trackers.items():
            for track in getattr(tracker, "removed_stracks", []):
                key = (group, int(track.track_id))
                if key in self._global_ids:
                    removed.add(self._global_ids[key])
        return removed

    @property
    def rejected_predictions(self) -> tuple[RejectedTrackPrediction, ...]:
        return tuple(self._rejected_predictions)

