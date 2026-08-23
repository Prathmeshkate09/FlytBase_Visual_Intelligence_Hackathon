from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


ASSOCIATION_GROUPS = {
    # Vulnerable road users share one category-agnostic association pool. This
    # preserves identity when a tiny rider alternates between person, bicycle,
    # and motorcycle detector labels. Final class is resolved temporally.
    "person": "vru",
    "people": "vru",
    "pedestrian": "vru",
    "bicycle": "vru",
    "cyclist": "vru",
    "motorcycle": "vru",
    "motorbike": "vru",
    "motor": "vru",
    "tricycle": "vru",
    "awning-tricycle": "vru",
    "car": "road_vehicle",
    "lgv": "road_vehicle",
    "van": "road_vehicle",
    "hgv": "road_vehicle",
    "bus": "road_vehicle",
    "truck": "road_vehicle",
}

DEFAULT_ROAD_USER_CLASSES = tuple(ASSOCIATION_GROUPS)

# When two different association pools produce essentially the same box, keep
# the more specific road-user mode.  This is intentionally used only for
# near-identical boxes; an ordinary pedestrian overlapping a vehicle must not
# disappear merely because the boxes intersect.
PHYSICAL_DUPLICATE_PRIORITY = {
    "vru": 1,
    "road_vehicle": 2,
}

VRU_CLASS_PRIORITY = {
    "person": 0,
    "people": 0,
    "pedestrian": 0,
    "bicycle": 1,
    "cyclist": 1,
    "motorcycle": 2,
    "motorbike": 2,
    "motor": 2,
    "tricycle": 2,
    "awning-tricycle": 2,
}


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    source: str = "standard"
    source_classes: tuple[str, ...] = ()
    merge_count: int = 1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def ground_point(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def association_group(self) -> str:
        return ASSOCIATION_GROUPS.get(self.class_name.lower(), self.class_name.lower())

    def validate(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2, self.confidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Detection contains non-finite values")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Detection bounding box must have positive area")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be in [0, 1]")
        if not self.class_name.strip():
            raise ValueError("Detection class_name cannot be empty")

    def clipped(self, frame_width: int, frame_height: int) -> "Detection":
        return replace(
            self,
            x1=max(0.0, min(float(frame_width), self.x1)),
            y1=max(0.0, min(float(frame_height), self.y1)),
            x2=max(0.0, min(float(frame_width), self.x2)),
            y2=max(0.0, min(float(frame_height), self.y2)),
        )


@dataclass(frozen=True)
class RejectedDetection:
    detection: Detection
    reason: str


@dataclass(frozen=True)
class PostprocessConfig:
    allowed_class_names: tuple[str, ...] = DEFAULT_ROAD_USER_CLASSES
    merge_iou_threshold: float = 0.75
    merge_ios_threshold: float = 0.90
    minimum_box_area_px2: float = 16.0
    physical_duplicate_iou_threshold: float = 0.90
    physical_duplicate_ios_threshold: float = 0.98
    physical_duplicate_area_ratio_threshold: float = 0.80
    suppress_rider_duplicates: bool = True
    rider_overlap_ios_threshold: float = 0.35
    rider_ground_margin_ratio: float = 0.10

    def validate(self) -> None:
        if not self.allowed_class_names:
            raise ValueError("At least one road-user class must be allowed")
        if not 0.0 < self.merge_iou_threshold <= 1.0:
            raise ValueError("merge_iou_threshold must be in (0, 1]")
        if not 0.0 < self.merge_ios_threshold <= 1.0:
            raise ValueError("merge_ios_threshold must be in (0, 1]")
        if self.minimum_box_area_px2 < 0:
            raise ValueError("minimum_box_area_px2 cannot be negative")
        if not 0.0 < self.physical_duplicate_iou_threshold <= 1.0:
            raise ValueError("physical_duplicate_iou_threshold must be in (0, 1]")
        if not 0.0 < self.physical_duplicate_ios_threshold <= 1.0:
            raise ValueError("physical_duplicate_ios_threshold must be in (0, 1]")
        if not 0.0 < self.physical_duplicate_area_ratio_threshold <= 1.0:
            raise ValueError("physical_duplicate_area_ratio_threshold must be in (0, 1]")
        if not 0.0 < self.rider_overlap_ios_threshold <= 1.0:
            raise ValueError("rider_overlap_ios_threshold must be in (0, 1]")
        if not 0.0 <= self.rider_ground_margin_ratio <= 1.0:
            raise ValueError("rider_ground_margin_ratio must be in [0, 1]")


def _point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


@dataclass(frozen=True)
class RoadUserROI:
    include_polygons: tuple[tuple[tuple[float, float], ...], ...] = ()
    exclude_polygons: tuple[tuple[tuple[float, float], ...], ...] = ()
    coordinate_space: str = "normalized"

    def validate(self) -> None:
        if self.coordinate_space not in {"normalized", "pixels"}:
            raise ValueError("ROI coordinate_space must be normalized or pixels")
        for polygon in (*self.include_polygons, *self.exclude_polygons):
            if len(polygon) < 3:
                raise ValueError("Every ROI polygon must contain at least three points")
            for x, y in polygon:
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("ROI coordinates must be finite")
                if self.coordinate_space == "normalized" and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ValueError("Normalized ROI coordinates must be in [0, 1]")

    @classmethod
    def from_json(cls, path: Path) -> "RoadUserROI":
        if not path.exists():
            raise FileNotFoundError(f"Road-user ROI file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))

        def polygons(key: str) -> tuple[tuple[tuple[float, float], ...], ...]:
            return tuple(
                tuple((float(point[0]), float(point[1])) for point in polygon)
                for polygon in payload.get(key, [])
            )

        roi = cls(
            include_polygons=polygons("include_polygons"),
            exclude_polygons=polygons("exclude_polygons"),
            coordinate_space=str(payload.get("coordinate_space", "normalized")),
        )
        roi.validate()
        return roi

    def contains(self, point: tuple[float, float], frame_width: int, frame_height: int) -> bool:
        self.validate()
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if self.coordinate_space == "normalized":
            candidate = (point[0] / frame_width, point[1] / frame_height)
        else:
            candidate = point
        included = not self.include_polygons or any(
            _point_in_polygon(candidate, polygon) for polygon in self.include_polygons
        )
        excluded = any(_point_in_polygon(candidate, polygon) for polygon in self.exclude_polygons)
        return included and not excluded


def box_iou(first: Detection, second: Detection) -> float:
    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(
        0.0, intersection_y2 - intersection_y1
    )
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def box_ios(first: Detection, second: Detection) -> float:
    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(
        0.0, intersection_y2 - intersection_y1
    )
    smaller_area = min(first.area, second.area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _merge_cluster(
    cluster: list[Detection], *, winner: Detection | None = None
) -> Detection:
    winner = winner or max(
        cluster,
        key=lambda detection: (
            VRU_CLASS_PRIORITY.get(detection.class_name.lower(), -1)
            if detection.association_group == "vru"
            else 0,
            detection.confidence,
        ),
    )
    weights = [max(detection.confidence, 1e-6) for detection in cluster]
    total_weight = sum(weights)

    def weighted(attribute: str) -> float:
        return sum(getattr(detection, attribute) * weight for detection, weight in zip(cluster, weights)) / total_weight

    sources = sorted({source for detection in cluster for source in detection.source.split("+")})
    classes = sorted(
        {
            class_name
            for detection in cluster
            for class_name in (detection.source_classes or (detection.class_name,))
        }
    )
    return Detection(
        x1=weighted("x1"),
        y1=weighted("y1"),
        x2=weighted("x2"),
        y2=weighted("y2"),
        confidence=winner.confidence,
        class_id=winner.class_id,
        class_name=winner.class_name,
        source="+".join(sources),
        source_classes=tuple(classes),
        merge_count=sum(detection.merge_count for detection in cluster),
    )


def _duplicate_clusters(
    detections: Iterable[Detection], config: PostprocessConfig
) -> list[list[Detection]]:
    config.validate()
    items = list(detections)
    for detection in items:
        detection.validate()
    if not items:
        return []

    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(items):
        for second_index in range(first_index + 1, len(items)):
            second = items[second_index]
            if first.association_group != second.association_group:
                continue
            if box_iou(first, second) >= config.merge_iou_threshold or box_ios(
                first, second
            ) >= config.merge_ios_threshold:
                union(first_index, second_index)

    clusters: dict[int, list[Detection]] = {}
    for index, detection in enumerate(items):
        clusters.setdefault(find(index), []).append(detection)
    return list(clusters.values())


def _physical_duplicate_clusters(
    detections: Iterable[Detection], config: PostprocessConfig
) -> list[list[Detection]]:
    """Cluster near-identical boxes emitted into different tracker pools.

    A rider can be predicted as both ``person`` and ``car``/``motorcycle`` with
    virtually identical geometry.  Group-specific association would otherwise
    create two persistent IDs for that one physical road user.  Requiring both
    very high overlap and comparable areas avoids suppressing a real pedestrian
    who is merely standing beside or partially inside a vehicle box.
    """
    config.validate()
    items = list(detections)
    if not items:
        return []
    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(items):
        for second_index in range(first_index + 1, len(items)):
            second = items[second_index]
            if first.association_group == second.association_group:
                continue
            area_ratio = min(first.area, second.area) / max(first.area, second.area)
            near_identical = box_iou(first, second) >= config.physical_duplicate_iou_threshold
            near_equal_containment = (
                box_ios(first, second) >= config.physical_duplicate_ios_threshold
                and area_ratio >= config.physical_duplicate_area_ratio_threshold
            )
            if near_identical or near_equal_containment:
                union(first_index, second_index)

    clusters: dict[int, list[Detection]] = {}
    for index, detection in enumerate(items):
        clusters.setdefault(find(index), []).append(detection)
    return list(clusters.values())


def _physical_winner(cluster: list[Detection]) -> Detection:
    return max(
        cluster,
        key=lambda detection: (
            PHYSICAL_DUPLICATE_PRIORITY.get(detection.association_group, -1),
            detection.confidence,
        ),
    )


def _is_rider_duplicate(
    pedestrian: Detection,
    motorcycle: Detection,
    config: PostprocessConfig,
) -> bool:
    """Return whether a pedestrian box represents a motorcycle rider.

    Aerial detectors commonly emit one box for the motorcycle and another for
    its rider. Counting both boxes creates two identities for one physical road
    user. Requiring both box overlap and the rider ground point to fall inside
    the motorcycle footprint preserves a person merely standing beside it.
    """
    if pedestrian.class_name.lower() not in {"person", "people", "pedestrian"}:
        return False
    if motorcycle.class_name.lower() not in {
        "motorcycle",
        "motorbike",
        "motor",
        "tricycle",
        "awning-tricycle",
    }:
        return False
    if box_ios(pedestrian, motorcycle) < config.rider_overlap_ios_threshold:
        return False
    ground_x, ground_y = pedestrian.ground_point
    vertical_margin = motorcycle.height * config.rider_ground_margin_ratio
    return (
        motorcycle.x1 <= ground_x <= motorcycle.x2
        and motorcycle.y1 <= ground_y <= motorcycle.y2 + vertical_margin
    )


def _suppress_rider_duplicates(
    detections: list[Detection], config: PostprocessConfig
) -> tuple[list[Detection], list[RejectedDetection]]:
    if not config.suppress_rider_duplicates:
        return detections, []
    motorcycles = [
        detection
        for detection in detections
        if detection.class_name.lower()
        in {"motorcycle", "motorbike", "motor", "tricycle", "awning-tricycle"}
    ]
    if not motorcycles:
        return detections, []

    accepted: list[Detection] = []
    rejected: list[RejectedDetection] = []
    for detection in detections:
        if any(
            _is_rider_duplicate(detection, motorcycle, config)
            for motorcycle in motorcycles
        ):
            rejected.append(RejectedDetection(detection, "rider_duplicate_suppressed"))
        else:
            accepted.append(detection)
    return accepted, rejected


def merge_group_duplicates(
    detections: Iterable[Detection], config: PostprocessConfig
) -> list[Detection]:
    """Merge overlapping boxes only inside the same road-user association group."""
    clusters = _duplicate_clusters(detections, config)
    merged = [_merge_cluster(cluster) for cluster in clusters]
    return sorted(merged, key=lambda detection: detection.confidence, reverse=True)


def postprocess_detections(
    detections: Iterable[Detection],
    *,
    frame_width: int,
    frame_height: int,
    config: PostprocessConfig | None = None,
    roi: RoadUserROI | None = None,
) -> tuple[list[Detection], list[RejectedDetection]]:
    active_config = config or PostprocessConfig()
    active_config.validate()
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive")
    allowed = {name.lower() for name in active_config.allowed_class_names}
    accepted: list[Detection] = []
    rejected: list[RejectedDetection] = []

    for original in detections:
        original.validate()
        detection = original.clipped(frame_width, frame_height)
        if detection.x2 <= detection.x1 or detection.y2 <= detection.y1:
            rejected.append(RejectedDetection(original, "outside_frame"))
            continue
        if detection.class_name.lower() not in allowed:
            rejected.append(RejectedDetection(detection, "class_not_allowed"))
            continue
        if detection.area < active_config.minimum_box_area_px2:
            rejected.append(RejectedDetection(detection, "box_too_small"))
            continue
        if roi is not None and not roi.contains(detection.ground_point, frame_width, frame_height):
            rejected.append(RejectedDetection(detection, "outside_road_user_roi"))
            continue
        accepted.append(detection)

    group_clusters = _duplicate_clusters(accepted, active_config)
    group_merged = [_merge_cluster(cluster) for cluster in group_clusters]
    for cluster in group_clusters:
        if len(cluster) <= 1:
            continue
        winner = max(cluster, key=lambda detection: detection.confidence)
        rejected.extend(
            RejectedDetection(detection, "duplicate_suppressed")
            for detection in cluster
            if detection is not winner
        )

    rider_filtered, rider_rejections = _suppress_rider_duplicates(
        group_merged, active_config
    )
    rejected.extend(rider_rejections)

    physical_clusters = _physical_duplicate_clusters(rider_filtered, active_config)
    merged: list[Detection] = []
    for cluster in physical_clusters:
        winner = _physical_winner(cluster)
        merged.append(_merge_cluster(cluster, winner=winner))
        if len(cluster) > 1:
            rejected.extend(
                RejectedDetection(detection, "physical_duplicate_suppressed")
                for detection in cluster
                if detection is not winner
            )

    merged.sort(key=lambda detection: detection.confidence, reverse=True)
    return merged, rejected

