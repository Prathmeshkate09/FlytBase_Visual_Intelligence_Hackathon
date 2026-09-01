from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .postprocess import Detection


@dataclass(frozen=True)
class DetectorConfig:
    """Detection settings for all required Level-1 road users.

    SAHI's first merge remains category-aware.  A second, explicit merge in
    :mod:`traffic_agent.postprocess` combines conflicting fine-grained vehicle
    labels without ever merging a vehicle with a pedestrian/cyclist.
    """
    model_path: str = "yolo26l.pt"
    device: str = "0"
    confidence: float = 0.05
    iou: float = 0.50
    class_ids: tuple[int, ...] = (0, 1, 2, 3, 5, 7)
    image_size: int = 1280
    use_sahi: bool = True
    slice_height: int = 960
    slice_width: int = 960
    overlap_height_ratio: float = 0.20
    overlap_width_ratio: float = 0.20
    batch_size: int = 4
    perform_standard_prediction: bool = True
    postprocess_type: str = "GREEDYNMM"
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.50
    postprocess_class_agnostic: bool = False
    class_name_map: dict[str, str] = field(default_factory=dict)
    class_confidence_thresholds: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("model_path cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 < self.iou <= 1.0:
            raise ValueError("iou must be in (0, 1]")
        if not self.class_ids:
            raise ValueError("class_ids cannot be empty")
        if any(not isinstance(class_id, int) or class_id < 0 for class_id in self.class_ids):
            raise ValueError("class_ids must contain non-negative integers")
        if len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("class_ids cannot contain duplicates")
        if self.image_size <= 0 or self.slice_height <= 0 or self.slice_width <= 0:
            raise ValueError("Detector image dimensions must be positive")
        if not 0.0 <= self.overlap_height_ratio < 1.0 or not 0.0 <= self.overlap_width_ratio < 1.0:
            raise ValueError("SAHI overlap ratios must be in [0, 1)")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.postprocess_type not in {"GREEDYNMM", "NMM", "NMS", "LSNMS"}:
            raise ValueError("Unsupported SAHI postprocess_type")
        if self.postprocess_match_metric not in {"IOU", "IOS"}:
            raise ValueError("SAHI postprocess_match_metric must be IOU or IOS")
        if not 0.0 < self.postprocess_match_threshold <= 1.0:
            raise ValueError("postprocess_match_threshold must be in (0, 1]")
        for source_name, target_name in self.class_name_map.items():
            if not str(source_name).strip() or not str(target_name).strip():
                raise ValueError("class_name_map keys and values cannot be empty")
        for class_name, threshold in self.class_confidence_thresholds.items():
            if not str(class_name).strip():
                raise ValueError("class_confidence_thresholds keys cannot be empty")
            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError("class confidence thresholds must be in [0, 1]")

    def canonical_class_name(self, raw_class_name: str) -> str:
        normalized = raw_class_name.strip().lower()
        mapping = {
            str(source).strip().lower(): str(target).strip().lower()
            for source, target in self.class_name_map.items()
        }
        return mapping.get(normalized, normalized)

    def confidence_threshold_for(self, canonical_class_name: str) -> float:
        thresholds = {
            str(name).strip().lower(): float(value)
            for name, value in self.class_confidence_thresholds.items()
        }
        return max(self.confidence, thresholds.get(canonical_class_name.lower(), 0.0))


class RoadUserDetector:
    """Lazy standard/SAHI detector that returns global-coordinate detections."""

    def __init__(self, config: DetectorConfig):
        config.validate()
        self.config = config
        self._model: Any = None

    @property
    def _sahi_device(self) -> str:
        if self.config.device.isdigit():
            return f"cuda:{self.config.device}"
        return self.config.device

    def _load_standard_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.config.model_path)
        return self._model

    def _load_sahi_model(self) -> Any:
        if self._model is None:
            try:
                from sahi import AutoDetectionModel
            except ImportError as error:
                raise RuntimeError("SAHI mode requires the sahi package") from error
            self._model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.config.model_path,
                confidence_threshold=self.config.confidence,
                device=self._sahi_device,
                image_size=self.config.image_size,
            )
        return self._model

    def predict(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Detector input must be a BGR image with shape HxWx3")
        return self._predict_sahi(frame) if self.config.use_sahi else self._predict_standard(frame)

    def _predict_standard(self, frame: np.ndarray) -> list[Detection]:
        model = self._load_standard_model()
        result = model.predict(
            source=frame,
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            classes=list(self.config.class_ids),
            agnostic_nms=True,
            device=self.config.device,
            half=self.config.device != "cpu",
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        coordinates = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        detections: list[Detection] = []
        for bounds, confidence, class_id in zip(coordinates, confidences, classes):
            raw_class_name = str(result.names[int(class_id)]).lower()
            class_name = self.config.canonical_class_name(raw_class_name)
            detections.append(
                Detection(
                    x1=float(bounds[0]),
                    y1=float(bounds[1]),
                    x2=float(bounds[2]),
                    y2=float(bounds[3]),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name,
                    source="standard",
                    source_classes=(raw_class_name,),
                )
            )
        return detections

    def _predict_sahi(self, frame: np.ndarray) -> list[Detection]:
        try:
            from sahi.predict import get_sliced_prediction
        except ImportError as error:
            raise RuntimeError("SAHI mode requires the sahi package") from error
        result = get_sliced_prediction(
            frame,
            self._load_sahi_model(),
            slice_height=self.config.slice_height,
            slice_width=self.config.slice_width,
            overlap_height_ratio=self.config.overlap_height_ratio,
            overlap_width_ratio=self.config.overlap_width_ratio,
            perform_standard_pred=self.config.perform_standard_prediction,
            postprocess_type=self.config.postprocess_type,
            postprocess_match_metric=self.config.postprocess_match_metric,
            postprocess_match_threshold=self.config.postprocess_match_threshold,
            postprocess_class_agnostic=self.config.postprocess_class_agnostic,
            batch_size=self.config.batch_size,
            # At confidence < 0.1, SAHI may select its safer NMS/IOU path.
            # Do not force a merge here: group-aware merging happens after this
            # call and preserves overlapping road-user types.
            force_postprocess_type=False,
            verbose=0,
        )
        allowed_ids = set(self.config.class_ids)
        detections: list[Detection] = []
        for prediction in result.object_prediction_list:
            class_id = int(prediction.category.id)
            if class_id not in allowed_ids:
                continue
            raw_class_name = str(prediction.category.name).lower()
            class_name = self.config.canonical_class_name(raw_class_name)
            confidence = float(prediction.score.value)
            x1, y1, x2, y2 = (float(value) for value in prediction.bbox.to_xyxy())
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    source="sahi",
                    source_classes=(raw_class_name,),
                )
            )
        return detections


def detector_model_exists(config: DetectorConfig) -> bool:
    """Return whether ``model_path`` currently resolves to a local weight file."""
    candidate = Path(config.model_path)
    return candidate.exists() if candidate.suffix else False
