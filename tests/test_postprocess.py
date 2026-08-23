import pytest

from traffic_agent.detection import DetectorConfig
from traffic_agent.postprocess import (
    Detection,
    PostprocessConfig,
    RoadUserROI,
    box_ios,
    merge_group_duplicates,
    postprocess_detections,
)


def _detection(
    class_name: str,
    class_id: int,
    confidence: float,
    box: tuple[float, float, float, float] = (100.0, 100.0, 150.0, 140.0),
) -> Detection:
    return Detection(
        x1=box[0],
        y1=box[1],
        x2=box[2],
        y2=box[3],
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
        source="test",
        source_classes=(class_name,),
    )


def test_cross_class_vehicle_duplicates_are_merged() -> None:
    car = _detection("car", 2, 0.90)
    truck = _detection("truck", 7, 0.60, (101.0, 100.0, 151.0, 140.0))

    merged = merge_group_duplicates([car, truck], PostprocessConfig())

    assert len(merged) == 1
    assert merged[0].class_name == "car"
    assert merged[0].merge_count == 2
    assert merged[0].source_classes == ("car", "truck")


def test_duplicate_suppression_is_auditable() -> None:
    car = _detection("car", 2, 0.90)
    truck = _detection("truck", 7, 0.60, (101.0, 100.0, 151.0, 140.0))

    merged, rejected = postprocess_detections(
        [car, truck], frame_width=400, frame_height=300
    )

    assert len(merged) == 1
    assert [(item.detection.class_name, item.reason) for item in rejected] == [
        ("truck", "duplicate_suppressed")
    ]


def test_pedestrian_is_not_merged_with_overlapping_vehicle() -> None:
    pedestrian = _detection("person", 0, 0.80)
    car = _detection("car", 2, 0.90)

    merged = merge_group_duplicates([pedestrian, car], PostprocessConfig())

    assert len(merged) == 2
    assert {item.association_group for item in merged} == {"vru", "road_vehicle"}


def test_near_identical_cross_group_duplicate_prefers_vehicle() -> None:
    pedestrian = _detection("person", 0, 0.90)
    car = _detection("car", 2, 0.25, (100.5, 100.0, 150.5, 140.0))

    merged, rejected = postprocess_detections(
        [pedestrian, car], frame_width=400, frame_height=300
    )

    assert len(merged) == 1
    assert merged[0].class_name == "car"
    assert merged[0].source_classes == ("car", "person")
    assert [(item.detection.class_name, item.reason) for item in rejected] == [
        ("person", "physical_duplicate_suppressed")
    ]


def test_contained_pedestrian_is_not_suppressed_as_vehicle_duplicate() -> None:
    pedestrian = _detection("person", 0, 0.85, (115.0, 105.0, 130.0, 137.0))
    car = _detection("car", 2, 0.90, (100.0, 100.0, 150.0, 140.0))

    merged, rejected = postprocess_detections(
        [pedestrian, car], frame_width=400, frame_height=300
    )

    assert len(merged) == 2
    assert not rejected


def test_motorcycle_rider_is_suppressed_as_one_physical_road_user() -> None:
    pedestrian = _detection("pedestrian", 0, 0.72, (12.0, 8.0, 28.0, 32.0))
    motorcycle = _detection("motorcycle", 3, 0.81, (10.0, 16.0, 32.0, 36.0))

    accepted, rejected = postprocess_detections(
        [pedestrian, motorcycle], frame_width=100, frame_height=100
    )

    assert [item.class_name for item in accepted] == ["motorcycle"]
    assert [(item.detection.class_name, item.reason) for item in rejected] == [
        ("pedestrian", "rider_duplicate_suppressed")
    ]


def test_pedestrian_beside_motorcycle_is_preserved() -> None:
    pedestrian = _detection("pedestrian", 0, 0.72, (0.0, 8.0, 18.0, 34.0))
    motorcycle = _detection("motorcycle", 3, 0.81, (12.0, 16.0, 34.0, 36.0))

    accepted, rejected = postprocess_detections(
        [pedestrian, motorcycle], frame_width=100, frame_height=100
    )

    assert {item.class_name for item in accepted} == {"pedestrian", "motorcycle"}
    assert rejected == []


def test_ios_merges_tile_boundary_box_containment() -> None:
    large = _detection("car", 2, 0.70, (90.0, 90.0, 160.0, 150.0))
    contained = _detection("truck", 7, 0.80, (100.0, 100.0, 150.0, 140.0))
    assert box_ios(large, contained) == pytest.approx(1.0)

    merged = merge_group_duplicates(
        [large, contained],
        PostprocessConfig(merge_iou_threshold=0.95, merge_ios_threshold=0.90),
    )

    assert len(merged) == 1
    assert merged[0].class_name == "truck"


def test_normalized_roi_keeps_road_and_rejects_exclusion_zone() -> None:
    roi = RoadUserROI(
        include_polygons=(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),),
        exclude_polygons=(((0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)),),
    )
    excluded = _detection("car", 2, 0.90, (180.0, 100.0, 220.0, 150.0))
    accepted = _detection("person", 0, 0.80, (20.0, 20.0, 40.0, 50.0))

    kept, rejected = postprocess_detections(
        [excluded, accepted],
        frame_width=400,
        frame_height=300,
        roi=roi,
    )

    assert [item.class_name for item in kept] == ["person"]
    assert [item.reason for item in rejected] == ["outside_road_user_roi"]


def test_invalid_detection_is_rejected_explicitly() -> None:
    invalid = _detection("car", 2, 1.2)
    with pytest.raises(ValueError, match="confidence"):
        postprocess_detections([invalid], frame_width=400, frame_height=300)


def test_detector_config_covers_required_coco_road_users() -> None:
    config = DetectorConfig()
    config.validate()

    assert set(config.class_ids) == {0, 1, 2, 3, 5, 7}
    assert config.postprocess_class_agnostic is False


def test_detector_config_rejects_duplicate_class_ids() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        DetectorConfig(class_ids=(0, 2, 2)).validate()


def test_detector_config_maps_visdrone_classes_and_thresholds() -> None:
    config = DetectorConfig(
        confidence=0.05,
        class_name_map={"people": "pedestrian", "van": "lgv", "motor": "motorcycle"},
        class_confidence_thresholds={"pedestrian": 0.15, "lgv": 0.10},
    )
    config.validate()

    assert config.canonical_class_name("People") == "pedestrian"
    assert config.canonical_class_name("VAN") == "lgv"
    assert config.confidence_threshold_for("pedestrian") == pytest.approx(0.15)
    assert config.confidence_threshold_for("car") == pytest.approx(0.05)

