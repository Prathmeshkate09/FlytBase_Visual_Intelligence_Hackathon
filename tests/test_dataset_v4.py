from __future__ import annotations

import pytest

from traffic_agent.dataset_v4 import validate_coco_payload, yolo_label_lines


def _payload() -> dict:
    return {
        "images": [
            {"id": 2, "file_name": "frame_000001.png", "width": 100, "height": 50},
            {"id": 1, "file_name": "frame_000000.png", "width": 100, "height": 50},
        ],
        "categories": [
            {"id": index + 1, "name": name}
            for index, name in enumerate(
                ["car", "lgv", "hgv", "bus", "truck", "motorcycle", "bicycle", "pedestrian", "ignore"]
            )
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]},
            {"id": 2, "image_id": 1, "category_id": 9, "bbox": [40, 10, 10, 10]},
        ],
    }


def test_coco_validation_orders_frames_and_omits_ignore_from_yolo() -> None:
    images, categories, annotations = validate_coco_payload(
        _payload(), expected_frames=2
    )
    lines, counts = yolo_label_lines(images[0], annotations[1], categories)

    assert [image["id"] for image in images] == [1, 2]
    assert lines == ["2 0.20000000 0.20000000 0.20000000 0.20000000"]
    assert counts["car"] == 1
    assert counts["ignore"] == 1


def test_coco_validation_rejects_noncontiguous_frames() -> None:
    payload = _payload()
    payload["images"][0]["file_name"] = "frame_000003.png"

    with pytest.raises(ValueError, match="out of range"):
        validate_coco_payload(payload, expected_frames=2)


def test_yolo_conversion_rejects_box_outside_image() -> None:
    images, categories, annotations = validate_coco_payload(
        _payload(), expected_frames=2
    )
    annotations[1][0]["bbox"] = [95, 0, 10, 5]

    with pytest.raises(ValueError, match="outside image"):
        yolo_label_lines(images[0], annotations[1], categories)
