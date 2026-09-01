from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


YOLO_CLASS_NAMES = (
    "pedestrian",
    "bicycle",
    "car",
    "lgv",
    "hgv",
    "bus",
    "truck",
    "motorcycle",
)
FRAME_NAME = re.compile(r"^frame_(\d{6})\.[A-Za-z0-9]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_coco_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            matches = [
                name
                for name in archive.namelist()
                if name.lower().endswith("instances_default.json")
            ]
            if len(matches) != 1:
                raise ValueError(
                    "COCO archive must contain exactly one instances_default.json"
                )
            return json.loads(archive.read(matches[0]).decode("utf-8-sig"))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_coco_payload(
    payload: dict[str, Any], *, expected_frames: int
) -> tuple[list[dict[str, Any]], dict[int, str], dict[int, list[dict[str, Any]]]]:
    images = list(payload.get("images") or [])
    categories = list(payload.get("categories") or [])
    annotations = list(payload.get("annotations") or [])
    if len(images) != expected_frames:
        raise ValueError(f"Expected {expected_frames} COCO images, found {len(images)}")
    category_names = {
        int(category["id"]): str(category["name"]).strip().lower()
        for category in categories
    }
    names = set(category_names.values())
    required = set(YOLO_CLASS_NAMES) | {"ignore"}
    if names != required:
        raise ValueError(
            f"COCO categories must be exactly {sorted(required)}, found {sorted(names)}"
        )
    ordered: list[dict[str, Any] | None] = [None] * expected_frames
    image_ids: set[int] = set()
    for image in images:
        match = FRAME_NAME.fullmatch(str(image.get("file_name", "")))
        if match is None:
            raise ValueError(f"Unsupported COCO frame name: {image.get('file_name')}")
        frame = int(match.group(1))
        if not 0 <= frame < expected_frames or ordered[frame] is not None:
            raise ValueError(f"COCO frame index is duplicated or out of range: {frame}")
        image_id = int(image["id"])
        if image_id in image_ids:
            raise ValueError(f"Duplicate COCO image id: {image_id}")
        image_ids.add(image_id)
        ordered[frame] = image
    if any(image is None for image in ordered):
        raise ValueError("COCO frame sequence is not contiguous")

    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in image_ids:
            raise ValueError(f"Annotation references unknown image id: {image_id}")
        if category_id not in category_names:
            raise ValueError(f"Annotation references unknown category id: {category_id}")
        bbox = [float(value) for value in annotation.get("bbox", [])]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"Annotation has invalid bbox: {annotation.get('id')}")
        by_image[image_id].append(annotation)
    return [image for image in ordered if image is not None], category_names, by_image


def yolo_label_lines(
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    category_names: dict[int, str],
) -> tuple[list[str], Counter[str]]:
    width = int(image["width"])
    height = int(image["height"])
    if width <= 0 or height <= 0:
        raise ValueError("COCO image dimensions must be positive")
    lines: list[str] = []
    counts: Counter[str] = Counter()
    for annotation in annotations:
        name = category_names[int(annotation["category_id"])]
        if name == "ignore":
            counts["ignore"] += 1
            continue
        class_id = YOLO_CLASS_NAMES.index(name)
        x, y, box_width, box_height = (
            float(value) for value in annotation["bbox"]
        )
        if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
            raise ValueError(f"COCO bbox lies outside image: {annotation.get('id')}")
        center_x = (x + box_width / 2.0) / width
        center_y = (y + box_height / 2.0) / height
        lines.append(
            f"{class_id} {center_x:.8f} {center_y:.8f} "
            f"{box_width / width:.8f} {box_height / height:.8f}"
        )
        counts[name] += 1
    return lines, counts


def prepare_yolo_dataset(
    *,
    video_path: Path,
    coco_path: Path,
    output_dir: Path,
    expected_video_sha256: str,
    expected_frames: int = 89,
    validation_start_frame: int = 71,
    all_train: bool = False,
) -> dict[str, Any]:
    import cv2

    actual_video_hash = sha256_file(video_path)
    if actual_video_hash.lower() != expected_video_sha256.lower():
        raise ValueError(
            f"Video SHA256 mismatch: expected {expected_video_sha256}, got {actual_video_hash}"
        )
    if not all_train and not 0 < validation_start_frame < expected_frames:
        raise ValueError("validation_start_frame must leave non-empty train and val splits")
    payload = load_coco_payload(coco_path)
    images, category_names, annotations = validate_coco_payload(
        payload, expected_frames=expected_frames
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_frames != expected_frames:
        capture.release()
        raise ValueError(f"Expected {expected_frames} video frames, found {source_frames}")
    counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    try:
        for frame, image in enumerate(images):
            success, pixels = capture.read()
            if not success:
                raise RuntimeError(f"Video decode failed at frame {frame}")
            height, width = pixels.shape[:2]
            if width != int(image["width"]) or height != int(image["height"]):
                raise ValueError(
                    f"Frame {frame} dimensions {width}x{height} do not match COCO"
                )
            split = "train" if all_train or frame < validation_start_frame else "val"
            stem = f"frame_{frame:06d}"
            image_path = output_dir / "images" / split / f"{stem}.png"
            if not cv2.imwrite(str(image_path), pixels):
                raise RuntimeError(f"Could not write frame image: {image_path}")
            lines, frame_counts = yolo_label_lines(
                image, annotations.get(int(image["id"]), []), category_names
            )
            (output_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            counts.update(frame_counts)
            split_counts[split] += 1
    finally:
        capture.release()

    yaml_lines = [
        "train: images/train",
        f"val: images/{'train' if all_train else 'val'}",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(YOLO_CLASS_NAMES)],
        "",
    ]
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "role": "development_training_only",
        "video": str(video_path),
        "video_sha256": actual_video_hash,
        "coco_annotations": str(coco_path),
        "coco_sha256": sha256_file(coco_path),
        "frames": expected_frames,
        "split": {
            "train_frames": [0, expected_frames - 1 if all_train else validation_start_frame - 1],
            "validation_frames": (
                "training_frames_reused_for_refit_only"
                if all_train
                else [validation_start_frame, expected_frames - 1]
            ),
            "train_count": split_counts["train"],
            "validation_count": split_counts["val"],
            "all_train_refit": all_train,
        },
        "class_names": list(YOLO_CLASS_NAMES),
        "annotation_counts": dict(sorted(counts.items())),
        "ignored_annotations_omitted": counts["ignore"],
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
