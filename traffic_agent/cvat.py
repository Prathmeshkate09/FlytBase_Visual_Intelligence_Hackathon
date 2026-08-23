from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


CVAT_LABELS = (
    "car",
    "lgv",
    "hgv",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "ignore",
)

SEED_CLASS_MAP = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "cyclist": "bicycle",
    "person": "pedestrian",
    "pedestrian": "pedestrian",
}

REQUIRED_TRACK_COLUMNS = {
    "frame",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "observed",
}
REQUIRED_SUMMARY_COLUMNS = {"track_id", "final_class_name"}


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    frame_count: int
    fps: float

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.frame_count <= 0:
            raise ValueError("Video width, height and frame count must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Video FPS must be a positive finite value")


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_attribute_definition(
    label: ET.Element,
    *,
    name: str,
    input_type: str,
    mutable: bool,
    default: str,
    values: str,
) -> None:
    attribute = ET.SubElement(label, "attribute")
    ET.SubElement(attribute, "name").text = name
    ET.SubElement(attribute, "mutable").text = str(mutable).lower()
    ET.SubElement(attribute, "input_type").text = input_type
    ET.SubElement(attribute, "default_value").text = default
    ET.SubElement(attribute, "values").text = values


def _add_meta(
    root: ET.Element,
    *,
    metadata: VideoMetadata,
    task_name: str,
    track_count: int,
) -> None:
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(metadata.frame_count)
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker")
    ET.SubElement(task, "created")
    ET.SubElement(task, "updated")
    ET.SubElement(task, "subset").text = "default"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(metadata.frame_count - 1)
    ET.SubElement(task, "frame_filter")
    ET.SubElement(task, "track_count").text = str(track_count)

    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = "0"
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(metadata.frame_count - 1)
    ET.SubElement(segment, "url")

    labels = ET.SubElement(task, "labels")
    colours = (
        "#33B5E5",
        "#2ECC71",
        "#8E44AD",
        "#F39C12",
        "#27AE60",
        "#F1C40F",
        "#E67E22",
        "#E91E63",
        "#9E9E9E",
    )
    for label_name, colour in zip(CVAT_LABELS, colours):
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = label_name
        ET.SubElement(label, "color").text = colour
        ET.SubElement(label, "type").text = "any"
        attributes = ET.SubElement(label, "attributes")
        _add_attribute_definition(
            attributes,
            name="v4_track_id",
            input_type="text",
            mutable=False,
            default="",
            values="",
        )
        _add_attribute_definition(
            attributes,
            name="seed_state",
            input_type="select",
            mutable=True,
            default="observed",
            values="observed\nkalman_prediction",
        )
        _add_attribute_definition(
            attributes,
            name="review_status",
            input_type="select",
            mutable=True,
            default="unchecked",
            values="unchecked\nconfirmed\ncorrected",
        )

    original_size = ET.SubElement(task, "original_size")
    ET.SubElement(original_size, "width").text = str(metadata.width)
    ET.SubElement(original_size, "height").text = str(metadata.height)

    ET.SubElement(meta, "dumped")


def _add_box_attribute(box: ET.Element, name: str, value: str) -> None:
    attribute = ET.SubElement(box, "attribute", {"name": name})
    attribute.text = value


def _box_attributes(row: pd.Series, *, outside: bool = False) -> dict[str, str]:
    return {
        "frame": str(int(row["frame"])),
        "outside": "1" if outside else "0",
        "occluded": "1" if not _bool_value(row["observed"]) else "0",
        "keyframe": "1",
        "xtl": f"{float(row['x1']):.3f}",
        "ytl": f"{float(row['y1']):.3f}",
        "xbr": f"{float(row['x2']):.3f}",
        "ybr": f"{float(row['y2']):.3f}",
        "z_order": "0",
    }


def build_cvat_interpolation_xml(
    tracks: pd.DataFrame,
    summaries: pd.DataFrame,
    *,
    metadata: VideoMetadata,
    task_name: str,
) -> tuple[bytes, dict[str, object]]:
    """Build a CVAT interpolation annotation file from V4 seed predictions.

    The result is deliberately labelled as model-generated seed data. It must
    be manually corrected before being used as detection or tracking truth.
    """
    metadata.validate()
    _require_columns(tracks, REQUIRED_TRACK_COLUMNS, "tracks")
    _require_columns(summaries, REQUIRED_SUMMARY_COLUMNS, "track_summary")
    if tracks.empty:
        raise ValueError("tracks cannot be empty")
    if summaries.empty:
        raise ValueError("track_summary cannot be empty")

    working = tracks.copy()
    working["frame"] = pd.to_numeric(working["frame"], errors="raise").astype(int)
    working["track_id"] = pd.to_numeric(
        working["track_id"], errors="raise"
    ).astype(int)
    if (working["frame"] < 0).any() or (working["frame"] >= metadata.frame_count).any():
        raise ValueError("Track frames must lie inside the source video")
    for column in ("x1", "y1", "x2", "y2"):
        working[column] = pd.to_numeric(working[column], errors="raise")
    invalid = (
        (working["x1"] < 0)
        | (working["y1"] < 0)
        | (working["x2"] > metadata.width)
        | (working["y2"] > metadata.height)
        | (working["x2"] <= working["x1"])
        | (working["y2"] <= working["y1"])
    )
    if invalid.any():
        raise ValueError(f"tracks contains {int(invalid.sum())} invalid boxes")

    class_lookup = {
        int(row.track_id): SEED_CLASS_MAP.get(
            str(row.final_class_name).strip().lower(), "ignore"
        )
        for row in summaries.itertuples(index=False)
    }
    missing_summaries = sorted(set(working["track_id"]) - set(class_lookup))
    if missing_summaries:
        raise ValueError(
            f"track_summary is missing final classes for IDs: {missing_summaries[:10]}"
        )

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    unique_ids = sorted(int(value) for value in working["track_id"].unique())
    _add_meta(
        root,
        metadata=metadata,
        task_name=task_name,
        track_count=len(unique_ids),
    )

    observed_boxes = 0
    predicted_boxes = 0
    outside_boxes = 0
    class_counts: dict[str, int] = {}
    for cvat_id, source_track_id in enumerate(unique_ids):
        label = class_lookup[source_track_id]
        class_counts[label] = class_counts.get(label, 0) + 1
        track_node = ET.SubElement(
            root,
            "track",
            {
                "id": str(cvat_id),
                "label": label,
                "source": "auto",
            },
        )
        group = working[working["track_id"] == source_track_id].copy()
        group["_observed"] = group["observed"].map(_bool_value)
        group = (
            group.sort_values(["frame", "_observed"])
            .drop_duplicates("frame", keep="last")
            .sort_values("frame")
        )
        for _, row in group.iterrows():
            observed = _bool_value(row["observed"])
            observed_boxes += int(observed)
            predicted_boxes += int(not observed)
            box = ET.SubElement(track_node, "box", _box_attributes(row))
            _add_box_attribute(box, "v4_track_id", str(source_track_id))
            _add_box_attribute(
                box,
                "seed_state",
                "observed" if observed else "kalman_prediction",
            )
            _add_box_attribute(box, "review_status", "unchecked")

        last = group.iloc[-1].copy()
        last_frame = int(last["frame"])
        if last_frame < metadata.frame_count - 1:
            last["frame"] = last_frame + 1
            outside_attributes = _box_attributes(last, outside=True)
            outside_attributes["occluded"] = "0"
            outside = ET.SubElement(
                track_node, "box", outside_attributes
            )
            _add_box_attribute(outside, "v4_track_id", str(source_track_id))
            _add_box_attribute(outside, "seed_state", "observed")
            _add_box_attribute(outside, "review_status", "unchecked")
            outside_boxes += 1

    try:
        ET.indent(root, space="  ")
    except AttributeError:  # pragma: no cover - Python 3.8 compatibility
        pass
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    report: dict[str, object] = {
        "schema_version": 1,
        "task_name": task_name,
        "ground_truth_status": "seed_predictions_only",
        "manual_review_required": True,
        "video": asdict(metadata),
        "tracks": len(unique_ids),
        "observed_seed_boxes": observed_boxes,
        "predicted_seed_boxes": predicted_boxes,
        "outside_markers": outside_boxes,
        "seed_tracks_by_class": dict(sorted(class_counts.items())),
    }
    return xml_bytes, report


def annotation_instructions(
    task_name: str, clip_role: str, metadata: VideoMetadata
) -> str:
    return f"""# {task_name} annotation instructions

This package contains model predictions only. They are not ground truth until
the entire clip has been reviewed and corrected in CVAT.

Clip role: **{clip_role}**

1. Create a CVAT task in **interpolation** mode using the exact source video.
   Configure start frame `0` and stop frame `{metadata.frame_count - 1}`.
2. Import `annotations.xml` using the **CVAT for video 1.1** format.
3. Review every frame from beginning to end. Do not validate only the seeded boxes.
4. Add every missing road user, delete false roof/building tracks, and repair ID switches.
5. One physical road user keeps one ID from first visibility until exit. Mark real
   temporary occlusions instead of ending and recreating the track.
6. Use: car, LGV, HGV, bus, truck, motorcycle, bicycle, pedestrian. Use `ignore`
   only for genuinely unresolvable or heavily truncated objects.
7. Correct `review_status` from `unchecked` to `confirmed` or `corrected`.
8. Recheck all in-frame track starts/ends and every overlap/crossing event.
9. Export both **CVAT for video 1.1** and **MOT 1.1** after review.
10. Keep this package separate from the corrected export so predictions can never
    be mistaken for verified labels.

Known held-out review anchors: inspect candidate handoffs 85->108 and 141->164.
Candidate 128->142 is a proxy false alarm involving different physical objects.
"""


def write_cvat_seed_package(
    *,
    video_path: Path,
    tracks_path: Path,
    summary_path: Path,
    output_dir: Path,
    metadata: VideoMetadata,
    task_name: str,
    clip_role: str,
) -> dict[str, object]:
    for path in (video_path, tracks_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_csv(tracks_path)
    summaries = pd.read_csv(summary_path)
    xml_bytes, report = build_cvat_interpolation_xml(
        tracks,
        summaries,
        metadata=metadata,
        task_name=task_name,
    )

    xml_path = output_dir / "annotations.xml"
    xml_path.write_bytes(xml_bytes)
    instructions_path = output_dir / "ANNOTATION_INSTRUCTIONS.md"
    instructions_path.write_text(
        annotation_instructions(task_name, clip_role, metadata), encoding="utf-8"
    )

    manifest = {
        **report,
        "clip_role": clip_role,
        "source_video": str(video_path),
        "source_video_sha256": _sha256(video_path),
        "source_tracks": str(tracks_path),
        "source_tracks_sha256": _sha256(tracks_path),
        "source_track_summary": str(summary_path),
        "source_track_summary_sha256": _sha256(summary_path),
        "outputs": {
            "cvat_annotations": xml_path.name,
            "instructions": instructions_path.name,
            "cvat_import_zip": "cvat_seed_annotations.zip",
        },
    }
    manifest_path = output_dir / "annotation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = output_dir / "cvat_seed_annotations.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(xml_path, xml_path.name)
        archive.write(instructions_path, instructions_path.name)
        archive.write(manifest_path, manifest_path.name)
    return manifest
