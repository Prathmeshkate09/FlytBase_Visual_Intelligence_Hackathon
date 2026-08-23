from __future__ import annotations

import json
from pathlib import Path

import pytest

from traffic_agent.evaluation_v4 import sha256_file
from traffic_agent.level2 import validate_level1_dependency


def _dependency_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tracks = tmp_path / "tracks.csv"
    tracks.write_text("frame,track_id\n0,1\n", encoding="utf-8")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"verified-video")
    tracks_hash = sha256_file(tracks)
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "tracks_sha256": tracks_hash,
                "input_video_sha256": sha256_file(video),
            }
        ),
        encoding="utf-8",
    )
    quality = tmp_path / "quality_report_ground_truth.json"
    quality.write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "quality_status": "passed",
                "gate_results": {
                    "detection_precision": True,
                    "detection_recall": True,
                    "idf1": True,
                    "hota": True,
                    "mode_accuracy": True,
                },
                "provenance": {"tracks_sha256": tracks_hash},
            }
        ),
        encoding="utf-8",
    )
    return tracks, video, manifest, quality


def test_level2_accepts_exact_passing_level1_run(tmp_path: Path) -> None:
    tracks, video, manifest, quality = _dependency_files(tmp_path)

    dependency = validate_level1_dependency(
        tracks_path=tracks,
        video_path=video,
        run_manifest_path=manifest,
        quality_report_path=quality,
    )

    assert dependency["run_id"] == "run-123"
    assert dependency["quality_status"] == "passed"


def test_level2_rejects_unverified_level1(tmp_path: Path) -> None:
    tracks, video, manifest, quality = _dependency_files(tmp_path)
    payload = json.loads(quality.read_text(encoding="utf-8"))
    payload["quality_status"] = "not_evaluated"
    quality.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="quality status is not passed"):
        validate_level1_dependency(
            tracks_path=tracks,
            video_path=video,
            run_manifest_path=manifest,
            quality_report_path=quality,
        )


def test_level2_rejects_changed_tracks(tmp_path: Path) -> None:
    tracks, video, manifest, quality = _dependency_files(tmp_path)
    tracks.write_text("frame,track_id\n0,999\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="tracks hash does not match run manifest"):
        validate_level1_dependency(
            tracks_path=tracks,
            video_path=video,
            run_manifest_path=manifest,
            quality_report_path=quality,
        )


def test_level2_rejects_different_video(tmp_path: Path) -> None:
    tracks, video, manifest, quality = _dependency_files(tmp_path)
    video.write_bytes(b"different-video")

    with pytest.raises(RuntimeError, match="source-video hash"):
        validate_level1_dependency(
            tracks_path=tracks,
            video_path=video,
            run_manifest_path=manifest,
            quality_report_path=quality,
        )
