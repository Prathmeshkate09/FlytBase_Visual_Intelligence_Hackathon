from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY = "https://github.com/Prathmeshkate09/FlytBase_Visual_Intelligence_Hackathon.git"
BRANCH = "level1-v4"
VIDEO_SHA256 = "c8ead5bc7f3fd82dfd3dfe345061996f8822e9a8bea2048b16b58d7b7edbeda1"
CANDIDATES_SHA256 = "99e047847b9f65f439af640aeccdae99f2f819cdcca5cf5e86321818cd1c4a6e"
MOT_SHA256 = "b10b5e92988aff11848a18b35e1f6b830f5ca4861d631ff20aad7cce90edac45"
DETECTOR_CONFIG_SHA256 = "8462f2b2a7ee77f8793c35f36810de5302eb2d09c99ffef99b0ef179d496fe84"

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/flytbase_v4_tuning")
REPO_DIR = WORK_ROOT / "repo"
OUTPUT_DIR = WORK_ROOT / "scene_full_1920"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_hashed_file(name: str, expected_sha256: str, *, required_part: str = "") -> Path:
    matches = []
    for path in INPUT_ROOT.rglob(name):
        if required_part and required_part not in path.as_posix():
            continue
        if sha256_file(path).lower() == expected_sha256.lower():
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one hash-matched {name!r}, found {len(matches)}: {matches}"
        )
    print(f"Verified {name}: {matches[0]} ({expected_sha256})")
    return matches[0]


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if not INPUT_ROOT.exists():
        raise RuntimeError("This runner must execute inside Kaggle")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    video = find_hashed_file(
        "source_video_89f_dev_c8ead5.mp4", VIDEO_SHA256, required_part="flytbase-v4-gpu-runner"
    )
    candidates = find_hashed_file(
        "candidate_detections.csv",
        CANDIDATES_SHA256,
        required_part="dev89_scene_full_1920",
    )
    mot = find_hashed_file(
        "FlytBase_L1_dev89_corrected_MOT.zip",
        MOT_SHA256,
        required_part="flytbase-v4-gpu-runner",
    )
    detector_config = find_hashed_file(
        "detection_scene_full_1920.json",
        DETECTOR_CONFIG_SHA256,
        required_part="flytbase-v4-gpu-runner",
    )

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(REPO_DIR)])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(REPO_DIR / "requirements-eval.txt"),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_tuning_v4.py",
            "tests/test_pipeline_v4.py",
            "tests/test_evaluation_v4.py",
        ],
        cwd=REPO_DIR,
    )

    run(
        [
            sys.executable,
            str(REPO_DIR / "tune_v4.py"),
            "--input",
            str(video),
            "--expected-video-sha256",
            VIDEO_SHA256,
            "--candidate-detections",
            str(candidates),
            "--mot-ground-truth",
            str(mot),
            "--detection-config",
            str(detector_config),
            "--tracker-config",
            str(REPO_DIR / "config/botsort_drone_v4.yaml"),
            "--road-user-roi",
            str(REPO_DIR / "config/road_user_roi_intersection_v4.json"),
            "--output",
            str(OUTPUT_DIR),
            "--max-seconds",
            "3.0",
        ],
        cwd=REPO_DIR,
    )

    summary_path = OUTPUT_DIR / "tuning_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("TUNING_RESULT_BEGIN")
    print(json.dumps(summary, indent=2))
    print("TUNING_RESULT_END")
    archive = shutil.make_archive(
        "/kaggle/working/FlytBase_V4_Scene_Full_1920_Tuning",
        "zip",
        root_dir=OUTPUT_DIR,
    )
    print("Tuning archive:", archive)


if __name__ == "__main__":
    main()
