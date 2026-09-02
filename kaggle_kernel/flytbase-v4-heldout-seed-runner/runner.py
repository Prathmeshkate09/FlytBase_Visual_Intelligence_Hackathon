from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPOSITORY = "https://github.com/Prathmeshkate09/FlytBase_Visual_Intelligence_Hackathon.git"
BRANCH = "level1-v4"
VIDEO_SHA256 = "da399b57b6faa978115ec2d936a1670d4f7e995298b9cb508070c5e9e40168ee"
MODEL_SHA256 = "02382dcc750f479ea23bfb6e51f82cbf257b1ffd95883b6b05b61822b3ea36c4"
TRACKER_CONFIG_SHA256 = "951f0d4166bed940f2fa1135516bd441fc1ea5cf06a221612e35a8f294357db6"

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/flytbase_v4_heldout_seed")
REPO_DIR = WORK_ROOT / "repo"
RUN_DIR = WORK_ROOT / "inference"
SEED_DIR = WORK_ROOT / "cvat_seed"
DELIVERY_DIR = WORK_ROOT / "delivery"
ARCHIVE_PATH = Path("/kaggle/working/FlytBase_L1_Heldout_Frozen_Seed.zip")


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


def ensure_compatible_gpu_runtime() -> str:
    capability = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()[0]
    print("Assigned GPU compute capability:", capability)
    if int(capability.split(".", maxsplit=1)[0]) < 7:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--force-reinstall",
                "torch==2.5.1",
                "torchvision==0.20.1",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        )
    return capability


def write_delivery_archive() -> dict[str, object]:
    DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    retained_files = {
        "annotations.xml": SEED_DIR / "annotations.xml",
        "ANNOTATION_INSTRUCTIONS.md": SEED_DIR / "ANNOTATION_INSTRUCTIONS.md",
        "annotation_manifest.json": SEED_DIR / "annotation_manifest.json",
        "cvat_seed_annotations.zip": SEED_DIR / "cvat_seed_annotations.zip",
        "tracks.csv": RUN_DIR / "tracks.csv",
        "track_summary.csv": RUN_DIR / "track_summary.csv",
        "quality_report.json": RUN_DIR / "quality_report.json",
        "run_manifest.json": RUN_DIR / "run_manifest.json",
        "frozen_detector_config.json": WORK_ROOT / "runtime_detector_config.json",
        "runtime_environment.json": WORK_ROOT / "runtime_environment.json",
        "development_freeze.json": REPO_DIR / "config/level1_development_freeze.json",
    }
    for name, source in retained_files.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, DELIVERY_DIR / name)

    provenance = {
        "purpose": "held-out CVAT seed only; no held-out labels were loaded or evaluated",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
        ).strip(),
        "heldout_video_sha256": VIDEO_SHA256,
        "frozen_model_sha256": MODEL_SHA256,
        "tracker_config_sha256": TRACKER_CONFIG_SHA256,
        "inference": {
            "image_size": 1920,
            "class_confidence": 0.4,
            "road_user_roi": None,
            "confirmation_observations": 5,
            "max_visible_prediction_frames": 2,
            "offline_stitching": True,
            "frames": 89,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(DELIVERY_DIR.iterdir())
            if path.is_file()
        },
    }
    provenance_path = DELIVERY_DIR / "seed_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DELIVERY_DIR.iterdir()):
            archive.write(path, path.name)
    result = {
        **provenance,
        "archive": {
            "name": ARCHIVE_PATH.name,
            "bytes": ARCHIVE_PATH.stat().st_size,
            "sha256": sha256_file(ARCHIVE_PATH),
        },
    }
    return result


def main() -> None:
    if not INPUT_ROOT.exists():
        raise RuntimeError("This runner must execute inside Kaggle")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    video = find_hashed_file(
        "source_video_89f_heldout_da399b.mp4",
        VIDEO_SHA256,
        required_part="flytbase-v4-inputs",
    )
    model = find_hashed_file(
        "last.pt",
        MODEL_SHA256,
        required_part="intersection_refit_all89/weights",
    )

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(REPO_DIR)])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")])
    gpu_capability = ensure_compatible_gpu_runtime()
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_pipeline_v4.py",
            "tests/test_cvat.py",
            "tests/test_project_context.py",
            "-k",
            "not registered_local_dataset_assets_exist",
        ],
        cwd=REPO_DIR,
    )

    tracker_config = REPO_DIR / "config/botsort_drone_v4.yaml"
    if sha256_file(tracker_config) != TRACKER_CONFIG_SHA256:
        raise RuntimeError("Frozen tracker configuration hash mismatch")
    freeze = json.loads(
        (REPO_DIR / "config/level1_development_freeze.json").read_text(encoding="utf-8")
    )
    if freeze["model"]["sha256"] != MODEL_SHA256:
        raise RuntimeError("Frozen model manifest hash mismatch")
    if freeze["heldout"]["video_sha256"] != VIDEO_SHA256:
        raise RuntimeError("Frozen held-out video manifest hash mismatch")
    if freeze["tracking"]["repository_lf_sha256"] != TRACKER_CONFIG_SHA256:
        raise RuntimeError("Frozen tracker manifest hash mismatch")

    detector_config = json.loads(
        (REPO_DIR / "config/detection_scene_full_1920_frozen.json").read_text(
            encoding="utf-8"
        )
    )
    detector_config["model_path"] = str(model)
    runtime_config = WORK_ROOT / "runtime_detector_config.json"
    runtime_config.write_text(json.dumps(detector_config, indent=2), encoding="utf-8")

    runtime_environment = {
        "gpu_compute_capability": gpu_capability,
        "torch_version": subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.__version__)"], text=True
        ).strip(),
    }
    (WORK_ROOT / "runtime_environment.json").write_text(
        json.dumps(runtime_environment, indent=2), encoding="utf-8"
    )

    # Probe one real frame before spending GPU time on the full 89-frame seed.
    probe_dir = WORK_ROOT / "gpu_probe"
    run(
        [
            sys.executable,
            str(REPO_DIR / "run_v4.py"),
            "--input",
            str(video),
            "--output",
            str(probe_dir),
            "--detection-config",
            str(runtime_config),
            "--tracker-config",
            str(tracker_config),
            "--max-seconds",
            "0.034",
            "--confirmation-observations",
            "5",
            "--max-prediction-frames",
            "2",
        ],
        cwd=REPO_DIR,
    )
    shutil.rmtree(probe_dir)

    run(
        [
            sys.executable,
            str(REPO_DIR / "run_v4.py"),
            "--input",
            str(video),
            "--output",
            str(RUN_DIR),
            "--detection-config",
            str(runtime_config),
            "--tracker-config",
            str(tracker_config),
            "--max-seconds",
            "3.0",
            "--confirmation-observations",
            "5",
            "--max-prediction-frames",
            "2",
        ],
        cwd=REPO_DIR,
    )
    run(
        [
            sys.executable,
            str(REPO_DIR / "prepare_cvat_annotations.py"),
            "--video",
            str(video),
            "--tracks",
            str(RUN_DIR / "tracks.csv"),
            "--track-summary",
            str(RUN_DIR / "track_summary.csv"),
            "--output",
            str(SEED_DIR),
            "--task-name",
            "FlytBase-L1-heldout-89f-frozen-seed",
            "--clip-role",
            "held-out",
            "--max-frames",
            "89",
        ],
        cwd=REPO_DIR,
    )

    result = write_delivery_archive()
    print("HELDOUT_SEED_RESULT_BEGIN")
    print(json.dumps(result, indent=2))
    print("HELDOUT_SEED_RESULT_END")

    # Keep only the small, self-contained delivery archive in Kaggle output.
    shutil.rmtree(WORK_ROOT)
    print("Held-out seed archive:", ARCHIVE_PATH)


if __name__ == "__main__":
    main()
