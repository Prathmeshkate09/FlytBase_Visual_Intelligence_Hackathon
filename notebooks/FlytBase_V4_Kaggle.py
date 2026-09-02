# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FlytBase V4 — Kaggle GPU workspace
#
# This notebook is the reproducible GPU runner for the `level1-v4` branch.
# It keeps responsibilities separate:
#
# - **GitHub** stores code and configuration.
# - **Kaggle Dataset inputs** store video, corrected annotations and trained weights.
# - **`/kaggle/working`** stores run outputs that should be saved as a Notebook version.
#
# Before running Cell 1, use the Kaggle notebook **Settings** panel:
#
# 1. Set **Accelerator → GPU**.
# 2. Set **Internet → On** so GitHub and Python packages can be accessed.
# 3. Use **Add Input** to attach the private FlytBase input dataset.
#
# An MP4 alone is enough for inference and frame extraction. It is **not** a
# training dataset. Fine-tuning requires corrected `images/`, `labels/` and a
# `data.yaml` file.

# %% [markdown]
# ## 1. User settings
#
# Run the notebook one cell at a time. Keep `RUN_FULL_INFERENCE` and
# `RUN_TRAINING` false until the preceding smoke tests pass.

# %%
import hashlib
import zipfile
from pathlib import Path

GITHUB_REPOSITORY = "https://github.com/Prathmeshkate09/FlytBase_Visual_Intelligence_Hackathon.git"
GITHUB_BRANCH = "level1-v4"

# Select the corrected development clip by both an unambiguous filename and
# its registered digest. The development and held-out clips both used the
# generic name ``source_video_89f.mp4`` historically.
VIDEO_FILENAME_HINT = "source_video_89f_dev_c8ead5.mp4"
VIDEO_SHA256 = "c8ead5bc7f3fd82dfd3dfe345061996f8822e9a8bea2048b16b58d7b7edbeda1"

# Compare every detector on the same corrected development clip. Each run first
# executes a one-frame CUDA/memory probe and then the full 89-frame segment.
DETECTOR_EXPERIMENTS = {
    "yolov9e_full_1920": {"image_size": 1920, "use_sahi": False},
    "yolov9e_full_2560": {"image_size": 2560, "use_sahi": False},
    "yolov9e_sahi_1280": {"image_size": 1920, "use_sahi": True},
}
MODEL_REPOSITORY = "dronefreak/visdrone-yolov9e"
MODEL_FILENAME = "best.pt"
MODEL_REVISION = "4593a8ea82676f41c46a7cf3e89e39984ac7a2af"

SMOKE_SECONDS = 3
FULL_RUN_SECONDS = 15
RUN_DETECTOR_COMPARISON = True
RUN_TRACKER_SWEEP = True
RUN_FULL_INFERENCE = False
EXTRACT_ANNOTATION_FRAMES = False

# Training is deliberately disabled until corrected labels are attached.
RUN_VISDRONE_PRETRAINING = False
RUN_SCENE_FINETUNING = False

KAGGLE_ROOT = Path("/kaggle")
INPUT_ROOT = KAGGLE_ROOT / "input"
WORK_ROOT = KAGGLE_ROOT / "working" / "flytbase_v4"
REPO_DIR = WORK_ROOT / "repo"
DATA_DIR = WORK_ROOT / "data"
RUNS_DIR = WORK_ROOT / "runs"
CONFIG_DIR = WORK_ROOT / "configs"
TRAINING_DIR = WORK_ROOT / "training"
MODELS_DIR = WORK_ROOT / "models"

if not KAGGLE_ROOT.exists():
    raise RuntimeError("This notebook must run inside Kaggle.")

for directory in (WORK_ROOT, DATA_DIR, RUNS_DIR, CONFIG_DIR, TRAINING_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print("Workspace:", WORK_ROOT)

# %% [markdown]
# ## 2. Verify that Kaggle actually assigned a CUDA GPU

# %%
import subprocess
import sys

subprocess.run(["nvidia-smi"], check=True)

# Kaggle may assign a Pascal-generation Tesla P100 (compute capability 6.0).
# Its default CUDA 12.8 PyTorch image can omit sm_60 kernels even though
# torch.cuda.is_available() returns True. The official CUDA 12.6 wheels retain
# Pascal support, so install them before torch is imported by this kernel.
# Install torchvision without dependencies: replacing Pillow/NumPy inside the
# live Papermill process can leave already-imported modules in an invalid state.
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "--force-reinstall",
        "torch==2.9.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
        "torchvision==0.24.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
    ],
    check=True,
)

import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Stop and set Kaggle Accelerator to GPU.")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print("PyTorch CUDA architectures:", torch.cuda.get_arch_list())

device_capability = torch.cuda.get_device_capability(0)
required_arch = f"sm_{device_capability[0]}{device_capability[1]}"
if required_arch not in torch.cuda.get_arch_list():
    raise RuntimeError(
        f"Installed PyTorch does not contain kernels for {required_arch}: "
        f"{torch.cuda.get_arch_list()}"
    )

# `is_available()` alone is insufficient: execute a real CUDA kernel.
cuda_probe = (torch.ones(1024, device="cuda") * 2).sum().item()
if cuda_probe != 2048.0:
    raise RuntimeError(f"CUDA computation probe returned {cuda_probe}, expected 2048.0")
print("CUDA computation probe: passed")

# %% [markdown]
# ## 3. Find the attached videos and select the requested one

# %%
videos = sorted(INPUT_ROOT.rglob("*.mp4"))
if not videos:
    raise FileNotFoundError(
        "No MP4 was found under /kaggle/input. Attach the private FlytBase dataset using Add Input."
    )

for index, path in enumerate(videos):
    print(f"[{index}] {path} ({path.stat().st_size / 2**20:.1f} MiB)")

exact_matches = [path for path in videos if path.name == VIDEO_FILENAME_HINT]
partial_matches = [path for path in videos if VIDEO_FILENAME_HINT.lower() in path.name.lower()]
matches = exact_matches or partial_matches
if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly one match for {VIDEO_FILENAME_HINT!r}; found {len(matches)}. "
        "Change VIDEO_FILENAME_HINT in Cell 1."
    )

VIDEO_SOURCE = matches[0]
video_digest = hashlib.sha256()
with VIDEO_SOURCE.open("rb") as video_stream:
    for chunk in iter(lambda: video_stream.read(1024 * 1024), b""):
        video_digest.update(chunk)
actual_video_sha256 = video_digest.hexdigest()
if actual_video_sha256 != VIDEO_SHA256:
    raise RuntimeError(
        f"Video SHA256 mismatch for {VIDEO_SOURCE}: "
        f"expected {VIDEO_SHA256}, found {actual_video_sha256}"
    )
VIDEO_PATH = DATA_DIR / VIDEO_SOURCE.name
print("Selected:", VIDEO_SOURCE)
print("Video SHA256:", actual_video_sha256)

# %% [markdown]
# ## 4. Pull the current implementation directly from GitHub
#
# Re-running this cell updates an existing checkout with a fast-forward pull.

# %%
if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", GITHUB_BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", GITHUB_BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", GITHUB_BRANCH], check=True)
else:
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            GITHUB_BRANCH,
            "--single-branch",
            GITHUB_REPOSITORY,
            str(REPO_DIR),
        ],
        check=True,
    )

commit = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
).strip()
print("Checked out commit:", commit)

# %% [markdown]
# ## 5. Install the pinned project dependencies

# %%
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements-eval.txt")],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.27,<2"],
    check=True,
)

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import hashlib
import pandas as pd

MODEL_NAME_OR_PATH = Path(
    hf_hub_download(
        repo_id=MODEL_REPOSITORY,
        filename=MODEL_FILENAME,
        revision=MODEL_REVISION,
        local_dir=MODELS_DIR,
    )
)
model_digest = hashlib.sha256(MODEL_NAME_OR_PATH.read_bytes()).hexdigest()
model_probe = YOLO(str(MODEL_NAME_OR_PATH))
print("Aerial model:", MODEL_NAME_OR_PATH)
print("Aerial model SHA256:", model_digest)
print("Aerial model classes:", model_probe.names)
del model_probe

print("OpenCV:", cv2.__version__)
print("Dependencies installed.")

# %% [markdown]
# ## 6. Copy the read-only Kaggle input video into the writable workspace

# %%
import shutil

if not VIDEO_PATH.exists() or VIDEO_PATH.stat().st_size != VIDEO_SOURCE.stat().st_size:
    shutil.copy2(VIDEO_SOURCE, VIDEO_PATH)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"OpenCV could not open {VIDEO_PATH}")

fps = float(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(
    {
        "video": str(VIDEO_PATH),
        "fps": fps,
        "frames": frame_count,
        "duration_s": frame_count / fps,
        "resolution": f"{width}x{height}",
    }
)

# %% [markdown]
# ## 7. Run the repository tests before spending GPU time

# %%
subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--deselect",
        "tests/test_project_context.py::test_registered_local_dataset_assets_exist",
    ],
    cwd=REPO_DIR,
    check=True,
)

# %% [markdown]
# ## 8. Build hashable detector experiment configurations

# %%
import json

base_detector_config = {
    "model_path": str(MODEL_NAME_OR_PATH),
    "device": "0",
    "confidence": 0.05,
    "iou": 0.50,
    "class_ids": [0, 1, 2, 3, 4, 5, 8, 9],
    "slice_height": 1280,
    "slice_width": 1280,
    "overlap_height_ratio": 0.20,
    "overlap_width_ratio": 0.20,
    "batch_size": 1,
    "postprocess_type": "GREEDYNMM",
    "postprocess_match_metric": "IOS",
    "postprocess_match_threshold": 0.50,
    "postprocess_class_agnostic": False,
    "class_name_map": {
        "pedestrian": "pedestrian", "people": "pedestrian",
        "bicycle": "bicycle", "car": "car", "van": "lgv",
        "truck": "truck", "bus": "bus", "motor": "motorcycle",
    },
    "class_confidence_thresholds": {
        "pedestrian": 0.15, "bicycle": 0.15, "car": 0.12,
        "lgv": 0.12, "truck": 0.12, "bus": 0.12, "motorcycle": 0.05,
    },
}

DETECTION_CONFIGS = {}
for experiment_name, overrides in DETECTOR_EXPERIMENTS.items():
    detector_config = {
        **base_detector_config,
        **overrides,
        "perform_standard_prediction": bool(overrides["use_sahi"]),
    }
    config_path = CONFIG_DIR / f"detection_{experiment_name}.json"
    config_path.write_text(json.dumps(detector_config, indent=2), encoding="utf-8")
    DETECTION_CONFIGS[experiment_name] = config_path
print({name: str(path) for name, path in DETECTION_CONFIGS.items()})

# %% [markdown]
# ## 9. One-frame probes and controlled 89-frame comparison

# %%
mot_archives = list(INPUT_ROOT.rglob("FlytBase_L1_dev89_corrected_MOT.zip"))
mot_directories = sorted({
    path.parent.parent
    for path in INPUT_ROOT.rglob("gt/gt.txt")
    if path.parent.parent.name == "FlytBase_L1_dev89_corrected_MOT"
})
if len(mot_archives) + len(mot_directories) != 1:
    raise RuntimeError(
        "Expected exactly one corrected development MOT input as an archive "
        f"or unpacked directory; archives={mot_archives}, directories={mot_directories}"
    )
if mot_archives:
    MOT_GROUND_TRUTH = mot_archives[0]
else:
    # Kaggle automatically expands ZIP dataset files. Recreate the archive
    # shape consumed by the evaluator while preserving the internal paths.
    mot_directory = mot_directories[0]
    MOT_GROUND_TRUTH = DATA_DIR / "FlytBase_L1_dev89_corrected_MOT.zip"
    mot_files = sorted(path for path in mot_directory.rglob("*") if path.is_file())
    if not mot_files or not (mot_directory / "gt" / "gt.txt").is_file():
        raise RuntimeError(f"Incomplete unpacked MOT input: {mot_directory}")
    with zipfile.ZipFile(MOT_GROUND_TRUTH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in mot_files:
            archive.write(path, path.relative_to(mot_directory).as_posix())
print("Development MOT ground truth:", MOT_GROUND_TRUTH)

comparison_reports = {}
comparison_outputs = {}
if RUN_DETECTOR_COMPARISON:
    for experiment_name, detection_config_path in DETECTION_CONFIGS.items():
        probe_output = RUNS_DIR / f"probe_{experiment_name}"
        run_output = RUNS_DIR / f"dev89_{experiment_name}"
        common_command = [
            sys.executable, str(REPO_DIR / "run_v4.py"),
            "--input", str(VIDEO_PATH),
            "--detection-config", str(detection_config_path),
            "--tracker-config", str(REPO_DIR / "config" / "botsort_drone_v4.yaml"),
            "--road-user-roi", str(REPO_DIR / "config" / "road_user_roi_intersection_v4.json"),
            "--max-prediction-frames", "2",
            "--confirmation-observations", "5",
        ]
        subprocess.run(
            [*common_command, "--output", str(probe_output), "--max-seconds", str(1.0 / fps + 0.001)],
            cwd=REPO_DIR,
            check=True,
        )
        subprocess.run(
            [*common_command, "--output", str(run_output), "--max-seconds", str(SMOKE_SECONDS)],
            cwd=REPO_DIR,
            check=True,
        )
        raw_report_path = run_output / "quality_report_candidates.json"
        subprocess.run(
            [
                sys.executable, str(REPO_DIR / "evaluate_detection_cache.py"),
                "--mot-ground-truth", str(MOT_GROUND_TRUTH),
                "--detections", str(run_output / "candidate_detections.csv"),
                "--output", str(raw_report_path),
            ],
            cwd=REPO_DIR,
            check=True,
        )
        gt_report_path = run_output / "quality_report_ground_truth.json"
        diagnostics_path = run_output / "error_diagnostics.json"
        subprocess.run(
            [
                sys.executable, str(REPO_DIR / "evaluate_v4.py"),
                "--mot-ground-truth", str(MOT_GROUND_TRUTH),
                "--tracks", str(run_output / "tracks.csv"),
                "--track-summary", str(run_output / "track_summary.csv"),
                "--run-manifest", str(run_output / "run_manifest.json"),
                "--output", str(gt_report_path),
                "--diagnostics-output", str(diagnostics_path),
            ],
            cwd=REPO_DIR,
            check=True,
        )
        comparison_reports[experiment_name] = json.loads(gt_report_path.read_text(encoding="utf-8"))
        comparison_outputs[experiment_name] = run_output

    def ranking(item):
        metrics = item[1]["ground_truth_metrics"]
        return (metrics["hota"], metrics["idf1"], -metrics["id_switches"], -metrics["fragmentations"])

    WINNER_NAME, WINNER_REPORT = max(comparison_reports.items(), key=ranking)
    SMOKE_OUTPUT = comparison_outputs[WINNER_NAME]
    DETECTION_CONFIG = DETECTION_CONFIGS[WINNER_NAME]
    print("Development comparison winner:", WINNER_NAME)
    print(json.dumps(WINNER_REPORT, indent=2))
else:
    raise RuntimeError("RUN_DETECTOR_COMPARISON must remain enabled for an auditable optimization run")

WINNER_TRACKER_CONFIG = REPO_DIR / "config" / "botsort_drone_v4.yaml"
WINNER_ROI = REPO_DIR / "config" / "road_user_roi_intersection_v4.json"
WINNER_CONFIRMATION = 5
WINNER_PREDICTION_FRAMES = 2
WINNER_STITCHING = True
if RUN_TRACKER_SWEEP:
    tuning_output = WORK_ROOT / "tuning" / WINNER_NAME
    subprocess.run(
        [
            sys.executable, str(REPO_DIR / "tune_v4.py"),
            "--input", str(VIDEO_PATH),
            "--expected-video-sha256", VIDEO_SHA256,
            "--candidate-detections", str(SMOKE_OUTPUT / "candidate_detections.csv"),
            "--mot-ground-truth", str(MOT_GROUND_TRUTH),
            "--detection-config", str(DETECTION_CONFIG),
            "--tracker-config", str(REPO_DIR / "config" / "botsort_drone_v4.yaml"),
            "--road-user-roi", str(REPO_DIR / "config" / "road_user_roi_intersection_v4.json"),
            "--output", str(tuning_output),
            "--max-seconds", str(SMOKE_SECONDS),
        ],
        cwd=REPO_DIR,
        check=True,
    )
    tuning_summary = json.loads(
        (tuning_output / "tuning_summary.json").read_text(encoding="utf-8")
    )
    WINNER_NAME = f"{WINNER_NAME}_{tuning_summary['winner']}"
    WINNER_REPORT = tuning_summary["winner_report"]
    SMOKE_OUTPUT = tuning_output / "runs" / tuning_summary["winner"]
    DETECTION_CONFIG = Path(tuning_summary["winner_spec"]["detection_config"])
    WINNER_TRACKER_CONFIG = Path(tuning_summary["winner_spec"]["tracker_config"])
    WINNER_ROI = (
        Path(tuning_summary["winner_spec"]["roi"])
        if tuning_summary["winner_spec"]["roi"]
        else None
    )
    WINNER_CONFIRMATION = int(tuning_summary["winner_spec"]["confirmation_observations"])
    WINNER_PREDICTION_FRAMES = int(tuning_summary["winner_spec"]["max_prediction_frames"])
    WINNER_STITCHING = bool(tuning_summary["winner_spec"]["enable_offline_stitching"])
    print("Tuned development winner:", WINNER_NAME)
    print(json.dumps(WINNER_REPORT, indent=2))

# %% [markdown]
# ## 10. Render the best development candidate
#
# Predicted boxes are visually different from observed detections. A visually
# clean video is not accepted as proof of tracking quality; it is only a review
# artifact.

# %%
from IPython.display import Video, display

SMOKE_VIDEO = SMOKE_OUTPUT / "tracking_verification.mp4"
SMOKE_RENDER_REPORT = SMOKE_OUTPUT / "tracking_verification_report.json"
subprocess.run(
    [
        sys.executable,
        str(REPO_DIR / "render_v4.py"),
        "--video",
        str(VIDEO_PATH),
        "--tracks",
        str(SMOKE_OUTPUT / "tracks.csv"),
        "--track-summary",
        str(SMOKE_OUTPUT / "track_summary.csv"),
        "--output",
        str(SMOKE_VIDEO),
        "--report",
        str(SMOKE_RENDER_REPORT),
        "--max-width",
        "1280",
        "--max-labels-per-frame",
        "40",
        "--hide-occluded-labels",
    ],
    cwd=REPO_DIR,
    check=True,
)
display(Video(str(SMOKE_VIDEO), embed=True, width=1000))
print(SMOKE_RENDER_REPORT.read_text(encoding="utf-8"))

# %% [markdown]
# ## 11. Optional 15-second inference with the selected configuration
#
# Set `RUN_FULL_INFERENCE = True` in Cell 1 only after the smoke video and
# counts are plausible. This remains an experiment until corrected MOT ground
# truth is evaluated.

# %%
FULL_OUTPUT = RUNS_DIR / f"level1_15s_{WINNER_NAME}"

if RUN_FULL_INFERENCE:
    full_command = [
        sys.executable, str(REPO_DIR / "run_v4.py"),
        "--input", str(VIDEO_PATH), "--output", str(FULL_OUTPUT),
        "--detection-config", str(DETECTION_CONFIG),
        "--tracker-config", str(WINNER_TRACKER_CONFIG),
        "--max-prediction-frames", str(WINNER_PREDICTION_FRAMES),
        "--confirmation-observations", str(WINNER_CONFIRMATION),
        "--max-seconds", str(FULL_RUN_SECONDS),
    ]
    if WINNER_ROI is not None:
        full_command.extend(["--road-user-roi", str(WINNER_ROI)])
    if not WINNER_STITCHING:
        full_command.append("--disable-offline-stitching")
    subprocess.run(full_command, cwd=REPO_DIR, check=True)
    print((FULL_OUTPUT / "quality_report.json").read_text(encoding="utf-8"))
else:
    print("Skipped. Change RUN_FULL_INFERENCE to True after the smoke test passes.")

# %% [markdown]
# ## 12. Extract the 89-frame development clip for CVAT correction
#
# This creates images to annotate; it does not create trustworthy labels.

# %%
ANNOTATION_FRAMES = 89
annotation_images = WORK_ROOT / "annotation_seed" / "images"
if EXTRACT_ANNOTATION_FRAMES:
    annotation_images.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    written = 0
    while written < ANNOTATION_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        output_path = annotation_images / f"frame_{written:06d}.jpg"
        if not output_path.exists():
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written += 1
    cap.release()

    archive = shutil.make_archive(
        str(WORK_ROOT / "FlytBase_89f_images_for_CVAT"),
        "zip",
        root_dir=annotation_images.parent,
        base_dir=annotation_images.name,
    )
    print("Extracted frames:", written)
    print("CVAT image archive:", archive)
else:
    print("Skipped: the 89-frame CVAT archive already exists in the private input workflow.")

# %% [markdown]
# ## 13. Optional aerial detector pretraining on VisDrone
#
# This downloads the official VisDrone detection subset and can take hours.
# Run it as a separate saved Kaggle version. A scene-specific fine-tune should
# start from the resulting `best.pt` rather than a generic COCO checkpoint.

# %%
VISDRONE_PROJECT = TRAINING_DIR / "visdrone"
VISDRONE_BEST = VISDRONE_PROJECT / "aerial_pretrain" / "weights" / "best.pt"

if RUN_VISDRONE_PRETRAINING:
    model = YOLO("yolo26s.pt")
    model.train(
        data="VisDrone.yaml",
        epochs=60,
        imgsz=1280,
        batch=4,
        device=0,
        workers=2,
        cache=False,
        amp=True,
        project=str(VISDRONE_PROJECT),
        name="aerial_pretrain",
        exist_ok=True,
    )
    print("VisDrone weights:", VISDRONE_BEST)
else:
    print("Skipped. Enable only in a dedicated training run.")

# %% [markdown]
# ## 14. Optional scene-specific fine-tuning from corrected development labels
#
# The corrected COCO archive is converted from the exact hash-verified video.
# Frames 0-70 select the epoch on frames 71-88; the selected epoch count is
# then refit on all 89 development frames. Held-out media is never read here.

# %%
if RUN_SCENE_FINETUNING:
    coco_archives = list(INPUT_ROOT.rglob("FlytBase_L1_dev89_corrected_COCO.zip"))
    coco_jsons = [
        path for path in INPUT_ROOT.rglob("instances_default.json")
        if path.parent.parent.name == "FlytBase_L1_dev89_corrected_COCO"
    ]
    if len(coco_archives) + len(coco_jsons) != 1:
        raise RuntimeError(
            "Expected exactly one corrected COCO input as an archive or unpacked JSON; "
            f"archives={coco_archives}, jsons={coco_jsons}"
        )
    coco_annotations = (coco_archives or coco_jsons)[0]
    split_dataset = TRAINING_DIR / "dev89_split"
    subprocess.run(
        [
            sys.executable, str(REPO_DIR / "prepare_yolo_dataset.py"),
            "--video", str(VIDEO_PATH),
            "--coco-annotations", str(coco_annotations),
            "--output", str(split_dataset),
            "--expected-video-sha256", VIDEO_SHA256,
            "--expected-frames", "89",
            "--validation-start-frame", "71",
        ],
        cwd=REPO_DIR,
        check=True,
    )
    selected_run = None
    for batch_size in (4, 2, 1):
        try:
            model = YOLO(str(MODEL_NAME_OR_PATH))
            selected_run = model.train(
                data=str(split_dataset / "data.yaml"), epochs=80, patience=15,
                imgsz=1280, batch=batch_size, device=0, workers=2,
                cache=False, amp=True, seed=42, deterministic=True,
                project=str(TRAINING_DIR / "scene"),
                name=f"intersection_select_b{batch_size}", exist_ok=True,
            )
            break
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or batch_size == 1:
                raise
            torch.cuda.empty_cache()
            print(f"CUDA OOM at batch {batch_size}; retrying smaller batch")
    if selected_run is None:
        raise RuntimeError("Scene fine-tuning did not produce a result")
    selection_dir = Path(selected_run.save_dir)
    results_table = pd.read_csv(selection_dir / "results.csv")
    map_columns = [name for name in results_table.columns if "mAP50-95" in name]
    if len(map_columns) != 1:
        raise RuntimeError(f"Could not identify validation mAP column: {results_table.columns}")
    best_epoch = int(results_table[map_columns[0]].idxmax()) + 1
    print("Selected epoch count:", best_epoch)

    refit_dataset = TRAINING_DIR / "dev89_all_train"
    subprocess.run(
        [
            sys.executable, str(REPO_DIR / "prepare_yolo_dataset.py"),
            "--video", str(VIDEO_PATH),
            "--coco-annotations", str(coco_annotations),
            "--output", str(refit_dataset),
            "--expected-video-sha256", VIDEO_SHA256,
            "--expected-frames", "89", "--all-train",
        ],
        cwd=REPO_DIR,
        check=True,
    )
    refit_model = YOLO(str(MODEL_NAME_OR_PATH))
    refit_run = refit_model.train(
        data=str(refit_dataset / "data.yaml"), epochs=best_epoch, patience=0,
        imgsz=1280, batch=1, device=0, workers=2, cache=False, amp=True,
        seed=42, deterministic=True, project=str(TRAINING_DIR / "scene"),
        name="intersection_refit_all89", exist_ok=True,
    )
    FROZEN_SCENE_WEIGHTS = Path(refit_run.save_dir) / "weights" / "last.pt"
    if not FROZEN_SCENE_WEIGHTS.exists():
        raise RuntimeError(f"Missing refit weights: {FROZEN_SCENE_WEIGHTS}")
    print("Frozen scene weights:", FROZEN_SCENE_WEIGHTS)
else:
    print("Skipped. Enable after the pretrained detector comparison is reviewed.")

# %% [markdown]
# ## 15. Evaluate and freeze the scene-fine-tuned detector

# %%
if RUN_SCENE_FINETUNING:
    fine_tuned_reports = {}
    fine_tuned_candidate_reports = {}
    for mode_name, use_sahi in (("scene_full_1920", False), ("scene_sahi_1280", True)):
        fine_config = {
            **base_detector_config,
            "model_path": str(FROZEN_SCENE_WEIGHTS),
            # Scene fine-tuning uses the eight canonical Level-1 classes,
            # unlike the ten-class VisDrone source checkpoint.
            "class_ids": list(range(8)),
            "image_size": 1920,
            "use_sahi": use_sahi,
            "perform_standard_prediction": use_sahi,
        }
        fine_config_path = CONFIG_DIR / f"detection_{mode_name}.json"
        fine_config_path.write_text(json.dumps(fine_config, indent=2), encoding="utf-8")
        fine_output = RUNS_DIR / f"dev89_{mode_name}"
        subprocess.run(
            [
                sys.executable, str(REPO_DIR / "run_v4.py"),
                "--input", str(VIDEO_PATH), "--output", str(fine_output),
                "--detection-config", str(fine_config_path),
                "--tracker-config", str(REPO_DIR / "config" / "botsort_drone_v4.yaml"),
                "--road-user-roi", str(REPO_DIR / "config" / "road_user_roi_intersection_v4.json"),
                "--max-prediction-frames", "2", "--confirmation-observations", "5",
                "--max-seconds", str(SMOKE_SECONDS),
            ],
            cwd=REPO_DIR,
            check=True,
        )
        fine_candidate_report_path = fine_output / "quality_report_candidates.json"
        subprocess.run(
            [
                sys.executable, str(REPO_DIR / "evaluate_detection_cache.py"),
                "--mot-ground-truth", str(MOT_GROUND_TRUTH),
                "--detections", str(fine_output / "candidate_detections.csv"),
                "--output", str(fine_candidate_report_path),
            ],
            cwd=REPO_DIR,
            check=True,
        )
        fine_report_path = fine_output / "quality_report_ground_truth.json"
        subprocess.run(
            [
                sys.executable, str(REPO_DIR / "evaluate_v4.py"),
                "--mot-ground-truth", str(MOT_GROUND_TRUTH),
                "--tracks", str(fine_output / "tracks.csv"),
                "--track-summary", str(fine_output / "track_summary.csv"),
                "--run-manifest", str(fine_output / "run_manifest.json"),
                "--output", str(fine_report_path),
                "--diagnostics-output", str(fine_output / "error_diagnostics.json"),
            ],
            cwd=REPO_DIR,
            check=True,
        )
        fine_tuned_candidate_reports[mode_name] = json.loads(
            fine_candidate_report_path.read_text(encoding="utf-8")
        )
        fine_tuned_reports[mode_name] = json.loads(fine_report_path.read_text(encoding="utf-8"))
        comparison_outputs[mode_name] = fine_output
    fine_winner_name, fine_winner_report = max(fine_tuned_reports.items(), key=ranking)
    if ranking((fine_winner_name, fine_winner_report)) > ranking((WINNER_NAME, WINNER_REPORT)):
        WINNER_NAME, WINNER_REPORT = fine_winner_name, fine_winner_report
        SMOKE_OUTPUT = comparison_outputs[WINNER_NAME]
    print("Frozen development winner:", WINNER_NAME)
    print(json.dumps(WINNER_REPORT, indent=2))
    print("Fine-tuned raw detector reports:")
    print(json.dumps(fine_tuned_candidate_reports, indent=2))

# %% [markdown]
# ## 16. Package outputs for Kaggle versioning/download
#
# `/kaggle/working` is preserved when the notebook is saved as a version.

# %%
DOWNLOAD_VIDEO = KAGGLE_ROOT / "working" / "FlytBase_V4_Aerial_3s_Verification.mp4"
DOWNLOAD_ARCHIVE_BASE = KAGGLE_ROOT / "working" / "FlytBase_V4_Aerial_3s_Results"
shutil.copy2(SMOKE_VIDEO, DOWNLOAD_VIDEO)
DOWNLOAD_ARCHIVE = Path(
    shutil.make_archive(
        str(DOWNLOAD_ARCHIVE_BASE),
        "zip",
        root_dir=SMOKE_OUTPUT,
    )
)

print("Direct video download:", DOWNLOAD_VIDEO)
print("Direct result archive:", DOWNLOAD_ARCHIVE)

summary_files = sorted(
    path for path in WORK_ROOT.rglob("*") if path.is_file() and path.stat().st_size < 5 * 2**20
)
print("Small result files:")
for path in summary_files:
    print(path.relative_to(WORK_ROOT))

print("\nSave a Kaggle Notebook version to preserve everything under:")
print(WORK_ROOT)
