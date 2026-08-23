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
from pathlib import Path

GITHUB_REPOSITORY = "https://github.com/Prathmeshkate09/FlytBase_Visual_Intelligence_Hackathon.git"
GITHUB_BRANCH = "level1-v4"

# The notebook searches every attached Kaggle input recursively.
VIDEO_FILENAME_HINT = "intersection_verify_30_45.mp4"

# Detector experiment: "full_frame" or "sahi".
DETECTOR_MODE = "full_frame"
MODEL_NAME_OR_PATH = "yolo26s.pt"

SMOKE_SECONDS = 1
FULL_RUN_SECONDS = 15
RUN_FULL_INFERENCE = False

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

if not KAGGLE_ROOT.exists():
    raise RuntimeError("This notebook must run inside Kaggle.")

for directory in (WORK_ROOT, DATA_DIR, RUNS_DIR, CONFIG_DIR, TRAINING_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print("Workspace:", WORK_ROOT)

# %% [markdown]
# ## 2. Verify that Kaggle actually assigned a CUDA GPU

# %%
import subprocess
import sys

subprocess.run(["nvidia-smi"], check=True)

import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Stop and set Kaggle Accelerator to GPU.")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))

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
VIDEO_PATH = DATA_DIR / VIDEO_SOURCE.name
print("Selected:", VIDEO_SOURCE)

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

from ultralytics import YOLO
import cv2
import pandas as pd

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
    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
    cwd=REPO_DIR,
    check=True,
)

# %% [markdown]
# ## 8. Build an explicit detector experiment configuration
#
# The current generic detector is only a baseline. `full_frame` disables SAHI;
# `sahi` enables sliced inference and class-agnostic merging so cross-class tile
# duplicates can be consolidated before tracking.

# %%
import json

detector_config = {
    "model_path": MODEL_NAME_OR_PATH,
    "device": "0",
    "confidence": 0.10,
    "iou": 0.50,
    "class_ids": [0, 1, 2, 3, 5, 7],
    "image_size": 1280,
    "use_sahi": DETECTOR_MODE == "sahi",
    "slice_height": 960,
    "slice_width": 960,
    "overlap_height_ratio": 0.20,
    "overlap_width_ratio": 0.20,
    "batch_size": 1,
    "perform_standard_prediction": DETECTOR_MODE == "sahi",
    "postprocess_type": "GREEDYNMM",
    "postprocess_match_metric": "IOS",
    "postprocess_match_threshold": 0.50,
    "postprocess_class_agnostic": True,
}

DETECTION_CONFIG = CONFIG_DIR / f"detection_{DETECTOR_MODE}.json"
DETECTION_CONFIG.write_text(json.dumps(detector_config, indent=2), encoding="utf-8")
print(DETECTION_CONFIG.read_text(encoding="utf-8"))

# %% [markdown]
# ## 9. One-second integration smoke test

# This validates video decoding, detector CUDA inference, tracker integration
# and file output before a long run.

# %%
SMOKE_OUTPUT = RUNS_DIR / f"smoke_{DETECTOR_MODE}"

command = [
    sys.executable,
    str(REPO_DIR / "run_v4.py"),
    "--input",
    str(VIDEO_PATH),
    "--output",
    str(SMOKE_OUTPUT),
    "--detection-config",
    str(DETECTION_CONFIG),
    "--tracker-config",
    str(REPO_DIR / "config" / "botsort_drone_v4.yaml"),
    "--max-seconds",
    str(SMOKE_SECONDS),
]
subprocess.run(command, cwd=REPO_DIR, check=True)

report = json.loads((SMOKE_OUTPUT / "quality_report.json").read_text(encoding="utf-8"))
print(json.dumps(report, indent=2))

# %% [markdown]
# ## 10. Render and inspect the smoke-test evidence
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
# ## 11. Optional 15-second inference
#
# Set `RUN_FULL_INFERENCE = True` in Cell 1 only after the smoke video and
# counts are plausible. This remains an experiment until corrected MOT ground
# truth is evaluated.

# %%
FULL_OUTPUT = RUNS_DIR / f"level1_15s_{DETECTOR_MODE}"

if RUN_FULL_INFERENCE:
    full_command = command.copy()
    full_command[full_command.index(str(SMOKE_OUTPUT))] = str(FULL_OUTPUT)
    full_command[full_command.index(str(SMOKE_SECONDS))] = str(FULL_RUN_SECONDS)
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
# ## 14. Optional scene-specific fine-tuning
#
# Attach a private labelled dataset containing this structure:
#
# ```text
# images/train/*.jpg
# images/val/*.jpg
# labels/train/*.txt
# labels/val/*.txt
# data.yaml
# ```
#
# The cell refuses to train if no `data.yaml` exists.

# %%
scene_yamls = [
    path
    for path in INPUT_ROOT.rglob("*.yaml")
    if path.name.lower() in {"data.yaml", "dataset.yaml"}
]
print("Candidate scene datasets:", [str(path) for path in scene_yamls])

if RUN_SCENE_FINETUNING:
    if len(scene_yamls) != 1:
        raise RuntimeError(
            "Attach exactly one corrected scene dataset, or edit this cell to select its data.yaml."
        )
    starting_weights = VISDRONE_BEST if VISDRONE_BEST.exists() else Path("yolo26s.pt")
    model = YOLO(str(starting_weights))
    model.train(
        data=str(scene_yamls[0]),
        epochs=80,
        imgsz=1280,
        batch=4,
        device=0,
        workers=2,
        cache=False,
        amp=True,
        project=str(TRAINING_DIR / "scene"),
        name="intersection_finetune",
        exist_ok=True,
    )
else:
    print("Skipped. Correct the CVAT labels before enabling scene fine-tuning.")

# %% [markdown]
# ## 15. Package outputs for Kaggle versioning/download

# `/kaggle/working` is preserved when the notebook is saved as a version.

# %%
summary_files = sorted(
    path for path in WORK_ROOT.rglob("*") if path.is_file() and path.stat().st_size < 5 * 2**20
)
print("Small result files:")
for path in summary_files:
    print(path.relative_to(WORK_ROOT))

print("\nSave a Kaggle Notebook version to preserve everything under:")
print(WORK_ROOT)

