# FlytBase V4 Kaggle GPU Guide

## Recommended architecture

- GitHub branch `level1-v4` is the source of truth for code.
- A private Kaggle Dataset stores MP4 files, corrected annotations and optional weights.
- A private Kaggle Notebook pulls the branch and runs GPU experiments.
- Local VS Code/Codex edits and tests code, then pushes the branch.

Do not copy changing source files into Kaggle manually. Pull the current Git
commit from GitHub so every result has a traceable revision.

## One-time Kaggle API setup on Windows

Install the current official CLI:

```powershell
py -3.13 -m pip install --user --upgrade kaggle
kaggle --version
```

Authenticate using OAuth:

```powershell
kaggle auth login
```

Complete the browser prompt. Never add the Kaggle credential, access token,
`kaggle.json` or `.kaggle` directory to Git.

Official references:

- <https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md>
- <https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md>
- <https://www.kaggle.com/docs/notebooks>

## Upload the videos as a private Kaggle Dataset

The browser method is simplest:

1. Open <https://www.kaggle.com/datasets>.
2. Select **New Dataset**.
3. Upload `intersection_verify_30_45.mp4` and `source_video_89f.mp4`.
4. Name it `FlytBase V4 Inputs`.
5. Keep it **Private**.
6. Create the dataset.

The API method is reproducible:

```powershell
New-Item -ItemType Directory -Path C:\path\to\flytbase-kaggle-input
Copy-Item -LiteralPath C:\path\to\intersection_verify_30_45.mp4 -Destination C:\path\to\flytbase-kaggle-input
Copy-Item -LiteralPath C:\path\to\source_video_89f.mp4 -Destination C:\path\to\flytbase-kaggle-input
kaggle datasets init -p C:\path\to\flytbase-kaggle-input
```

Edit `dataset-metadata.json` and replace the placeholders:

```json
{
  "title": "FlytBase V4 Inputs",
  "id": "YOUR_KAGGLE_USERNAME/flytbase-v4-inputs",
  "licenses": [{"name": "other"}]
}
```

Create the private dataset. Private is the default; do not pass `--public`:

```powershell
kaggle datasets create -p C:\path\to\flytbase-kaggle-input
```

## Create the Kaggle Notebook

1. Open <https://www.kaggle.com/code> and select **New Notebook**.
2. Use **File → Import Notebook**.
3. Upload `FlytBase_V4_Kaggle.ipynb`.
4. In **Settings**, set **Accelerator → GPU**.
5. Turn **Internet On** so the notebook can pull GitHub and install packages.
6. Select **Add Input** and attach the private `FlytBase V4 Inputs` dataset.
7. Run the notebook one cell at a time.

Kaggle inputs are read-only under `/kaggle/input`. The notebook copies the
selected video into `/kaggle/working/flytbase_v4`, which is writable and can be
saved as notebook output.

## Notebook execution order

1. Edit the user settings cell.
2. Verify the GPU name, supported CUDA architectures and real CUDA tensor probe.
3. Confirm the correct MP4 was discovered.
4. Pull `level1-v4` from GitHub.
5. Install dependencies.
6. Run all repository tests.
7. Run the one-second smoke inference.
8. Inspect its counts and evidence video.
9. Enable the 15-second run only if the smoke test is valid.
10. Extract the 89 development frames for CVAT correction.
11. Train only after corrected labels and `data.yaml` are attached.

## Video is not training data by itself

An MP4 supplies images but no correct object boxes, classes or persistent IDs.
Detector fine-tuning requires:

```text
images/train/*.jpg
images/val/*.jpg
labels/train/*.txt
labels/val/*.txt
data.yaml
```

Tracking evaluation also requires corrected frame-level identities, exported
from CVAT as MOT format. Seed predictions may reduce annotation work, but they
must be manually corrected before being called ground truth.

## Optimization plan on Kaggle

Kaggle is the execution environment; the engineering plan is unchanged:

1. Fix stale prediction rendering while retaining internal recovery state.
2. Correct the 89-frame ground truth.
3. Fine-tune an aerial detector on VisDrone and the intersection labels.
4. Cache detector outputs.
5. Compare full-frame, SAHI-only and combined inference.
6. Compare BoT-SORT, OC-SORT, BoostTrack and StrongSORT++ on identical detections.
7. Select using recall, false positives, HOTA, IDF1 and identity switches.
8. Freeze the winner and test 15 seconds, then unseen 30-second footage.

The notebook initially runs the existing V4 pipeline. Tracker adapters and
evaluation improvements will arrive through new commits on `level1-v4`; rerun
the Git pull cell to receive them.

### Kaggle P100 compatibility

Kaggle may assign a Tesla P100 with CUDA compute capability `sm_60`. A newer
preinstalled PyTorch CUDA image can report CUDA as available while omitting the
actual `sm_60` kernels. The notebook therefore installs the official PyTorch
2.9.1 CUDA 12.6 build before importing torch and performs a real tensor
calculation on the GPU. Do not remove this runtime check merely because
`torch.cuda.is_available()` returns true.

## Saving and downloading results

Use **Save Version** in Kaggle after a successful run. Kaggle preserves files
under `/kaggle/working` as notebook outputs. From Windows, notebook outputs can
also be downloaded with:

```powershell
kaggle kernels output YOUR_KAGGLE_USERNAME/YOUR_NOTEBOOK-SLUG -p C:\path\to\download
```

Do not commit videos, datasets, trained weights or API credentials to GitHub.
