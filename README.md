# FlytBase Traffic Analysis Agent

Solo-hackathon pipeline for **Level 1: Detection & Tracking**, designed so the same trajectory store can power Levels 2–5.

## Decision

- **Primary compute:** Google Colab T4. It mounts Google Drive directly and avoids moving hours of video through the local laptop.
- **Backup:** Kaggle T4 if Colab quota or availability fails.
- **Detector:** `yolo26l.pt` at 1280 px. Increase to 1600 for very small vehicles.
- **Tracker:** BoT-SORT with sparse optical-flow camera-motion compensation. This is safer than plain ByteTrack for drone or moving-camera footage.
- **No custom training initially:** there are no annotations and the same-day deadline makes fine-tuning a poor first move.
- **Ground point:** bottom-center of each box. This is the correct point to later project onto the road plane.

## What the first pipeline produces

- Annotated MP4 with persistent IDs and trajectory trails
- Raw detection/track CSV
- Cleaned and gap-interpolated trajectory CSV
- Per-object duration, path, continuity, class consistency, and speed profile in pixels
- Per-second visible-object counts
- Trajectory-density heatmap
- JSON summary with honest no-ground-truth quality proxies
- Streamlit demo dashboard

> Important: speed remains **pixels/second** until a homography or telemetry-based ground calibration is added. Do not present it as km/h yet.

## Colab quick start

1. In Google Drive, add a shortcut for the shared dataset folder to **My Drive**.
2. Upload this project folder or ZIP to `MyDrive/FlytBase/`.
3. In Colab select **Runtime → Change runtime type → T4 GPU**.
4. Run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/drive/MyDrive/FlytBase/flytbase-traffic-agent
!pip install -q -r requirements.txt
```

First run only a 60-second clip so we can tune confidence and image size:

```bash
!python run.py \
  --input "/content/drive/MyDrive/REPLACE_WITH_DATASET_PATH/video.mp4" \
  --output "/content/drive/MyDrive/FlytBase/results/level1_smoke" \
  --model yolo26l.pt \
  --imgsz 1280 \
  --conf 0.10 \
  --max-seconds 60
```

If vehicles are missed:

```bash
!python run.py \
  --input "/content/drive/MyDrive/REPLACE_WITH_DATASET_PATH/video.mp4" \
  --output "/content/drive/MyDrive/FlytBase/results/level1_highres" \
  --model yolo26l.pt \
  --imgsz 1600 \
  --conf 0.05 \
  --max-seconds 60
```

After selecting the best settings, remove `--max-seconds` for the submission clip.

## Dashboard

```bash
!streamlit run demo_app.py --server.port 8501
```

For a same-day submission, the annotated video plus `trajectories.csv`, `summary.json`, and the density heatmap are the core evidence.

## Execution order for today

1. **Unlock Level 1 quickly:** use one representative 45–90 second clip, produce IDs + CSV + evidence video, and submit.
2. **Level 2:** speed profile, stopped duration, trajectory shape, class and turning movement per road user.
3. **Level 3:** class/movement counts, queue evolution, flow/density, heatmap and congestion timeline.
4. **Level 4:** map bottom-centres through four or more road control points into metres; fuse telemetry if the dataset supplies synchronized fields.
5. **Level 5:** pairwise conflict metrics (TTC/PET), failed merges, congestion-origin reasoning, and natural-language search over computed events.

## Required next input

To tune this correctly, provide:

- Screenshot of the Drive folder file listing with names and sizes
- One representative video or a 30–60 second sample
- Any telemetry CSV/JSON associated with that video
- The full Level 1 submission page, including accepted files and scoring fields

