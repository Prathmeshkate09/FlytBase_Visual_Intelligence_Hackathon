# FlytBase Traffic Analysis Agent

Colab-ready pipeline for **Level 1 detection/tracking** and **Level 2 object insight** from aerial traffic video.

## Engineering boundary

This project separates what the pixels support from what requires calibration:

- Level 1 produces persistent IDs, raw tracks, cleaned trajectories, object metrics, counts and heatmaps.
- Level 2 reuses those tracks and adds coarse vehicle colour, scene-relative vehicle size, per-frame velocity/acceleration, stopped duration and a clean evidence video.
- Metric speed is emitted only when a validated image-to-ground homography is supplied.
- Make/model and licence-plate text are not claimed from low-resolution top-down crops. Each track receives an explicit plate-resolution assessment instead.

## Level 1 command used for the selected V3 run

```bash
python run.py \
  --input /content/intersection_60s.mp4 \
  --output /content/flytbase_results/level1_botsort_drone_v3 \
  --model yolo26l.pt \
  --tracker config/botsort_drone.yaml \
  --imgsz 1280 \
  --conf 0.05 \
  --classes 2,3,5,7 \
  --device 0 \
  --stride 1 \
  --max-seconds 15
```

## Level 2 quick run: no detection rerun

Use the exact source video and `raw_tracks.csv` from Level 1:

```bash
python run_level2.py \
  --raw-tracks /content/flytbase_results/level1_botsort_drone_v3/raw_tracks.csv \
  --video /content/intersection_60s.mp4 \
  --output /content/flytbase_results/level2_object_insight
```

This immediately produces defensible appearance features and pixel-space motion. The summary will explicitly mark metric kinematics unavailable.

## Preferred metric run: use the supplied DJI SRT telemetry

The dataset includes frame-level GPS, relative altitude, 35 mm-equivalent focal length and gimbal yaw/pitch/roll. For a clip that begins at telemetry frame 0:

```bash
python run_level2.py \
  --raw-tracks /content/flytbase_results/level1_botsort_drone_v3/raw_tracks.csv \
  --video /content/intersection_60s.mp4 \
  --output /content/flytbase_results/level2_object_insight_metric \
  --srt /content/drive/MyDrive/Intersection_1080p.srt \
  --srt-segment-index 0 \
  --srt-max-interpolation-gap-frames 2 \
  --srt-frame-offset 0
```

The intersection SRT contains two concatenated `FrameCnt` sequences. Segment selection is therefore explicit rather than guessed. Segment 0 also has one missing log record at frame 407; the bounded-gap setting permits interpolation only across gaps this small and records every interpolated frame in the projection report. The projector uses a pinhole camera model, translates GPS into local east/north offsets and intersects each bottom-centre image ray with a locally flat ground plane. A 31-frame Savitzky-Golay window is used to suppress detector-box jitter while preserving road-scale motion. Speeds are explicitly labelled **telemetry-derived estimates**, not survey-grade measurements. Reliability flags require valid downward rays, sufficient duration, high track continuity and valid per-frame telemetry.

## Manual homography fallback

1. Export a labelled frame:

```bash
python prepare_calibration.py \
  --video /content/intersection_60s.mp4 \
  --frame 200 \
  --output /content/calibration_frame.png
```

2. Copy `config/calibration.example.json` to `config/intersection_calibration.json`.
3. Replace the example image points with four or more road-plane points from the labelled frame.
4. Replace the world points with the corresponding measured metre coordinates.
5. State the source honestly as `surveyed`, `map-derived`, or `assumption-based`.
6. Run:

```bash
python run_level2.py \
  --raw-tracks /content/flytbase_results/level1_botsort_drone_v3/raw_tracks.csv \
  --video /content/intersection_60s.mp4 \
  --output /content/flytbase_results/level2_object_insight_metric \
  --calibration config/intersection_calibration.json
```

The homography loader rejects degenerate control points and calibrations whose reprojection error exceeds the configured limit. Metric summaries only include tracks with at least 80% reliable ground projections, at least 70% observed frames, and at least two seconds of duration.

## Level 2 outputs

- `track_appearance.csv`: track-level coarse colour and confidence
- `level2_kinematics.csv`: frame-level smoothed position, velocity, acceleration and heading
- `level2_object_insights.csv`: one row per track segment with appearance and kinematic summaries
- `level2_speed_profiles.png`: profiles for the longest tracks
- `level2_evidence.mp4`: clean annotated evidence video
- `metric_projection_report.json`: SRT camera model or homography provenance and validation details
- `level2_summary.json`: submission-ready facts and limitations
- `level2_run_metadata.json`: reproducibility metadata

## Browser-compatible evidence video

OpenCV writes a portable intermediate MP4. Convert it for browser playback in Colab:

```bash
ffmpeg -y \
  -i /content/flytbase_results/level2_object_insight/level2_evidence.mp4 \
  -c:v libx264 -preset veryfast -crf 25 -an \
  /content/FlytBase_Level2_Evidence.mp4
```

## Tests

```bash
python -m pytest -q
```

The tests cover trajectory cleaning, pre-stabilisation class consistency, colour families, relative-size classification, homography validation, SRT parsing, camera projection scale and the strict metric/uncalibrated kinematics boundary.
