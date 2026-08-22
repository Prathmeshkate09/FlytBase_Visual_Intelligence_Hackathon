# FlytBase Traffic Analysis Agent

Colab-ready pipeline for **Level 1 detection/tracking** and **Level 2 object insight** from aerial traffic video.

## V4 revision status

The submitted Level 1 and Level 2 source states are preserved in Git tags
`submission-level1-v1` and `submission-level2-v1`. Active development happens
on the `level1-v4` branch.

V4 follows a strict dependency boundary: Level 2 must not consume a Level 1
trajectory set until detection and identity quality have been evaluated. The
first V4 component is a reproducible proxy audit that detects duplicate IDs,
probable in-frame ID handoffs, raw class instability, internal track starts or
ends, and evidence-video filtering. These proxy checks expose regressions but
do not replace labelled precision/recall, IDF1 and HOTA evaluation.

Run the audit against an existing Level 1 result:

```bash
python audit_tracks.py \
  --raw-tracks /content/flytbase_results/level1_botsort_drone_v3/raw_tracks.csv \
  --metadata /content/flytbase_results/level1_botsort_drone_v3/run_metadata.json \
  --output /content/flytbase_results/level1_botsort_drone_v3/audit
```

The command writes `tracking_audit.json` and `tracking_failures.csv`. A result
with `verification_status: not_ground_truth_verified` must never be presented
as formally verified tracking accuracy.

## V4 Colab smoke test (new pipeline)

V4 separates SAHI detection, group-aware duplicate merging, BoT-SORT
association and lifecycle accounting. It includes pedestrians and cyclists as
road users; car, bus and truck detector votes share one association group so a
temporary fine-grained class change cannot create a second tracker pool.

Start with 30 frames before paying the cost of a full 4K sliced-inference run:

```bash
python run_v4.py \
  --input /content/intersection_60s.mp4 \
  --output /content/flytbase_results/level1_v4_smoke \
  --detection-config config/detection_v4.json \
  --tracker-config config/botsort_drone_v4.yaml \
  --max-seconds 1
```

The ROI is optional until a verified polygon has been drawn. Do not use the
example full-frame ROI as evidence that roof/building false positives were
removed. A V4 run emits:

- `detections.csv`: consolidated detections before tracking.
- `rejected_detections.csv`: rejected boxes and explicit reasons.
- `tracks.csv`: both real observations and `observed=False` Kalman predictions.
- `raw_tracks.csv`: observed rows only, retaining Level-2 compatibility.
- `track_events.csv`: created, confirmed, lost, recovered, exited and expired events.
- `track_summary.csv`: confidence-weighted final class per identity.
- `quality_report.json`: remains `not_evaluated` until labelled evaluation passes.
- `run_manifest.json`: input/trajectory hashes, package versions and exact configuration.

SAHI at 4K is compute-heavy. After the smoke test passes, benchmark a short
clip before changing `batch_size`; reduce it from 4 if a Colab T4 runs out of
GPU memory.

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
