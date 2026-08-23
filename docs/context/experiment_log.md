# Experiment log

This is append-only. Results without preserved artifacts are labelled accordingly.

## E001 — Submitted V3 BoT-SORT drone configuration

- Date: 2026-08-22
- Code state: legacy submitted state, preserved by Git tag according to `README.md`
- Detector: generic `yolo26l.pt`, 1280 pixels, confidence 0.05, vehicle COCO classes only
- Tracker: drone-tuned BoT-SORT
- Clip: 15 seconds
- Reported proxy results: 102 unique tracks, median duration 3.804 seconds, 67.72% at least two seconds, class consistency 1.0 after stabilisation
- Decision: useful baseline, rejected as verified Level 1 after host and visual review found missed users, false tracks, clutter, and identity concerns
- Limitation: the proxy metrics do not measure missed objects or physical identity accuracy

## E002 — Generic detector with SAHI at 960

- Run ID: `4c24ff8d-c5bf-4bfa-959a-bc389783c25d`
- Evidence: `../local_runs/level1_v4_15s_raw_960_r3/quality_report.json`
- Frames: 449
- Results: 126,807 merged detections; 59,912 observed rows; 51,893 prediction rows; 468 native tracks; 452 final tracks; 141 internal expirations
- Decision: reject as the default candidate; preserve SAHI for a labelled detector experiment
- Limitation: no ground-truth metrics

## E003 — Full-frame aerial detector, 89-frame BoT-SORT replay

- Run ID: `7caa751a-f132-4bb6-bf33-0804fe3051c0`
- Evidence: `../local_runs/v4_yolov9e_hybrid_roi_v4/quality_report.json`
- Detector: cached YOLOv9e VisDrone-style, 1920-pixel full frame
- Tracker: BoT-SORT, match threshold 0.80
- Results: 13,676 accepted detections; 11,740 observed rows; 181 native tracks; 164 final tracks; three stitches; zero expirations within 89 frames
- Decision: current short-clip baseline
- Limitation: no ground-truth metrics

## E004 — BoT-SORT match threshold 0.90

- Run ID: `5c267590-0457-4259-8282-40190fd11f60`
- Evidence: `../local_runs/v4_yolov9e_hybrid_roi_v4_match090/quality_report.json`
- Same detector cache and 89 frames as E003
- Results: 181 native tracks; 166 final tracks; three stitches
- Decision: reject; no proxy improvement over 0.80
- Limitation: no ground-truth metrics

## E005 — Current 15-second Level-1 candidate

- Date: 2026-08-24
- Run ID: `b3f327a8-bc5b-4405-968d-be1581880954`
- Code commit: `99d4974f6f14f1a48a3c247a905fbb60b2eb5a2d`
- Input hash: `6a997aeaae688932f0dd93dcff03e301d66e8f2d5ee50bfee07ec66b7fead42b`
- Tracks hash: `064709641693961cdfeac1e48ca454f7745487590961b96dd38bd3c6c9255c6f`
- Detector cache hash: `5091bf141e6af9cb490f5b1ebe8b2c2a9833322d20be233014614fa272d4dd0a`
- Evidence directory: `../local_runs/level1_v4_yolov9e_15s_roi_v4_3/`
- Human report: external output `FlytBase_Level1_V4_Candidate_Report.md`
- Frames: 449 at 3840 x 2160, 29.97 FPS
- Configuration: full-frame aerial cached detections, group-specific ROI, BoT-SORT 0.80, five-observation confirmation, two-frame visible prediction, offline stitching enabled
- Results: 58,353 accepted detections; 51,954 observed rows; 1,405 displayed short predictions; 279 native tracks; 241 final tracks; six stitches; 57 internal expirations
- Renderer: 53,359/53,359 boxes and visible IDs; zero missing ID labels
- Internal audit: zero persistent duplicates, zero heuristic handoffs, 118 internal starts, 100 internal ends, 40 unstable-class tracks
- Anchors: car 7 retains ID 7 through tree occlusion; car 2 retains ID 2 while stationary then moving; turning road user final ID 141 is stitched across sign occlusion; pedestrian-class anchors 50, 60, and 62 span all 449 frames
- Tests at candidate packaging: 73 passed
- Decision: best current visual candidate, **not verified**
- Blocking evidence: corrected COCO/MOT ground truth

## E006 — Exploratory OC-SORT replay

- Date: 2026-08-24
- Script: external workspace file `../ocsort_replay.py`
- Frames: 89, cached detections intended to match E003
- Reported result: 202 unique tracks versus 181 native tracks for BoT-SORT
- Decision: do not adopt
- Evidence limitation: output CSV, environment lock, manifest, and evaluator result were not preserved as a registered experiment. Repeat before treating this as proof.

## Next experiment ID

Use `E007` for the corrected 89-frame COCO/MOT baseline evaluation. It must preserve the annotation export, split definition, evaluator configuration, metrics, and error examples.
