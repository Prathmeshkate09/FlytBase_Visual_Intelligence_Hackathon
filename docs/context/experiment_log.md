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

## E007 — Corrected 89-frame development baseline

- Date: 2026-09-01
- Run ID: `fb89713e-c1e8-4cdf-ac60-4d1b78efc5a2`
- Development video SHA-256: `c8ead5bc7f3fd82dfd3dfe345061996f8822e9a8bea2048b16b58d7b7edbeda1`
- MOT archive SHA-256: `36cbe4c61077f9676f1a04e8dd6839a22c778ee9266fb3dcb0602243b96049ae`
- Tracks SHA-256: `ee2fa6594ae5879cdb48c1f8db18432e5c42cc4f32189c8ab01144d0c1a1aad1`
- Evidence: `../evaluation_results/e007_dev89_v9/quality_report_ground_truth.json` and `error_diagnostics.json`
- Configuration: YOLOv9e VisDrone-style full-frame 1920, scene ROI, grouped BoT-SORT, five-observation confirmation, two-frame predictions, offline stitching
- Counts: 89 frames, 14,415 ground-truth boxes, 11,692 prediction boxes, 174 ground-truth tracks, 164 prediction tracks
- Metrics: precision 0.6891, recall 0.5589, IDF1 0.6135, HOTA 0.5547, mode accuracy 0.8954, 10 ID switches, 157 fragmentations
- Pipeline losses: 12,171 accepted detections, 11,407 native observed rows, and 11,358 confirmed observed rows; the historic run predates raw candidate-cache export
- Error concentration: 3,429 motorcycle and 2,338 pedestrian false negatives dominate recall loss; false positives are concentrated in motorcycles and pedestrians
- Decision: valid development baseline, failed four of five quality gates; improve detection before treating tracker tuning as the primary lever

## E008 - Controlled detector comparison and scene refit

- Date: 2026-09-02
- Code commit: `fb2f4545d2bb9cd5549f6a3c2b267121dc2c87b0`
- Kaggle kernel: `seb09prathameshkate/flytbase-v4-gpu-runner`, version 12
- Development video SHA-256: `c8ead5bc7f3fd82dfd3dfe345061996f8822e9a8bea2048b16b58d7b7edbeda1`
- Reconstructed MOT SHA-256: `b10b5e92988aff11848a18b35e1f6b830f5ca4861d631ff20aad7cce90edac45`
- Pretrained tracked results: full 1920 was best at precision 0.6805, recall
  0.5598, IDF1 0.6107, and HOTA 0.5535; full 2560 and SAHI 1280 did not improve
  the complete gate result.
- Training: frames 0-70 train and 71-88 validation, 80 epochs maximum, seed 42,
  image size 1280, AMP, batch 4; validation selected epoch 67. Refit used all 89
  development frames for 67 epochs with batch 1.
- Frozen model: `last.pt`, 117,334,505 bytes, SHA-256
  `02382dcc750f479ea23bfb6e51f82cbf257b1ffd95883b6b05b61822b3ea36c4`.
- Raw scene-full operating points: confidence 0.30 produced precision 0.9149
  and recall 0.9439; confidence 0.40 produced precision 0.9408 and recall 0.9211.
- Decision: freeze the scene-refit full-frame candidate cache and tune
  post-detection/tracking from that immutable cache.
- Limitation: these are development results; the held-out labels were not used.

## E009 - Frozen-candidate confidence, ROI, BoT-SORT, and lifecycle sweep

- Date: 2026-09-02
- Code commit: `4ae29363812a0d12157f1ecfe240ce5df8965151`
- Kaggle kernel: `seb09prathameshkate/flytbase-v4-cached-tuning-runner`, version 1
- Candidate cache SHA-256: `99e047847b9f65f439af640aeccdae99f2f819cdcca5cf5e86321818cd1c4a6e`
- Inputs: the E008 development video and reconstructed MOT archive; 18 focused
  Kaggle tests passed before replay.
- Sweep: seven confidence variants with and without ROI, eight association
  variants, then confirmation 1/3/5, prediction horizon 0/2, and stitching
  disabled/enabled for the two best association variants; 46 total runs.
- Winner run ID: `e18e20ab-85db-40b7-b5ee-fee5213617c4`.
- Winner: uniform confidence 0.40, no ROI, default BoT-SORT, confirmation 5,
  two-frame visible prediction horizon, offline stitching enabled.
- Counts: 89 frames, 14,415 ground-truth boxes, 14,225 prediction boxes, 174
  ground-truth tracks, and 177 prediction tracks.
- Metrics: precision 0.9391, recall 0.9267, IDF1 0.9230, HOTA 0.7968, mode
  accuracy 1.0000, 21 ID switches, and 100 fragmentations; all five development
  gates passed.
- Hashes: tracks `20cd0d5f...8976a`, track summary `42416823...48220`, run
  manifest `b6c9ebae...862ef`, and tuning summary `199a6697...74059`.
- Decision: accept as the frozen development configuration for held-out proof.
- Limitation: Level 1 remains incomplete until the separately corrected held-out
  clip passes all gates in one frozen evaluation.

## Next experiment ID

Use `E010` for the one-time frozen held-out evaluation after manual correction.
