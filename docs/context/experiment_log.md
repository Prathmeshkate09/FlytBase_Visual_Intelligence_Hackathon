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

## E010 - One-time frozen held-out evaluation

- Date: 2026-09-19; evaluator commit: `3e8d76ddf545d2c1ed3a8f1ae8509a67c355e614`.
- Inference commit: `595ab7bd1c187e2b9620e76d12a6a93da0628303`;
  Kaggle seed version 4, run `ff2b49d3-a668-40ac-b3f0-50cb728ad8aa`.
- Inputs: exact da399 held-out video, frozen seed archive SHA-256
  `5ae95e0ea6a56d2215b81fcddb4c8d5b0876bb89b4cd71f543c2c49cefbe4135`,
  user-corrected CVAT task 2570030 exports dated 2026-09-19.
- Full input hashes and cross-format validation:
  `C:/Users/PRATHAMESH/Documents/Codex/2026-09-05/the-frozen-held-out-seed-is/outputs/e010_heldout/preflight.json`.
- Configuration: E009 frozen scene-refit full-frame 1920, confidence 0.40,
  no ROI, default BoT-SORT, confirmation 5, visible prediction horizon 2, stitching.
- Counts: 89 frames; 171 GT identities / 14,273 boxes; 143 prediction
  identities / 10,742 boxes. All three GT formats agree after MOT ID renumbering.
- Metrics: precision 0.9883634333, recall 0.7438520283,
  IDF1 0.8470117929, HOTA 0.8331234233, mode accuracy 0.9791843270;
  one ID switch, 84 fragmentations. Recall and IDF1 fail.
- Diagnostics: 1,809 missed pedestrian, 1,758 missed motorcycle and 89 missed
  car frame-boxes. Category-agnostic matching; class accuracy reported separately.
- Tests: 8 existing evaluator tests passed outside Windows sandbox temp restrictions.
  Installed TrackEval provenance matches pinned commit 12c8791.
- Preservation: original manifest and prediction files retained byte-for-byte.
  Evaluation-only manifest omits unavailable cache output links and records the
  original manifest hash; no predictions or numerical configuration changed.
- Limitations: Kaggle retains only the handoff ZIP, so raw candidate/accepted/native
  stage counts are unavailable. User reported exhaustive correction; 12,729 XML
  visible boxes retain unchecked review flags (1,518 corrected, 26 confirmed).
  No independent full-frame visual or anchor audit was performed. No renderer
  completeness claim or broad generalization claim is made for this 2.97-second clip.
- Conclusion: Level 1 fails held-out gates; Level 2 stays blocked. Detection
  coverage is the dominant observed deficit; exact pipeline cause is unresolved.
  Do not retune on this interval or repeat its score as new unseen evidence.

## Next experiment ID

E011 is recorded below. Use E012 for the next independently validated candidate
experiment; register validation/test intervals before any candidate selection.

## E011 - Small-object development threshold and resolution experiment

- Date: 2026-09-20. User explicitly authorized private Kaggle upload and GPU run.
- Kernel: seb09prathameshkate/flytbase-e011-small-object-development, v2 COMPLETE.
- Code: 595ab7bd1c187e2b9620e76d12a6a93da0628303; GPU capability 7.5.
- Inputs: E009 c8ead5 development video, 02382d frozen weights, 99e047 candidate
  cache and b10b5e reconstructed MOT, all verified by full SHA-256.
- Five prespecified runs, unchanged association/lifecycle, no retraining.
- Baseline 1920 confidence .40 exactly reproduces E009.
- 1920 vulnerable-user .30: precision .913538, recall .947000, IDF1 .913686,
  HOTA .787966; all development gates pass.
- 1920 vulnerable-user .20: precision .901408, recall .959001, IDF1 .912037,
  HOTA .783653; all development gates pass but precision margin is thin.
- 2560 .40: precision .911531, recall .834131, IDF1 .857640, HOTA .711134.
  2560 vulnerable-user .30: precision .885015, recall .875130, IDF1 .859744,
  HOTA .713662. Both fail development gates.
- Evidence: current task outputs/e011_results/FlytBase_E011_Development.zip,
  10,516,561 bytes; SHA-256
  8b6f27b745dfbb4f8015906c9ead82bdb0325a6074b57e84ddeab9838d896a08.
  Every one of 82 internal hashes and ZIP CRC passed. Full stage outputs retained.
- Verification: 15 worker tests passed; baseline metrics within 1e-6 of E009.
- Conclusion: .30 vulnerable-user threshold is a validation candidate only;
  higher resolution is not supported. No replacement adopted, baseline remains
  strongest on development HOTA/IDF1. E010 unchanged; Level 1 still failed.
- Limitations: these are the original training/development frames; no independent
  validation or unseen claim. No new visual anchor audit performed.

## E012 - Recover detector-stage evidence and diagnose exposed E010 clip

- Date: 2026-09-20; user asked to continue using existing annotations and
  authorized continuing E012 after its specific remote-upload approval block.
- Kernel: seb09prathameshkate/flytbase-e012-cache-diagnosis, v1 COMPLETE.
- Code 595ab7b; original da399 video and 02382d model hashes pinned. No labels
  uploaded. GPU job generated baseline .40 and prespecified VRU .30/.20 replays;
  local scoring used corrected task 2570030 labels as post-hoc diagnostics.
- This is exposed-test diagnostic reuse, not another independent evaluation.
  Original E010 report and frozen configuration remain unchanged.
- Baseline raw candidate recall .795978; thresholded .725636; accepted .725636;
  native observed .724094; final .743712. Of 3,658 unmatched final boxes, 2,856
  lacked a raw match, 767 lost their match at thresholding, 0 at subsequent
  postprocessing, and 35 after accepted detections. Stage assignment differences
  mean these are diagnostic partitions, not individual causal proofs.
- Diagnostic precision/recall/IDF1/HOTA:
  baseline .988177/.743712/.846852/.832500;
  VRU .30 .920131/.769215/.830452/.814444;
  VRU .20 .899205/.776921/.826612/.809364. All fail recall/IDF1.
- Raw candidates already include NMS and .05 floor. Main deficit precedes
  tracking; low-confidence filtering contributes but does not explain most loss.
- Regeneration is not byte-identical: candidate/track hashes differ. Baseline
  recall differs from E010 by -.014 percentage points and HOTA by -.062 points.
  Cause of numerical drift not established; diagnostics kept separate.
- Evidence: task outputs/e012_results, archive 5,027,107 bytes, SHA-256
  296ad7e3ebfc0d68508615b0bb77f406e11fe2de5f6e0adf44577e9db1f23510.
  ZIP CRC, outer hash and all 47 internal hashes passed.
- Tests: eight worker tests and local synthetic stage/frame-offset check passed.
- Conclusion: lower thresholds alone rejected as solution; prioritize
  small-object candidate generation using existing labels. No new training or
  model adoption occurred. E013 is the next detector-improvement experiment.
