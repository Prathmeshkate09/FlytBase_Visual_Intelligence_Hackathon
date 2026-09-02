# Engineering work log

This is an append-only record of material work on the FlytBase project. It
records actions, inspected evidence, engineering rationale, results,
limitations, and the next step. It does not contain credentials, private data,
or hidden chain-of-thought. Engineering rationale is written as concise,
reviewable reasons that another engineer can verify.

## Entry template

```text
## YYYY-MM-DD HH:MM IST - short title

- Request:
- Actions and evidence:
- Engineering rationale:
- Result:
- Limitations:
- Next:
```

## 2026-08-24 - Prepare the beginner CVAT ground-truth workflow

- Request: persist what was done and why for every material action, and explain
  exactly what the project owner should upload to CVAT from the available 6 GB
  and 4 GB source videos.
- Actions and evidence:
  - inspected the repository context, current Level-1 gate, CVAT seed generator,
    existing development/held-out annotation packages, and the two original
    videos under `D:\Flybase`;
  - confirmed `Intersection_Merged.MP4` is 6,495,679,393 bytes and
    `Multi_Road_Merged.MP4` is 4,959,342,247 bytes;
  - confirmed the development annotation clip is 3840 x 2160, 89 frames, about
    2.97 seconds, and 48,862,056 bytes;
  - compared decoded frames 0, 1, 44, and 88 of the development clip with the
    same frames of the original intersection video; sampled pixel arrays were
    identical (mean absolute error 0.0);
  - checked the current official CVAT task-creation, track-mode, annotation
    import, COCO, and MOT documentation;
  - added a paste-ready Level-1 label schema and a beginner task/review/export
    guide; and
  - removed held-out anchor hints from the generic seed instructions because
    they were incorrectly included in the development package and could bias a
    later held-out review;
  - created the local destination directory for corrected CVAT, MOT, and COCO
    exports;
  - `python tools/context_status.py --check-local-assets` passed;
  - the eight focused context/CVAT tests passed in the project virtual
    environment; and
  - the full suite reached 62 passes and 16 setup errors caused by Windows
    `PermissionError` while pytest created its temporary directory. No full-suite
    assertion failure was observed, but the full suite is not recorded as
    passing.
- Engineering rationale: uploading the complete source videos is slow and risks
  mixing development with held-out evaluation. A short, exact 4K development
  clip makes exhaustive correction feasible while preserving the tiny-object
  evidence needed to diagnose detection and identity failures. Model seed boxes
  reduce manual work but remain predictions until every frame is corrected.
- Result: the next user action is a single 89-frame CVAT development task using
  `source_video_89f.mp4`, followed by import of `annotations.xml` and complete
  manual correction.
- Limitations: sampled frame equality verifies the tested frame indices, not an
  independent byte-level proof for every decoded frame. Formal precision,
  recall, IDF1, and HOTA remain unavailable until corrected exports exist. The
  pre-existing Windows pytest temporary-directory permission issue still blocks
  a clean full-suite result in this shell.
- Next: receive corrected CVAT for video, MOT, and COCO exports; validate them;
  then run experiment E007 before changing the detector or tracker again.

## 2026-08-24 - Correct CVAT Online Raw-label validation failure

- Request: resolve the CVAT task-creation error shown in the project owner's
  screenshot: `v4_track_id: attribute values must be a non-empty array`.
- Actions and evidence: inspected the exact failing Raw JSON and current CVAT
  attribute serializer behavior. The schema defined the free-text
  `v4_track_id` attribute with `"values": []`; CVAT Online rejected it before
  the task could be saved. Updated all nine label definitions to use the
  non-empty placeholder `"values": [""]`, and added a regression assertion.
- Engineering rationale: retain `v4_track_id` because the imported seed XML uses
  it to map model IDs during failure analysis. A single blank placeholder meets
  the Raw editor's array requirement without restricting free-text track IDs.
- Result: the corrected JSON can be pasted into the still-open task dialog.
- Limitations: local JSON and regression validation cannot submit the form in the
  user's authenticated CVAT Online session. The user must confirm that the Save
  button becomes available.
- Next: paste the corrected complete JSON, save the labels, then upload the
  89-frame development video.

## 2026-08-24 - Declare the CVAT Raw label type explicitly

- Request: resolve the next task-creation error shown in the project owner's
  screenshot: `Label "car": unknown label type "undefined"`.
- Actions and evidence: traced the error to the Raw schema omitting the `type`
  property. Added `"type": "any"` to all nine labels, matching the label type
  already embedded in the CVAT 1.1 seed XML, and added a regression assertion.
- Engineering rationale: `any` preserves compatibility with rectangle tracks
  imported from the existing seed and avoids changing the annotation semantics
  while satisfying CVAT Online's explicit type validation.
- Result: the corrected complete JSON is ready to replace the Raw editor text.
- Limitations: final acceptance still requires clicking Save in the user's
  authenticated CVAT session.
- Next: replace the complete Raw JSON, click Save, and report either the enabled
  file upload section or the exact next validation message.

## 2026-08-24 - Review CVAT video task settings

- Request: verify the CVAT advanced settings and explain why the uploaded clip
  is displayed as two seconds.
- Actions and evidence: inspected the task-creation screenshot. Start frame 0,
  stop frame 88, frame step 1, segment size 89, overlap 0, chunk size 4,
  consensus replicas 0, and cache enabled were configured correctly. Image
  quality remained 70 and Prefer zip chunks remained enabled.
- Engineering rationale: 89 frames at 29.97003 FPS represent approximately
  2.97 seconds, so a whole-second UI display can show `2 sec` without frames
  being missing. Image quality 100 is retained for this 4K ground-truth task to
  reduce loss of detail on tiny pedestrians and bicycles. Native video chunks
  are preferred over ZIP chunks for video annotation.
- Result: two changes are required before task submission: set image quality to
  100 and turn Prefer zip chunks off. The frame range itself is correct.
- Limitations: the screenshot is the creation form, not the processed task. The
  final task must still be checked for frame range 0-88 after creation.
- Next: apply the two changes, create the task, and confirm that the annotation
  job can navigate to frame 88 before importing the seed XML.

## 2026-08-24 - Choose CVAT validation mode for the pilot

- Request: review the final CVAT task-creation section showing Local source and
  target storage plus None, Ground Truth, and Honeypots validation modes.
- Actions and evidence: checked the current official CVAT quality-control
  documentation. CVAT's Ground Truth mode creates a separate validation job and
  compares regular annotation jobs against it; its annotations remain separate
  from ordinary task annotations. Honeypots are not suitable for ordered video
  tracks.
- Engineering rationale: this pilot needs one complete, manually corrected
  89-frame tracking job that can receive the seed XML and be exported directly
  as CVAT, MOT, and COCO. CVAT's optional internal annotator-QA subsystem adds a
  different job topology without improving this single-owner correction pass.
- Result: keep source storage Local, target storage Local, and validation mode
  None; then use Submit & Open.
- Limitations: independent annotation quality review is still required outside
  CVAT's automated QA mode before the labels are treated as final ground truth.
- Next: process the task, confirm frame 88 exists, and import `annotations.xml`
  into the regular task before any manual edits.

## 2026-08-24 - Confirm the CVAT development task

- Request: verify the newly created CVAT task from its task-details screenshot.
- Actions and evidence: confirmed task `2540401`, regular annotation job
  `4400803`, all nine required labels, frame count 89 at 100%, and frame range
  0-88. The job is in annotation stage and new state.
- Engineering rationale: frame count and range must be confirmed before seed
  import; otherwise annotation identities and boxes could be shifted onto the
  wrong frames and produce invalid evaluation ground truth.
- Result: the media task is structurally correct and ready for seed annotation
  import.
- Limitations: the screenshot does not yet show imported boxes or prove that the
  XML importer accepts the schema.
- Next: open job `4400803`, choose Menu -> Upload annotations, select CVAT for
  video 1.1, and upload the development package's `annotations.xml` before any
  manual edits.

## 2026-08-24 - Review the CVAT seed import dialog

- Request: verify the annotation-import settings for job `4400803`.
- Actions and evidence: inspected the import dialog showing format CVAT 1.1,
  `annotations.xml` selected, default settings enabled, and import mode Replace.
  Convert masks to polygons was enabled even though the seed contains rectangle
  tracks and no masks.
- Engineering rationale: Replace is safe because this is a new empty job and
  prevents accidental duplication. CVAT 1.1 matches the seed XML schema. Mask
  conversion has no role in a bounding-box tracking task and is disabled to
  keep import behavior minimal and explicit.
- Result: keep CVAT 1.1, default settings, and Replace; turn Convert masks to
  polygons off; then confirm the import.
- Limitations: successful import and track/frame alignment are not proven until
  CVAT finishes processing and rendered seed boxes are inspected.
- Next: click OK, wait for completion, and inspect frame 0 and frame 88 before
  beginning manual corrections.

## 2026-08-24 - Confirm replacement of the empty CVAT job

- Request: review CVAT's final `Replace existing annotations?` confirmation for
  job `4400803`.
- Actions and evidence: confirmed this is the expected second-stage warning from
  Replace import mode and that no manual correction work has begun in the new
  job.
- Engineering rationale: replacing an empty job with the preserved seed XML is
  deterministic and avoids duplicate tracks. Replacement would become unsafe
  after manual corrections, so this approval applies only to this initial
  import.
- Result: approve `Replace annotations` and wait for processing to complete.
- Limitations: the imported annotations still require visual frame-alignment and
  box/identity checks.
- Next: inspect the rendered tracks at frames 0 and 88, then mark the seed import
  state in the project context.

## 2026-08-24 - Confirm CVAT seed import and identify initial correction scope

- Request: inspect the first screenshots after importing the seed XML.
- Actions and evidence: confirmed that rectangle tracks, labels, seed attributes,
  and object lists render in job `4400803`. The screenshots show frames 1 and 87
  rather than the requested boundary frames 0 and 88. Visible seed failures
  include pedestrian-labelled roof/building details, boxes without a real object,
  and oversized/drifted boxes near structures.
- Engineering rationale: successful rendering proves the import pathway works,
  but it does not prove accuracy. These false positives are retained in the seed
  intentionally so the human correction pass can distinguish detector,
  association, and prediction failures instead of hiding them before labelling.
- Result: mark the seed import successful and the task as awaiting complete
  manual correction; do not classify it as ground truth yet.
- Limitations: exact boundary alignment at frames 0 and 88 has not been visually
  captured, and no object identity has been manually validated end-to-end.
- Next: save, type frame 88 explicitly and inspect it, then return to frame 0 and
  begin the false-track/coverage correction pass.

## 2026-08-24 - Confirm exact CVAT boundary-frame alignment

- Request: inspect explicit screenshots of the first and last frames after seed
  import.
- Actions and evidence: confirmed the CVAT navigation field shows frame 0 in the
  first screenshot and frame 88 in the second. Imported rectangle tracks render
  over the same intersection scene at both boundaries, and seed attributes such
  as `v4_track_id`, `seed_state`, and `review_status` are available.
- Engineering rationale: verifying both inclusive boundaries rules out a common
  one-frame offset or truncated-job error before any human correction changes
  the source evidence.
- Result: mark seed-to-video boundary alignment as visually confirmed. The task
  remains uncorrected seed data, not ground truth.
- Limitations: alignment does not validate any box, class, or physical identity.
  Visible roof/empty-space false tracks and drifted boxes remain.
- Next: save the imported job and begin Pass A at frame 0 by reviewing and
  removing only visually proven false tracks, while preserving ambiguous tracks
  for closer inspection.

## 2026-08-24 - Verify CVAT frame resolution after quality concern

- Request: determine whether the development job is actually using 4K imagery
  because the annotation view appears soft.
- Actions and evidence: inspected authenticated CVAT job `4400803` read-only in
  the browser. Its active source-frame canvas reports width `3840` and height
  `2160`, confirming that the current job is decoding a 4K frame. The job is
  displayed fit-to-workspace on a substantially smaller screen area. The task
  creation screenshot previously showed Image quality `70`; no post-creation
  evidence currently proves that value was changed to `100`.
- Engineering rationale: source resolution, browser display scale, and CVAT's
  annotation-image compression are separate variables. A 3840x2160 frame fitted
  into the browser can look soft even though the stored geometry remains 4K;
  quality 70 can additionally obscure tiny pedestrians and bicycles.
- Result: source-frame resolution is verified as 3840x2160. Do not treat the
  fit-to-screen view as a resolution failure. Use region zoom/fullscreen for
  inspection. Because image quality 100 is not verified, do not begin the
  expensive manual correction pass until the task-compression decision is made.
- Limitations: the current CVAT UI inspection proves canvas dimensions but does
  not expose the immutable task-creation Image quality value. The only captured
  creation value is 70.
- Next: compare a deeply zoomed tiny road user in this job against the local 4K
  frame. If compression impairs object boundaries, recreate the empty/manual-
  work-free task with Image quality 100, Prefer zip chunks off, chunk size 4,
  frames 0-88, and frame step 1, then re-import the preserved seed XML.

## 2026-09-02 20:16 IST - Publish Kaggle scene-refit fixes

- Request: push the relevant completed Level-1 code to GitHub.
- Actions and evidence:
  - inspected the repository root, required project context, active branch,
    working tree, outgoing diff, and GitHub remote;
  - confirmed that `level1-v4` was three commits ahead of its remote and that
    the outgoing changes affected only the synchronized Kaggle notebook source
    and notebook files;
  - preserved the two unrelated untracked detector/ROI configuration files;
  - fetched `origin/level1-v4` and confirmed zero remote-only commits before
    pushing commits `9f0cffe`, `971d836`, and `fb2f454`; and
  - verified that the local and GitHub branch heads both resolved to
    `fb2f4545d2bb9cd5549f6a3c2b267121dc2c87b0` after the push.
- Engineering rationale: the changes make Kaggle ground-truth discovery work
  with unpacked inputs, correct the all-frame scene-refit input, evaluate raw
  fine-tuned candidates, and normalize scene-model classes to the canonical
  FlytBase taxonomy. Fetching first prevented accidental remote overwrite.
- Result: the relevant scene-fine-tuning workflow is published on GitHub branch
  `level1-v4`.
- Limitations:
  - `tools/context_status.py --check-local-assets` reports the four compressed
    media/SRT copies missing from their registered local directory;
  - the local full pytest run reached 67 passes but was not clean because
    Windows denied pytest temporary-directory access and the explicit local
    asset-presence test failed; the earlier Kaggle smoke test remains the clean
    code-test evidence for this notebook revision; and
  - the live fine-tuning/evaluation run must finish before its metrics can be
    recorded as experiment evidence.
- Next: retrieve the completed Kaggle artifacts, validate their hashes and
  metrics, and record the detector-selection conclusion before freezing the
  detection cache for BoT-SORT tuning.

## 2026-09-02 - Start frozen scene-detector replay tuning

- Request: continue from the completed scene fine-tuning run and improve the
  remaining Level-1 recall and identity gates.
- Actions and evidence:
  - verified that Kaggle v12 completed without a traceback and downloaded only
    its small evaluation reports and frozen replay inputs;
  - verified development video hash `c8ead5bc...beda1`, candidate-cache hash
    `99e04784...1c4a6e`, reconstructed MOT hash `b10b5e92...dac45`, and detector
    config hash `8462f2b2...6fe84`;
  - confirmed the fine-tuned full-frame detector has raw passing operating
    points at confidence 0.30 and 0.40, while the existing replay sweep stopped
    below that range;
  - added uniform 0.30, 0.35, and 0.40 confidence variants plus a regression
    test, which passed with the existing tuning tests (`3 passed`); and
  - added a private, CPU-only Kaggle runner that consumes the completed v12
    output as a kernel source and enforces every input hash before replay.
- Engineering rationale: the detector already exceeds the raw precision and
  recall margin, so repeating GPU training would add cost without addressing
  the observed post-detection loss. Cached replay keeps detector observations
  identical while isolating confidence, ROI, association, and lifecycle effects.
- Result: the reproducible tuning-only runner is ready to publish and execute.
- Limitations: the local virtual environment lacks PyTorch, which BoT-SORT
  requires even when detector inference is cached; no local replay metrics were
  produced. The held-out clip was not opened or evaluated.
- Next: run the hash-pinned Kaggle replay, rank only development configurations,
  and freeze the detector/tracker configuration only if every development gate
  passes.
