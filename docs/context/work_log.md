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
