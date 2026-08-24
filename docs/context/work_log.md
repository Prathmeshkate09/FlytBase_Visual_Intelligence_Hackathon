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
