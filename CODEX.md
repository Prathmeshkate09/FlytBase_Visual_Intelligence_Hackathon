# FlytBase Visual Intelligence — durable project context

Last updated: 2026-08-24

This is the compact project handbook. `AGENTS.md` tells an agent how to work; this file tells it what the project is, what has been learned, and what to do next. Detailed evidence lives in `docs/context/`.

## Mission

Build a real-world aerial traffic-intelligence system that turns drone video and SRT telemetry into:

1. persistent identities for every relevant road user;
2. reliable object class and real-unit kinematics;
3. aggregate traffic measures;
4. map- and lane-grounded trajectories; and
5. evidence-backed reasoning about congestion, signals, conflicts, desire lines, and obstruction.

The five levels are one dependency chain, not five disconnected demos:

```text
4K video + SRT telemetry
        |
        v
L1 detections and persistent identities
        |
        +------> L2 object attributes and metric kinematics
        |                    ^
        |                    |
        +------> shared ground projection <------ L4 map/lane grounding
        |
        +------> L3 counts, flows, queues and distributions
                             |
                             v
                 L5 spatiotemporal reasoning
```

## Current truth

- Repository: `FlytBase_Visual_Intelligence_Hackathon`
- Active branch at this snapshot: `level1-v4`
- Snapshot commit: `99d4974f6f14f1a48a3c247a905fbb60b2eb5a2d`
- Level 1 status: **candidate requiring corrected COCO/MOT ground-truth verification**.
- Level 2 status: blocked by the strict Level-1 quality gate.
- Levels 3-5: challenge requirements captured; production implementation not started.
- Current detector candidate: community YOLOv9e checkpoint trained for VisDrone-style aerial classes, full-frame inference at 1920 pixels. It is not fine-tuned on this intersection.
- Current tracker candidate: BoT-SORT with sparse optical-flow camera-motion compensation, class-group association, short visible prediction horizon, and auditable offline stitching.
- Current biggest bottleneck: corrected scene-specific ground truth and recall/classification for tiny pedestrians, bicycles, and motorcycles.

Do not translate “candidate” into “accurate.” The current video is materially better, but there is no formal ground-truth score yet.

## Selected and rejected directions

Selected provisionally:

- Aerial-domain detector over generic COCO detector.
- Full-frame YOLOv9e/VisDrone inference as the current detector baseline.
- BoT-SORT because the drone camera moves and sparse optical-flow compensation is already integrated.
- Association groups so temporary car/LGV/truck class changes do not create separate tracker pools.
- Two-frame visible prediction horizon; longer lost-track state may be retained internally but not drawn as observed.
- Group-specific ROI exclusions rather than global polygons.
- Every rendered confirmed box receives a full label or compact `#ID`.
- Official COCO/MOT evaluation as the quality gate.

Rejected or deferred:

- Generic COCO YOLO as the final detector: domain and taxonomy mismatch.
- SAHI as the default: the tested generic sliced run produced many more tracks, predictions, and internal expirations; keep it experimental until ground-truth recall proves a net gain.
- A 180-frame visible Kalman prediction horizon: it created boxes in empty space.
- Raising BoT-SORT `match_thresh` from 0.80 to 0.90: the 89-frame replay did not reduce native tracks and slightly increased final tracks.
- DeepSORT/ReID as an assumed fix: deferred because tiny top-down crops may not contain discriminative appearance. It must win a controlled evaluation before adoption.
- OC-SORT as the current default: an exploratory 89-frame replay fragmented more identities than the current BoT-SORT replay, but its result artifact was not preserved in the repository, so this remains a provisional rejection rather than formal proof.

See `docs/context/decision_log.md` for reasons and evidence.

## Current Level-1 candidate evidence

Run ID: `b3f327a8-bc5b-4405-968d-be1581880954`

- 449 frames at 3840 x 2160 and 29.97 FPS.
- 58,353 accepted merged detections.
- 51,954 observed track rows.
- 1,405 short displayed prediction rows.
- 241 final tracks after six audited stitches.
- 57 internal expirations.
- 53,359 of 53,359 rendered track rows have a box and visible ID.
- Internal audit: zero persistent duplicate pairs and zero heuristic handoffs, but 118 internal starts, 100 internal ends, and 40 unstable-class tracks.
- Test snapshot: 73 tests passed.
- Formal quality: `not_evaluated` because corrected ground truth is absent.

Anchor checks include car 7 retaining ID 7 through a tree occlusion, car 2 retaining ID 2 while stopped and then moving, a turning motorcycle-class track stitched across a sign occlusion, and long-lived pedestrian-class anchors. These anchors are not a substitute for whole-clip annotation.

## Dataset

The extracted local dataset has two 4K, 29.97 FPS videos and matching DJI-style SRT files. The intersection video is approximately 6:39; the multi-road video is approximately 5:05.

Critical telemetry fact: both SRT files contain two `FrameCnt` sequences that reset to zero, while each MP4 has one continuous output-frame sequence. Full-video projection must use subtitle block order plus an explicit source-segment mapping. Do not join the entire merged video to telemetry on raw `FrameCnt` alone.

See `docs/context/dataset_registry.md`.

## Next executable milestone

Create and correct a small but complete labelled evaluation set before more tracker tuning:

1. Use 89 frames as the development/annotation pilot.
2. Correct every relevant road user and identity in CVAT.
3. Export COCO detection labels and MOT 1.0 identities.
4. Run `evaluate_v4.py` and record precision, recall, IDF1, HOTA, ID switches, and class accuracy.
5. Fix the layer actually responsible for failures.
6. Freeze the passing Level-1 manifest.
7. Build the shared SRT-to-ground projection needed by Level 2 and Level 4.

## Context update protocol

At the end of every material experiment or architecture change:

1. append a concise entry to `docs/context/work_log.md` describing the action,
   evidence, engineering rationale, result, limitations, and next step;
2. append one experiment entry when a material experiment ran;
3. add or supersede a decision when the conclusion changes;
4. update `current_state.json` and this compact summary when the project state changes;
5. run `python tools/context_status.py` and tests;
6. commit source and context together.

Never erase rejected experiments. Mark them superseded and explain why.
