# Engineering decision log

Statuses: `accepted`, `provisional`, `rejected`, `deferred`, or `superseded`.

## D001 — Ground truth is the Level-1 release gate

- Status: accepted
- Date: 2026-08-23
- Decision: require corrected COCO/MOT labels and report detection precision/recall plus IDF1/HOTA before Level 1 can pass.
- Reason: internal duration, duplicate, lifecycle, and renderer metrics cannot reveal missed objects or incorrect physical identities.
- Consequence: the current candidate remains `not_evaluated` even though its verification video improved.

## D002 — Use an aerial-domain detector candidate

- Status: provisional
- Date: 2026-08-23
- Decision: use the community YOLOv9e VisDrone-style checkpoint with full-frame 1920-pixel inference as the current detector candidate.
- Reason: the generic COCO checkpoint produced roof/building false positives and lacks the required van/LGV aerial taxonomy. The aerial candidate covers pedestrian, people, bicycle, car, van, truck, bus, and motorcycle classes.
- Limitation: this checkpoint was not fine-tuned on the exact intersection and has no registered weight provenance or ground-truth score in this repository.
- Exit condition: replace or accept it only after labelled detector comparison.

## D003 — Do not use SAHI as the default

- Status: provisional
- Date: 2026-08-23
- Decision: retain SAHI as an experiment, not the default production path.
- Evidence: the 449-frame generic SAHI run `4c24ff8d-c5bf-4bfa-959a-bc389783c25d` produced 126,807 merged detections, 452 final tracks, 51,893 displayed predictions, and 141 internal expirations. The current full-frame aerial candidate produced 58,353 merged detections, 241 final tracks, 1,405 short predictions, and 57 expirations on the reviewed 449-frame clip.
- Reason: slicing enlarged tiny detail but also amplified domain-mismatched false positives and fragmentation. Counts are not accuracy, so the rejection is provisional until matched ground-truth recall is available.
- Reconsider when: a scene-fine-tuned detector plus sliced training/inference improves small-object recall and ID metrics on the same labelled split.

## D004 — Keep BoT-SORT as the current tracker candidate

- Status: provisional
- Date: 2026-08-24
- Decision: use BoT-SORT with `sparseOptFlow` camera-motion compensation, association groups, `track_buffer=180`, `match_thresh=0.80`, and `fuse_score=false`.
- Reason: the drone camera moves; camera-motion compensation is directly relevant. Low-confidence tiny aerial detections were penalised when score fusion was enabled.
- Important boundary: `track_buffer=180` retains a lost identity internally; it does not authorize drawing 180 predicted boxes.
- Exit condition: a tracker may replace it only on identical cached detections with corrected MOT ground truth and better HOTA/IDF1/ID-switch results.

## D005 — Restrict visible predictions to two frames

- Status: accepted
- Date: 2026-08-24
- Decision: render at most two missing-detection prediction frames, marked as occluded/predicted.
- Rejected alternative: rendering the full 180-frame buffer.
- Evidence: the earlier broad horizon produced 31,341 predicted rows; 22,920 were older than 15 frames, the median age was 46, and the maximum was 181. Manual review showed boxes drifting into empty space.
- Reason: a long Kalman-only extrapolation is not an observation and becomes unreliable for turning or tiny objects.

## D006 — Use class groups for association and temporal class voting

- Status: accepted
- Date: 2026-08-23
- Decision: associate related vehicle labels as `road_vehicle`, vulnerable users as controlled groups, and preserve raw class votes for later confidence-weighted classification.
- Reason: per-frame car/LGV/truck or pedestrian/bicycle/motorcycle confusion should not itself force a new physical identity.
- Constraint: association grouping must not be reported as perfect class consistency.

## D007 — Make ROI exclusions group-specific

- Status: accepted
- Date: 2026-08-24
- Decision: use association-group-specific exclusion polygons where possible.
- Evidence: a global foliage exclusion deleted car 7 after frame 299 even though high-confidence car detections existed. Restricting that polygon to vulnerable-road-user detections restored car 7 through frame 448.
- Reason: a false-positive region for one class may contain valid members of another class.

## D008 — Every confirmed rendered box needs a visible ID

- Status: accepted
- Date: 2026-08-24
- Decision: descriptive labels may be capped for readability, but every rendered track gets either a full label or compact `#ID`.
- Evidence: the previous 40-label cap suppressed 43,693 labels. The candidate renderer reports 53,359 boxes, 53,359 visible IDs, and zero missing ID labels.

## D009 — `match_thresh=0.90` does not replace 0.80

- Status: rejected
- Date: 2026-08-24
- Evidence: on the same 89-frame replay, both settings produced 181 native tracks. The 0.80 run produced 164 final tracks; 0.90 produced 166.
- Reason: no improvement in the available proxy metrics, and no ground-truth evidence of fewer identity switches.

## D010 — DeepSORT/appearance ReID is not an assumed solution

- Status: deferred
- Date: 2026-08-24
- Decision: do not switch to DeepSORT or enable ReID by reputation alone.
- Reason: many top-down pedestrian and motorcycle crops contain too few discriminative pixels. Appearance embeddings can help after longer occlusion, but may also create confident wrong matches.
- Required experiment: compare BoT-SORT without ReID, BoT-SORT/StrongSORT with aerial-appropriate ReID, and a motion-only candidate on identical detections and corrected MOT ground truth.

## D011 — OC-SORT remains a provisional rejection

- Status: provisional
- Date: 2026-08-24
- Observation: an exploratory 89-frame cached-detection replay reported 202 unique OC-SORT tracks versus 181 native BoT-SORT tracks, suggesting more fragmentation.
- Evidence limitation: the OC-SORT output artifact and manifest are not present in the repository; only the replay script remains outside the Git root. This is not reproducible proof.
- Decision: keep BoT-SORT for now, but repeat the comparison after ground truth and preserve all artifacts.

## D012 — Metric kinematics require validated projection

- Status: accepted
- Date: 2026-08-23
- Decision: pixel velocity may be reported as pixel velocity; m/s and km/h require validated telemetry or homography projection.
- Reason: perspective makes a pixel represent different ground distances at different image locations.
- Consequence: the Level-4 projection component is a shared prerequisite for defensible Level-2 real-unit motion.

## D013 — Full merged SRT alignment uses block order plus segment mapping

- Status: accepted
- Date: 2026-08-24
- Decision: represent merged-video output frame, subtitle block index, source segment, and raw `FrameCnt` separately.
- Evidence: both SRT files contain two raw frame-count sequences that reset to zero; raw counts also contain gaps, while SRT block counts exactly equal MP4 frame counts.
- Rejected alternative: joining the full merged video to telemetry using raw `FrameCnt` alone.
- Consequence: update the telemetry loader before full-video Level-2/4 processing.

## D014 — Detection improvement precedes further tracker selection

- Status: accepted
- Date: 2026-09-01
- Evidence: E007 achieved precision 0.6891, recall 0.5589, IDF1 0.6135, and HOTA 0.5547 on corrected development ground truth. Reconsidering all ROI-rejected detections raised raw recall only to 0.6848 while reducing precision to 0.5707.
- Decision: compare higher-resolution and sliced detector candidates, then fine-tune on development labels if raw precision and recall cannot both approach 0.92. Tune BoT-SORT only after freezing the detector cache.
- Constraint: the `da399` clip remains held out and may be evaluated only after the development configuration is frozen.

## D015 - Freeze the scene-refit full-frame detector and default BoT-SORT

- Status: accepted for held-out proof
- Date: 2026-09-02
- Evidence: E008/E009. The scene-refit candidate at uniform confidence 0.40
  achieved raw precision 0.9408 and recall 0.9211. Hash-identical cached replay
  then achieved precision 0.9391, recall 0.9267, IDF1 0.9230, HOTA 0.7968, and
  mode accuracy 1.0000 on corrected development ground truth.
- Decision: freeze full-frame 1920 inference, uniform class confidence 0.40,
  no scene road-user ROI, the existing default BoT-SORT association settings,
  five-observation confirmation, the two-frame maximum visible prediction rule,
  and offline stitching.
- Reason: this is the highest-HOTA passing development configuration. Removing
  the ROI recovered valid road users, while the 0.40 detector threshold retained
  the raw precision/recall margin. Alternative tracker thresholds did not beat
  the default settings on identical detections.
- Constraint: development success is not Level-1 completion. The `da399` clip
  must now receive independent manual ground truth and one frozen evaluation;
  it must not become a repeated tuning set.
