# Metrics, mathematics, and acceptance gates

These are project gates, not official hackathon scoring thresholds. They make claims falsifiable and comparable.

## Detection

For a predicted box \(B_p\) and ground-truth box \(B_g\):

\[
IoU(B_p,B_g) = \frac{|B_p \cap B_g|}{|B_p \cup B_g|}
\]

At a declared IoU threshold, usually 0.5 for this project:

\[
Precision = \frac{TP}{TP+FP}, \qquad Recall = \frac{TP}{TP+FN}
\]

Precision measures how many reported objects are real. Recall measures how many real objects were found. A tracker cannot recover an object through frames in which the detector never provides a usable observation.

Report overall road-user precision/recall and per-class results. Also report small-object recall by labelled box area; otherwise large parked cars can hide failures on pedestrians and bicycles.

## Identity tracking

Identity precision, recall, and F1 are:

\[
IDP = \frac{IDTP}{IDTP+IDFP}, \qquad
IDR = \frac{IDTP}{IDTP+IDFN}
\]

\[
IDF1 = \frac{2\,IDTP}{2\,IDTP+IDFP+IDFN}
\]

HOTA balances detection and association. At a localisation threshold \(\alpha\):

\[
HOTA_\alpha = \sqrt{DetA_\alpha \cdot AssA_\alpha}
\]

The reported HOTA score averages over localisation thresholds. Use the pinned TrackEval implementation rather than a handwritten approximation.

MOTA is retained for diagnosis:

\[
MOTA = 1 - \frac{FN + FP + IDSW}{GT}
\]

It can obscure identity quality when detection errors dominate, so HOTA and IDF1 are the primary tracking gates.

Also report:

- ID switches;
- fragmentations;
- mostly tracked and mostly lost trajectories;
- internal expirations away from valid exits;
- duplicate identities on one physical object; and
- track recovery after occlusion.

## Level-1 release gate

Provisional minimums on corrected held-out ground truth:

| Measure | Gate |
|---|---:|
| Road-user precision | >= 0.90 |
| Road-user recall | >= 0.90 |
| IDF1 | >= 0.85 |
| HOTA | >= 0.70 |
| Persistent duplicate-track rate | < 0.01 |
| Renderer box completeness | 1.00 |
| Renderer visible-ID completeness | 1.00 |

Additional non-numeric gates:

- no unexplained identity change on reviewed pedestrian, bicycle, motorcycle, moving-car, stopped-car, and parked-car anchors;
- no long-lived track on roofs, signs, buildings, foliage, or empty space;
- real stationary objects are retained;
- every internal expiration is classified as exit, full occlusion, or failure where ground truth allows;
- the held-out split was not used for tuning.

If a class has too few labelled examples, report the sample count and confidence interval rather than claiming the per-class gate passed.

## Ground projection and kinematics

For a planar homography:

\[
s[X, Y, 1]^T = H[u, v, 1]^T
\]

where \((u,v)\) is the image contact point and \((X,Y)\) is a road-plane point in metres. For telemetry projection, form a camera ray from the calibrated camera model and intersect it with the ground plane. In either method, validate projection error using control points not used to fit the transform.

For uniformly sampled, smoothed ground positions \(\mathbf{p}_t\):

\[
\mathbf{v}_t \approx \frac{\mathbf{p}_{t+1}-\mathbf{p}_{t-1}}{2\Delta t}, \qquad
\mathbf{a}_t \approx \frac{\mathbf{p}_{t+1}-2\mathbf{p}_t+\mathbf{p}_{t-1}}{\Delta t^2}
\]

Do not differentiate long predicted gaps. Record interpolation and reject spikes caused by ID switches or box jitter.

Level-2 metric gates:

- validated spatial projection provenance;
- held-out ground-control reprojection error below the configured threshold;
- at least 80% reliable projected samples for a reported track;
- at least 70% observed samples and two seconds of duration;
- plausible speed/acceleration bounds by mode;
- comparison against at least one measured distance/time reference before product claims.

## Aggregate traffic measures

For a count \(N\) crossing a line during interval \(\Delta t\):

\[
q = \frac{N}{\Delta t}
\]

For \(N_L\) road users on a lane segment of length \(L\):

\[
k = \frac{N_L}{L}
\]

Under consistent definitions, the fundamental relationship is approximately:

\[
q = k\,\bar{v}
\]

Always state the interval, lane/link geometry, class filter, spatial coverage, and trajectory quality coverage. Aggregate outputs must link back to contributing track IDs.

## Network-reasoning evidence gate

A Level-5 explanation must contain:

1. measured events with timestamps and map locations;
2. the rule/model that connects events;
3. supporting track and signal/queue evidence;
4. alternative explanations considered; and
5. uncertainty or insufficient-evidence status.

Natural-language fluency is not an accuracy metric.
