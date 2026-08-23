# Five-level challenge brief

Source inspected on 2026-08-24: [FlytBase Visual Intelligence Hackathon](https://fbhackathonplatform-production.up.railway.app/), participant quest page.

The platform landing page currently says “three levels,” but the participant quest page displays five levels. The five-level participant brief is the project target. The hackathon has ended, so this repository now treats the brief as a personal-project product specification.

## Level 1 — Detection & Tracking

Detect and track every road user. Basic modes shown by the platform are car, LGV, HGV, bus, trucks, motorcycle, and pedestrian. Hold stable identity through occlusion, crossing paths, and long dwell times.

Project extension: bicycle is included as a road user even though it is absent from the displayed Level-1 list, because the supplied scene and aerial class taxonomy contain bicycles.

Required product output:

- one persistent identity per physical road user from first valid visibility to exit;
- observed/predicted state separation;
- class with uncertainty;
- track lifecycle events and reproducible evidence;
- corrected COCO/MOT evaluation.

## Level 2 — Object-Level Insight

Fine-grained vehicle classification and per-object velocity and acceleration in real units.

Required product output:

- stable class and attribute confidence per Level-1 identity;
- ground-plane position in metres;
- smoothed velocity and acceleration with reliability flags;
- no physical-unit claim outside a validated projection.

The spatial projection component is shared with Level 4 and should be implemented early rather than inventing pixel-to-km/h scaling.

## Level 3 — Aggregate Insight

Summarise across road users, time windows, or road regions. The platform examples are:

1. classified counts by movement and interval, including turning movements, directions, and lane volumes;
2. origin-destination distributions;
3. speed profiles and speeding concentration;
4. lane volume and modal split;
5. queue lengths; and
6. density, occupancy, and flow-density relationships.

Required product output should include temporal windows, spatial provenance, confidence/coverage, and drill-down to the contributing Level-1 identities.

## Level 4 — Spatial Grounding

The footage carries SRT telemetry: GPS position, altitude, gimbal orientation, and per-frame timestamps. Recover the ground-plane projection, then bind every trajectory to the road network moment by moment: correct link, approach, direction, and lane.

The platform expects map-native outputs such as:

- per-lane metrics on real geometry;
- desire lines over the road layout; and
- queue extents drawn along the carriageway.

Engineering interpretation:

1. align every merged-video frame to the correct SRT block;
2. calibrate camera intrinsics/extrinsics and validate ground error;
3. project bottom-centre contact points to local metric coordinates;
4. transform to geographic coordinates;
5. map-match trajectories to road links, approaches, directions, and lanes;
6. retain off-network and uncertain states instead of forcing a lane assignment.

## Level 5 — Network Reasoning

Reason across space and time and explain the conclusion. The complete visible platform brief includes:

- **Congestion origination:** trace a visible jam back through space and time to where it actually started.
- **Signal performance:** starting and discharge headways, saturation flow, green utilisation, cycle failure rate, arrival-on-green, and spillback across adjacent links.
- **Weaving, merging, and gap acceptance:** lane changes per kilometre, merge behaviour, and conflict concentration.
- **Desire-line analysis:** compare observed movement with intended geometry, including corner-cutting, lane straddling, informal paths, and mis-sited crossings.
- **Obstruction census:** double parking, bus-stop blocking, bike-lane obstruction, and loading-zone abuse with dwell duration.

Required product behavior:

- produce a time-ordered evidence chain linked to tracks, road geometry, and computed events;
- separate measurement, inference, and explanation;
- expose uncertainty and alternative explanations;
- let a reviewer reproduce the conclusion from the underlying events.

## Dependency rule

No later level may silently repair or hide Level-1 identity failures. Every aggregate, map event, and explanation must link back to immutable source run IDs and trajectory hashes.
