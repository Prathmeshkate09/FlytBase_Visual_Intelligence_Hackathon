# FlytBase project instructions

These instructions apply to the entire repository. They are the operational entry point for future Codex sessions.

## Required context read

Before changing code, configuration, experiment data, or project claims, read these files in order:

1. `CODEX.md`
2. `docs/context/current_state.json`
3. `docs/context/decision_log.md`
4. `docs/context/experiment_log.md`
5. `docs/context/metrics_and_gates.md`
6. `docs/context/dataset_registry.md`
7. `docs/context/challenge_brief.md`

Run `python tools/context_status.py` to print the compact state. Add `--check-local-assets` when working on the machine that contains the dataset.

## Evidence rules

- Inspect the real Git root, status, active branch, relevant code, configuration, and result artifacts before editing.
- Preserve unrelated user changes. In particular, do not overwrite untracked configuration files without first inspecting them.
- Diagnose failures by layer: detection, post-processing/ROI, association, lifecycle/stitching, rendering, or spatial calibration.
- Never call Level 1 `passed`, `verified`, or `submission-ready` without corrected ground truth and reported detection precision/recall plus tracking IDF1/HOTA.
- Internal proxy statistics, a clean video, track duration, low duplicate count, or a manual spot check are supporting evidence, not formal accuracy.
- Record observed detections separately from predicted/occluded states. Never render old Kalman predictions as real observations.
- Track real stationary and parked road users. Motion is not a validity filter.
- Do not report speed in m/s or km/h unless the image-to-ground projection is validated for those samples.
- Do not infer fine-grained classes such as LGV versus HGV when the pixels or labels do not support the claim; use an explicit unknown class.
- Keep training, development, and held-out clips separate. Never tune on the held-out labels and then report them as an unbiased test.
- Compare trackers on identical cached detections and identical frames. Compare detectors with the same ground-truth split and thresholds.
- Every material experiment must record the code commit, input hash, configuration, output location, metrics, anchor audit, conclusion, and limitations in `docs/context/experiment_log.md`.
- Every accepted or rejected architectural choice must update `docs/context/decision_log.md` and `docs/context/current_state.json` in the same change.

## Current product boundary

- Level 1 is the identity foundation. Levels 2-5 must not silently consume an unverified trajectory set.
- Level 2 may use the Level-4 projection component early because real-unit kinematics require spatial calibration.
- Level 3 aggregates only verified object trajectories and reliable spatial measurements.
- Level 4 must align the merged video frame sequence with the correct SRT block and road geometry.
- Level 5 must produce an evidence chain and uncertainty, not only a natural-language explanation.

## Repository and data hygiene

- Do not commit source videos, generated videos, CSV results, model weights, virtual environments, credentials, or private dataset copies.
- Store portable source, tests, configuration templates, and context documents in Git. Store large run artifacts outside the repository and retain their SHA-256 values in the experiment log.
- Treat the absolute dataset path in `dataset_registry.md` and `current_state.json` as local-machine configuration, not a portable public URL.
- Use small, scoped code changes that follow existing patterns. Add or update tests for any changed behavior.

## Verification commands

```powershell
python tools/context_status.py --check-local-assets
python -m pytest -q
git status --short
```

If a command cannot be run, state that limit explicitly. Explain results in plain language because the project owner is learning computer vision.
