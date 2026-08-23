# Project context system

This directory is the durable evidence layer for the FlytBase project.

- `current_state.json`: machine-readable compact state and next action.
- `challenge_brief.md`: exact five-level target captured from the platform.
- `dataset_registry.md`: local assets, metadata, telemetry structure, and hazards.
- `decision_log.md`: accepted, rejected, deferred, and provisional choices.
- `experiment_log.md`: append-only run history with evidence locations.
- `metrics_and_gates.md`: mathematical definitions and acceptance criteria.

`AGENTS.md` requires future coding sessions to read these files before making material changes. `CODEX.md` is the short human-readable summary.

## Evidence hierarchy

From strongest to weakest:

1. corrected held-out ground truth plus official evaluator output;
2. reproducible run manifest, hashes, metrics, and full anchor audit;
3. internal consistency audit;
4. visual review of the complete evidence video;
5. intuition or model reputation.

A weaker layer may suggest an experiment but cannot overrule stronger contradictory evidence.

## Recording a new experiment

Append an entry to `experiment_log.md` with:

- experiment ID and date;
- question/hypothesis;
- code commit and clean/dirty status;
- input clip and SHA-256;
- detector cache or weight hash;
- exact configuration;
- metrics and anchor results;
- artifact path;
- decision and limitations.

Update the decision log only when the experiment changes an engineering choice. Keep old entries; use `Superseded by Dxxx` rather than deleting history.
