from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_CONTEXT_FILES = (
    "AGENTS.md",
    "CODEX.md",
    "docs/context/README.md",
    "docs/context/challenge_brief.md",
    "docs/context/dataset_registry.md",
    "docs/context/decision_log.md",
    "docs/context/experiment_log.md",
    "docs/context/metrics_and_gates.md",
    "docs/context/current_state.json",
)


def load_and_validate_state(repo_root: Path) -> dict[str, object]:
    missing = [path for path in REQUIRED_CONTEXT_FILES if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"Missing context files: {', '.join(missing)}")

    state_path = repo_root / "docs/context/current_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported context schema_version")

    levels = state.get("levels")
    if not isinstance(levels, dict) or set(levels) != {"1", "2", "3", "4", "5"}:
        raise ValueError("Context must define exactly Levels 1 through 5")

    ground_truth = state.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError("ground_truth must be an object")
    level_1 = levels["1"]
    if not isinstance(level_1, dict):
        raise ValueError("Level 1 state must be an object")
    level_1_status = str(level_1.get("status", "")).lower()
    claims_passed = "passed" in level_1_status or "verified" in level_1_status
    required_metrics = ("precision", "recall", "idf1", "hota")
    metrics_present = all(ground_truth.get(metric) is not None for metric in required_metrics)
    if claims_passed and (ground_truth.get("status") != "passed" or not metrics_present):
        raise ValueError(
            "Level 1 cannot be marked passed/verified without passing ground truth and "
            "precision, recall, IDF1, and HOTA"
        )
    return state


def check_local_assets(state: dict[str, object]) -> list[str]:
    dataset = state.get("dataset")
    if not isinstance(dataset, dict):
        return ["dataset is not an object"]
    root_value = dataset.get("local_root")
    if not isinstance(root_value, str) or not root_value:
        return ["dataset.local_root is missing"]
    root = Path(root_value)
    expected = (
        "Intersection_Merged_convert_4k.mp4",
        "Intersection_1080p.srt",
        "Multi_Road_Merged_convert_4k.mp4",
        "Multi_Road_1080p.srt",
    )
    return [str(root / name) for name in expected if not (root / name).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and print FlytBase project context")
    parser.add_argument("--check-local-assets", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    state = load_and_validate_state(repo_root)
    missing_assets = check_local_assets(state) if args.check_local_assets else []
    if missing_assets:
        raise SystemExit("Missing local dataset assets:\n- " + "\n- ".join(missing_assets))

    levels = state["levels"]
    print(f"Context date: {state['as_of']}")
    repository = state["repository"]
    print(
        "Implementation baseline: "
        f"{repository['branch']} @ {repository['implementation_baseline_commit'][:12]}"
    )
    for number in sorted(levels, key=int):
        level = levels[number]
        print(f"L{number} {level['name']}: {level['status']}")
    print(f"Next: {state['next_action']['id']} - {state['next_action']['description']}")
    if args.check_local_assets:
        print("Local dataset assets: present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
