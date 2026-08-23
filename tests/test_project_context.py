from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "context_status", REPO_ROOT / "tools/context_status.py"
)
assert SPEC is not None and SPEC.loader is not None
context_status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(context_status)


def test_project_context_is_complete_and_unverified() -> None:
    state = context_status.load_and_validate_state(REPO_ROOT)

    assert state["levels"]["1"]["status"] == "candidate_needs_ground_truth"
    assert state["ground_truth"]["status"] == "missing_corrected_coco_and_mot"
    assert state["next_action"]["id"] == "E007"


def test_context_rejects_false_verified_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    original_loads = context_status.json.loads
    original_text = (REPO_ROOT / "docs/context/current_state.json").read_text(encoding="utf-8")
    invalid = original_loads(original_text)
    invalid = copy.deepcopy(invalid)
    invalid["levels"]["1"]["status"] = "verified"

    monkeypatch.setattr(context_status.json, "loads", lambda _text: invalid)

    with pytest.raises(ValueError, match="cannot be marked passed/verified"):
        context_status.load_and_validate_state(REPO_ROOT)


def test_registered_local_dataset_assets_exist() -> None:
    state = context_status.load_and_validate_state(REPO_ROOT)

    assert context_status.check_local_assets(state) == []
