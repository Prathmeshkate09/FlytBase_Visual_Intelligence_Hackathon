from __future__ import annotations

import copy
import importlib.util
import json
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

    assert state["levels"]["1"]["status"] == "development_baseline_failed_gates"
    assert state["ground_truth"]["status"] == "corrected_development_exports_validated"
    assert state["ground_truth"]["development_frames"] == 89
    assert state["ground_truth"]["cvat_frame_range"] == "0-88"
    assert state["next_action"]["id"] == "E008"


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


def test_cvat_label_schema_matches_level_1_taxonomy() -> None:
    labels = json.loads(
        (REPO_ROOT / "config/cvat_labels_level1.json").read_text(encoding="utf-8")
    )

    assert [label["name"] for label in labels] == [
        "car",
        "lgv",
        "hgv",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "ignore",
    ]
    assert all(label["type"] == "any" for label in labels)
    assert all(
        {attribute["name"] for attribute in label["attributes"]}
        == {"v4_track_id", "seed_state", "review_status"}
        for label in labels
    )
    assert all(
        next(
            attribute
            for attribute in label["attributes"]
            if attribute["name"] == "v4_track_id"
        )["values"]
        == [""]
        for label in labels
    )
