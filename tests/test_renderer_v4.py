from __future__ import annotations

import pandas as pd
import pytest

from traffic_agent.renderer_v4 import _class_lookup, _normalize_tracks


def _tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 1,
                "association_group": "road_vehicle",
                "x1": 10.0,
                "y1": 10.0,
                "x2": 30.0,
                "y2": 30.0,
                "observed": "True",
            },
            {
                "frame": 1,
                "track_id": 1,
                "association_group": "road_vehicle",
                "x1": 12.0,
                "y1": 10.0,
                "x2": 32.0,
                "y2": 30.0,
                "observed": "False",
            },
        ]
    )


def test_normalize_tracks_preserves_observed_and_occluded_rows() -> None:
    normalized = _normalize_tracks(_tracks())

    assert normalized["observed"].tolist() == [True, False]


def test_normalize_tracks_rejects_duplicate_frame_identity() -> None:
    duplicate = pd.concat([_tracks(), _tracks().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate frame/track_id"):
        _normalize_tracks(duplicate)


def test_class_lookup_requires_one_summary_per_final_id() -> None:
    summary = pd.DataFrame(
        [
            {"track_id": 1, "final_class_name": "car"},
            {"track_id": 1, "final_class_name": "truck"},
        ]
    )

    with pytest.raises(ValueError, match="duplicate track IDs"):
        _class_lookup(summary)

