from __future__ import annotations

import pandas as pd
import pytest

import numpy as np

from traffic_agent.renderer_v4 import (
    RendererV4Config,
    _class_lookup,
    _draw_compact_id,
    _normalize_tracks,
)


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


def test_renderer_rejects_zero_label_limit() -> None:
    with pytest.raises(ValueError, match="max_labels_per_frame"):
        RendererV4Config(max_labels_per_frame=0).validate()


def test_compact_id_is_drawn_when_full_label_is_not_available() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    occupied = [(8, 34, 150, 75)]

    drawn = _draw_compact_id(
        frame,
        track_id=27,
        bounds=(20, 40, 45, 65),
        colour=(255, 170, 50),
        occupied=occupied,
        scale=0.34,
    )

    assert drawn is True
    assert int(frame.sum()) > 0
    assert len(occupied) == 2


