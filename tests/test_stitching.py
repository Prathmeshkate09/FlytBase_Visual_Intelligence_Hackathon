from __future__ import annotations

import pandas as pd

from traffic_agent.stitching import StitchingConfig, stitch_tracks


def _row(
    frame: int,
    track_id: int,
    x: float,
    *,
    observed: bool = True,
    group: str = "road_vehicle",
) -> dict[str, object]:
    return {
        "frame": frame,
        "track_id": track_id,
        "association_group": group,
        "x1": x,
        "y1": 10.0,
        "x2": x + 20.0,
        "y2": 30.0,
        "observed": observed,
        "confidence": 0.8 if observed else None,
    }


def test_stitches_short_unambiguous_linear_track_break() -> None:
    rows = [_row(frame, 10, float(frame * 4)) for frame in range(5)]
    rows += [_row(frame, 20, float(frame * 4)) for frame in range(7, 12)]

    stitched, decisions = stitch_tracks(pd.DataFrame(rows))

    assert decisions[["old_track_id", "new_track_id", "canonical_track_id"]].to_dict("records") == [
        {"old_track_id": 10, "new_track_id": 20, "canonical_track_id": 10}
    ]
    assert set(stitched["track_id"]) == {10}
    assert set(stitched["source_track_id"]) == {10, 20}


def test_does_not_stitch_different_association_groups() -> None:
    rows = [_row(frame, 10, float(frame * 4)) for frame in range(5)]
    rows += [
        _row(frame, 20, float(frame * 4), group="pedestrian") for frame in range(7, 12)
    ]

    stitched, decisions = stitch_tracks(pd.DataFrame(rows))

    assert decisions.empty
    assert set(stitched["track_id"]) == {10, 20}


def test_rejects_ambiguous_successor() -> None:
    rows = [_row(frame, 10, float(frame * 4)) for frame in range(5)]
    rows += [_row(frame, 20, float(frame * 4)) for frame in range(7, 12)]
    rows += [_row(frame, 30, float(frame * 4 + 1)) for frame in range(7, 12)]

    stitched, decisions = stitch_tracks(pd.DataFrame(rows))

    assert decisions.empty
    assert set(stitched["track_id"]) == {10, 20, 30}


def test_observed_row_replaces_prediction_after_stitch() -> None:
    rows = [_row(frame, 10, float(frame * 4)) for frame in range(5)]
    rows += [_row(frame, 10, float(frame * 4), observed=False) for frame in range(5, 9)]
    rows += [_row(frame, 20, float(frame * 4)) for frame in range(7, 12)]

    stitched, decisions = stitch_tracks(pd.DataFrame(rows))

    assert len(decisions) == 1
    frame_seven = stitched[(stitched["frame"] == 7) & (stitched["track_id"] == 10)]
    assert len(frame_seven) == 1
    assert bool(frame_seven.iloc[0]["observed"]) is True
    assert int(frame_seven.iloc[0]["source_track_id"]) == 20


def test_rejects_large_direction_change() -> None:
    rows = [_row(frame, 10, float(frame * 4)) for frame in range(5)]
    rows += [_row(frame, 20, float(40 - frame * 4)) for frame in range(7, 12)]

    stitched, decisions = stitch_tracks(
        pd.DataFrame(rows), StitchingConfig(maximum_direction_change_degrees=45.0)
    )

    assert decisions.empty
    assert set(stitched["track_id"]) == {10, 20}

