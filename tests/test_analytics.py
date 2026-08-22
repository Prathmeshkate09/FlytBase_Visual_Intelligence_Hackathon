import pandas as pd

from traffic_agent.analytics import calculate_object_metrics, clean_trajectories


def test_clean_and_measure_single_track() -> None:
    raw = pd.DataFrame(
        [
            {
                "frame": frame,
                "timestamp_s": frame / 10,
                "track_id": 7,
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.9,
                "x1": frame,
                "y1": 5,
                "x2": frame + 10,
                "y2": 15,
                "center_x": frame + 5,
                "center_y": 10,
                "ground_x": frame + 5,
                "ground_y": 15,
            }
            for frame in range(10)
        ]
    )
    cleaned = clean_trajectories(raw, fps=10, min_track_seconds=0.2)
    objects, motion = calculate_object_metrics(cleaned, fps=10)
    assert len(objects) == 1
    assert objects.iloc[0]["class_name"] == "car"
    assert objects.iloc[0]["duration_s"] == 1.0
    assert motion["speed_px_s"].max() > 0

