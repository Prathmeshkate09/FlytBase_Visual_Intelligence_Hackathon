from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from traffic_agent.evaluation_v4 import detection_cache_metrics, read_cvat_mot_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-tracker detection cache against corrected MOT boxes."
    )
    parser.add_argument("--mot-ground-truth", required=True, type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-count", type=int, default=89)
    parser.add_argument("--frame-width", type=int, default=3840)
    parser.add_argument("--frame-height", type=int, default=2160)
    parser.add_argument(
        "--confidence-thresholds",
        default="0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = [float(value) for value in args.confidence_thresholds.split(",")]
    ground_truth = read_cvat_mot_archive(args.mot_ground_truth)
    detections = pd.read_csv(args.detections)
    reports = [
        detection_cache_metrics(
            ground_truth,
            detections,
            frame_count=args.frame_count,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            minimum_confidence=threshold,
        )
        for threshold in thresholds
    ]
    result = {
        "schema_version": 1,
        "mot_ground_truth": str(args.mot_ground_truth),
        "detections": str(args.detections),
        "threshold_results": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
