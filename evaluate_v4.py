from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_agent.evaluation_v4 import QualityGates, evaluate_v4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a V4 run against a manually corrected CVAT MOT export."
    )
    parser.add_argument("--mot-ground-truth", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--track-summary", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-detection-precision", type=float, default=0.90)
    parser.add_argument("--min-detection-recall", type=float, default=0.90)
    parser.add_argument("--min-idf1", type=float, default=0.85)
    parser.add_argument("--min-hota", type=float, default=0.70)
    parser.add_argument("--min-mode-accuracy", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_v4(
        mot_archive=args.mot_ground_truth,
        tracks_path=args.tracks,
        summary_path=args.track_summary,
        run_manifest_path=args.run_manifest,
        output_path=args.output,
        gates=QualityGates(
            detection_precision=args.min_detection_precision,
            detection_recall=args.min_detection_recall,
            idf1=args.min_idf1,
            hota=args.min_hota,
            mode_accuracy=args.min_mode_accuracy,
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
