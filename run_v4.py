from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_agent.pipeline_v4 import PipelineV4Config, run_pipeline_v4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V4 SAHI + grouped BoT-SORT road-user pipeline."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--detection-config", type=Path, default=Path("config/detection_v4.json")
    )
    parser.add_argument(
        "--tracker-config", type=Path, default=Path("config/botsort_drone_v4.yaml")
    )
    parser.add_argument("--road-user-roi", type=Path, default=None)
    parser.add_argument("--exit-roi", type=Path, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--confirmation-observations", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline_v4(
        PipelineV4Config(
            input_video=args.input,
            output_dir=args.output,
            detection_config=args.detection_config,
            tracker_config=args.tracker_config,
            road_user_roi=args.road_user_roi,
            exit_roi=args.exit_roi,
            max_seconds=args.max_seconds,
            confirmation_observations=args.confirmation_observations,
        )
    )
    print(json.dumps(result["quality_report"], indent=2))


if __name__ == "__main__":
    main()
