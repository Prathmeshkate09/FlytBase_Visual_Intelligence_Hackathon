from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_agent.pipeline_v4 import PipelineV4Config, run_pipeline_v4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V4 aerial detector + grouped BoT-SORT road-user pipeline."
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
    parser.add_argument(
        "--cached-detections",
        type=Path,
        default=None,
        help="Replay a prior detections.csv without running detector inference.",
    )
    parser.add_argument(
        "--cached-rejected-detections",
        type=Path,
        default=None,
        help=(
            "Reconsider only outside_road_user_roi rows from a prior "
            "rejected_detections.csv using the current ROI."
        ),
    )
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--confirmation-observations", type=int, default=3)
    parser.add_argument(
        "--max-prediction-frames",
        type=int,
        default=15,
        help="Write Kalman-only boxes for at most this many frames while retaining the internal track ID.",
    )
    parser.add_argument(
        "--disable-offline-stitching",
        action="store_true",
        help="Keep native BoT-SORT IDs in the final trajectory files.",
    )
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
            cached_detections=args.cached_detections,
            cached_rejected_detections=args.cached_rejected_detections,
            max_seconds=args.max_seconds,
            confirmation_observations=args.confirmation_observations,
            max_prediction_frames=args.max_prediction_frames,
            enable_offline_stitching=not args.disable_offline_stitching,
        )
    )
    print(json.dumps(result["quality_report"], indent=2))


if __name__ == "__main__":
    main()

