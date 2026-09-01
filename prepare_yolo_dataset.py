from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_agent.dataset_v4 import prepare_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a hash-verified YOLO dataset from corrected CVAT COCO labels."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--coco-annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-video-sha256", required=True)
    parser.add_argument("--expected-frames", type=int, default=89)
    parser.add_argument("--validation-start-frame", type=int, default=71)
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Put all frames in train and reuse that path as val for a fixed-epoch refit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_yolo_dataset(
        video_path=args.video,
        coco_path=args.coco_annotations,
        output_dir=args.output,
        expected_video_sha256=args.expected_video_sha256,
        expected_frames=args.expected_frames,
        validation_start_frame=args.validation_start_frame,
        all_train=args.all_train,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
