from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_agent.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract road-user tracks and Level-1 traffic evidence from drone video.")
    parser.add_argument("--input", required=True, type=Path, help="Input drone video")
    parser.add_argument("--output", required=True, type=Path, help="Directory for CSV, video, and report outputs")
    parser.add_argument("--model", default="yolo26l.pt", help="Ultralytics detection model")
    parser.add_argument("--tracker", default="config/botsort_drone.yaml", help="Ultralytics tracker YAML")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size; try 1600 for tiny vehicles")
    parser.add_argument("--conf", type=float, default=0.10, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="Detector NMS IoU threshold")
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument("--classes", default="0,1,2,3,5,7", help="COCO IDs: person,bicycle,car,motorcycle,bus,truck")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-seconds", type=float, default=None, help="Limit runtime for rapid experiments")
    parser.add_argument("--no-video", action="store_true", help="Skip annotated-video generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_ids = tuple(int(value.strip()) for value in args.classes.split(",") if value.strip())
    config = PipelineConfig(
        input_video=args.input,
        output_dir=args.output,
        model=args.model,
        tracker=args.tracker,
        imgsz=args.imgsz,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        classes=class_ids,
        frame_stride=args.stride,
        max_seconds=args.max_seconds,
        save_video=not args.no_video,
    )
    result = run_pipeline(config)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

