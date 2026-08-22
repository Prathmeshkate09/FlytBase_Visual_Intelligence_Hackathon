from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a coordinate-labelled video frame for homography setup.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0, help="Zero-based source frame")
    parser.add_argument("--grid-step", type=int, default=100, help="Grid spacing in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame < 0:
        raise ValueError("--frame must be zero or greater")
    if args.grid_step < 20:
        raise ValueError("--grid-step must be at least 20 pixels")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        success, image = capture.read()
    finally:
        capture.release()
    if not success:
        raise RuntimeError(f"Could not read frame {args.frame}")

    height, width = image.shape[:2]
    for x in range(0, width, args.grid_step):
        cv2.line(image, (x, 0), (x, height - 1), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(x), (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    for y in range(0, height, args.grid_step):
        cv2.line(image, (0, y), (width - 1, y), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(y), (3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise RuntimeError(f"Could not write calibration frame: {args.output}")
    print(f"Created {args.output} at {width}x{height} from frame {args.frame}")


if __name__ == "__main__":
    main()
