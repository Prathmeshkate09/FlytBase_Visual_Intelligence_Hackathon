from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_agent.cvat import VideoMetadata, write_cvat_seed_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CVAT video-annotation seed package from V4 tracks."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--track-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--clip-role", choices=("development", "held-out"), required=True
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the CVAT task to frames 0..N-1 of the source clip.",
    )
    return parser.parse_args()


def video_metadata(path: Path, *, max_frames: int | None = None) -> VideoMetadata:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {path}")
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None and max_frames <= 0:
        capture.release()
        raise ValueError("max_frames must be positive")
    metadata = VideoMetadata(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=(
            min(source_frame_count, max_frames)
            if max_frames is not None
            else source_frame_count
        ),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
    )
    capture.release()
    metadata.validate()
    return metadata


def main() -> None:
    args = parse_args()
    metadata = video_metadata(args.video, max_frames=args.max_frames)
    report = write_cvat_seed_package(
        video_path=args.video,
        tracks_path=args.tracks,
        summary_path=args.track_summary,
        output_dir=args.output,
        metadata=metadata,
        task_name=args.task_name,
        clip_role=args.clip_role,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
