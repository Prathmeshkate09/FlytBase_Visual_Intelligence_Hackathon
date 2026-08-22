from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_agent.audit import AuditThresholds, write_tracking_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit tracking output for duplicate IDs, fragmentation, class instability and hidden evidence."
    )
    parser.add_argument("--raw-tracks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, help="run_metadata.json containing FPS and frame dimensions")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--frame-width", type=int)
    parser.add_argument("--frame-height", type=int)
    parser.add_argument("--duplicate-iou", type=float, default=0.90)
    parser.add_argument("--persistent-duplicate-frames", type=int, default=5)
    parser.add_argument("--boundary-margin-px", type=float, default=50.0)
    parser.add_argument("--handoff-max-gap-frames", type=int, default=10)
    parser.add_argument("--display-confidence", type=float, default=0.20)
    parser.add_argument("--stable-track-seconds", type=float, default=1.0)
    return parser.parse_args()


def _video_properties(args: argparse.Namespace) -> tuple[float, int, int]:
    metadata: dict[str, object] = {}
    if args.metadata is not None:
        if not args.metadata.exists():
            raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    fps = args.fps if args.fps is not None else metadata.get("source_fps")
    width = args.frame_width if args.frame_width is not None else metadata.get("frame_width")
    height = args.frame_height if args.frame_height is not None else metadata.get("frame_height")
    if fps is None or width is None or height is None:
        raise ValueError("Provide --metadata or explicit --fps, --frame-width and --frame-height")
    return float(fps), int(width), int(height)


def main() -> None:
    args = parse_args()
    fps, width, height = _video_properties(args)
    thresholds = AuditThresholds(
        duplicate_iou=args.duplicate_iou,
        persistent_duplicate_frames=args.persistent_duplicate_frames,
        boundary_margin_px=args.boundary_margin_px,
        handoff_max_gap_frames=args.handoff_max_gap_frames,
        display_confidence=args.display_confidence,
        stable_track_seconds=args.stable_track_seconds,
    )
    report = write_tracking_audit(
        args.raw_tracks,
        args.output,
        fps=fps,
        frame_width=width,
        frame_height=height,
        thresholds=thresholds,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

