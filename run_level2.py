from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_agent.level2 import Level2Config, run_level2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add appearance and calibration-aware per-object kinematics to Level 1 tracks."
    )
    parser.add_argument("--raw-tracks", required=True, type=Path, help="Level 1 raw_tracks.csv")
    parser.add_argument("--video", required=True, type=Path, help="The exact video used for Level 1")
    parser.add_argument("--output", required=True, type=Path, help="Level 2 output directory")
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Optional image-to-ground homography JSON. Without it, metric speed is not claimed.",
    )
    metric_group.add_argument(
        "--srt",
        type=Path,
        default=None,
        help="Frame-aligned DJI-style SRT telemetry for camera-derived ground projection.",
    )
    parser.add_argument(
        "--srt-frame-offset",
        type=int,
        default=0,
        help="Telemetry frame corresponding to source-video frame zero.",
    )
    parser.add_argument(
        "--srt-segment-index",
        type=int,
        default=None,
        help="Zero-based FrameCnt segment in a concatenated SRT; required when frame counts reset.",
    )
    parser.add_argument(
        "--srt-max-interpolation-gap-frames",
        type=int,
        default=2,
        help="Maximum consecutive missing SRT frames that may be linearly interpolated.",
    )
    parser.add_argument("--min-track-seconds", type=float, default=1.0)
    parser.add_argument("--max-gap-frames", type=int, default=10)
    parser.add_argument("--smoothing-window", type=int, default=31)
    parser.add_argument("--color-samples", type=int, default=7)
    parser.add_argument("--appearance-conf", type=float, default=0.20)
    parser.add_argument("--display-conf", type=float, default=0.25)
    parser.add_argument("--stop-speed-m-s", type=float, default=0.50)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Level2Config(
        raw_tracks=args.raw_tracks,
        video=args.video,
        output_dir=args.output,
        calibration=args.calibration,
        srt=args.srt,
        srt_frame_offset=args.srt_frame_offset,
        srt_segment_index=args.srt_segment_index,
        srt_max_interpolation_gap_frames=args.srt_max_interpolation_gap_frames,
        min_track_seconds=args.min_track_seconds,
        max_gap_frames=args.max_gap_frames,
        trajectory_smoothing_window=args.smoothing_window,
        max_color_samples=args.color_samples,
        appearance_confidence=args.appearance_conf,
        stop_speed_m_s=args.stop_speed_m_s,
        display_confidence=args.display_conf,
        save_video=not args.no_video,
    )
    print(json.dumps(run_level2(config), indent=2))


if __name__ == "__main__":
    main()
