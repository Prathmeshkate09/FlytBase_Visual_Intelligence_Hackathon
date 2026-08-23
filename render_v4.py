from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_agent.renderer_v4 import RendererV4Config, render_tracking_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render complete V4 tracking verification evidence."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--track-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--trail-length", type=int, default=20)
    parser.add_argument(
        "--max-labels-per-frame",
        type=int,
        default=None,
        help="Limit labels only; all boxes are still rendered.",
    )
    parser.add_argument(
        "--hide-occluded-labels",
        action="store_true",
        help="Keep dashed occlusion boxes but omit their labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = render_tracking_video(
        video_path=args.video,
        tracks_path=args.tracks,
        summary_path=args.track_summary,
        output_path=args.output,
        report_path=args.report,
        config=RendererV4Config(
            max_width=args.max_width,
            trail_length=args.trail_length,
            max_labels_per_frame=args.max_labels_per_frame,
            label_occluded=not args.hide_occluded_labels,
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


