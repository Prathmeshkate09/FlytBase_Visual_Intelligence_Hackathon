from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from traffic_agent.evaluation_v4 import evaluate_v4, sha256_file
from traffic_agent.pipeline_v4 import PipelineV4Config, run_pipeline_v4
from traffic_agent.tuning_v4 import (
    DETECTION_THRESHOLD_VARIANTS,
    TRACKER_VARIANTS,
    detector_config_variant,
    load_yaml_mapping,
    report_rank,
    tracker_config_variant,
    write_json_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune detector filtering and BoT-SORT from one immutable candidate cache."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-video-sha256", required=True)
    parser.add_argument("--candidate-detections", required=True, type=Path)
    parser.add_argument("--mot-ground-truth", required=True, type=Path)
    parser.add_argument("--detection-config", required=True, type=Path)
    parser.add_argument("--tracker-config", required=True, type=Path)
    parser.add_argument("--road-user-roi", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=3.0)
    return parser.parse_args()


def run_candidate(
    *,
    name: str,
    args: argparse.Namespace,
    detection_config: Path,
    tracker_config: Path,
    roi: Path | None,
    confirmation_observations: int,
    max_prediction_frames: int,
    enable_offline_stitching: bool,
) -> dict[str, Any]:
    output = args.output / "runs" / name
    report_path = output / "quality_report_ground_truth.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    run_pipeline_v4(
        PipelineV4Config(
            input_video=args.input,
            output_dir=output,
            detection_config=detection_config,
            tracker_config=tracker_config,
            road_user_roi=roi,
            cached_candidates=args.candidate_detections,
            max_seconds=args.max_seconds,
            confirmation_observations=confirmation_observations,
            max_prediction_frames=max_prediction_frames,
            enable_offline_stitching=enable_offline_stitching,
        )
    )
    return evaluate_v4(
        mot_archive=args.mot_ground_truth,
        tracks_path=output / "tracks.csv",
        summary_path=output / "track_summary.csv",
        run_manifest_path=output / "run_manifest.json",
        output_path=report_path,
        diagnostics_output_path=output / "error_diagnostics.json",
    )


def main() -> None:
    args = parse_args()
    if sha256_file(args.input).lower() != args.expected_video_sha256.lower():
        raise ValueError("Input video hash does not match --expected-video-sha256")
    for path in (
        args.candidate_detections,
        args.mot_ground_truth,
        args.detection_config,
        args.tracker_config,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    configs = args.output / "configs"
    base_detector = json.loads(args.detection_config.read_text(encoding="utf-8"))
    base_tracker = load_yaml_mapping(args.tracker_config)
    reports: dict[str, dict[str, Any]] = {}
    run_specs: dict[str, dict[str, Any]] = {}

    roi_options = {"roi": args.road_user_roi, "no_roi": None}
    for threshold_name, thresholds in DETECTION_THRESHOLD_VARIANTS.items():
        detector_path = configs / f"detector_{threshold_name}.json"
        write_json_yaml(
            detector_path, detector_config_variant(base_detector, thresholds)
        )
        for roi_name, roi in roi_options.items():
            name = f"detect_{threshold_name}_{roi_name}"
            reports[name] = run_candidate(
                name=name,
                args=args,
                detection_config=detector_path,
                tracker_config=args.tracker_config,
                roi=roi,
                confirmation_observations=3,
                max_prediction_frames=2,
                enable_offline_stitching=True,
            )
            run_specs[name] = {
                "detection_config": str(detector_path),
                "tracker_config": str(args.tracker_config),
                "roi": str(roi) if roi else None,
                "confirmation_observations": 3,
                "max_prediction_frames": 2,
                "enable_offline_stitching": True,
            }

    best_detection_name = max(reports, key=lambda name: report_rank(reports[name]))
    best_spec = run_specs[best_detection_name]
    best_detection_config = Path(best_spec["detection_config"])
    best_roi = Path(best_spec["roi"]) if best_spec["roi"] else None
    association_names: list[str] = []
    for variant_name, overrides in TRACKER_VARIANTS.items():
        tracker_path = configs / f"tracker_{variant_name}.yaml"
        write_json_yaml(
            tracker_path, tracker_config_variant(base_tracker, overrides)
        )
        name = f"associate_{variant_name}"
        reports[name] = run_candidate(
            name=name,
            args=args,
            detection_config=best_detection_config,
            tracker_config=tracker_path,
            roi=best_roi,
            confirmation_observations=3,
            max_prediction_frames=2,
            enable_offline_stitching=False,
        )
        run_specs[name] = {
            "detection_config": str(best_detection_config),
            "tracker_config": str(tracker_path),
            "roi": str(best_roi) if best_roi else None,
            "confirmation_observations": 3,
            "max_prediction_frames": 2,
            "enable_offline_stitching": False,
        }
        association_names.append(name)

    top_associations = sorted(
        association_names, key=lambda name: report_rank(reports[name]), reverse=True
    )[:2]
    for association_name in top_associations:
        association_spec = run_specs[association_name]
        for confirmation in (1, 3, 5):
            for prediction_frames in (0, 2):
                for stitching in (False, True):
                    name = (
                        f"lifecycle_{association_name.removeprefix('associate_')}_"
                        f"c{confirmation}_p{prediction_frames}_s{int(stitching)}"
                    )
                    reports[name] = run_candidate(
                        name=name,
                        args=args,
                        detection_config=best_detection_config,
                        tracker_config=Path(association_spec["tracker_config"]),
                        roi=best_roi,
                        confirmation_observations=confirmation,
                        max_prediction_frames=prediction_frames,
                        enable_offline_stitching=stitching,
                    )
                    run_specs[name] = {
                        **association_spec,
                        "confirmation_observations": confirmation,
                        "max_prediction_frames": prediction_frames,
                        "enable_offline_stitching": stitching,
                    }

    winner = max(reports, key=lambda name: report_rank(reports[name]))
    summary = {
        "schema_version": 1,
        "input_video_sha256": sha256_file(args.input),
        "candidate_detections_sha256": sha256_file(args.candidate_detections),
        "ground_truth_sha256": sha256_file(args.mot_ground_truth),
        "best_detection_stage": best_detection_name,
        "winner": winner,
        "winner_spec": run_specs[winner],
        "winner_report": reports[winner],
        "ranked_runs": [
            {
                "name": name,
                "quality_status": reports[name]["quality_status"],
                **reports[name]["ground_truth_metrics"],
            }
            for name in sorted(reports, key=lambda item: report_rank(reports[item]), reverse=True)
        ],
    }
    (args.output / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
