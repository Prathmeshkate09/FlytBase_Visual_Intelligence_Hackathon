import numpy as np
import pytest

from traffic_agent.telemetry import TelemetryGroundProjector, TelemetrySample, parse_dji_srt


def test_parse_dji_srt(tmp_path) -> None:
    srt = tmp_path / "sample.srt"
    srt.write_text(
        """1
00:00:00,000 --> 00:00:00,033
FrameCnt: 0 2026-08-21 17:38:52.463
[focal_len: 24.00], [latitude: 18.566227] [longitude: 73.771846] [rel_alt: 70.472 abs_alt: 607.273] [gb_yaw: -125.5 gb_pitch: -63.1 gb_roll: 0.0]

2
00:00:00,033 --> 00:00:00,066
FrameCnt: 1 2026-08-21 17:38:52.497
[focal_len: 24.00], [latitude: 18.566228] [longitude: 73.771847] [rel_alt: 70.473 abs_alt: 607.274] [gb_yaw: -125.4 gb_pitch: -63.0 gb_roll: 0.1]
""",
        encoding="utf-8",
    )
    samples = parse_dji_srt(srt)
    assert [sample.frame for sample in samples] == [0, 1]
    assert samples[0].relative_altitude_m == 70.472
    assert samples[1].gimbal_roll_deg == 0.1


def test_straight_down_projection_has_metric_scale() -> None:
    sample = TelemetrySample(
        frame=0,
        latitude=18.0,
        longitude=73.0,
        relative_altitude_m=100.0,
        gimbal_yaw_deg=0.0,
        gimbal_pitch_deg=-90.0,
        gimbal_roll_deg=0.0,
        focal_length_35mm=36.0,
    )
    projector = TelemetryGroundProjector([sample], frame_width=3600, frame_height=1800)
    center_x = (3600 - 1) / 2
    center_y = (1800 - 1) / 2
    points, reliable = projector.project(
        np.asarray([0, 0, 0]),
        np.asarray(
            [
                [center_x, center_y],
                [center_x + 360, center_y],
                [center_x, center_y + 360],
            ]
        ),
    )
    assert reliable.all()
    assert np.allclose(points[0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(points[1], [10.0, 0.0], atol=1e-6)
    assert np.allclose(points[2], [0.0, -10.0], atol=1e-6)


def test_concatenated_srt_requires_explicit_segment(tmp_path) -> None:
    block = """{index}
00:00:00,000 --> 00:00:00,033
FrameCnt: {frame} 2026-08-21 17:38:52.463
[focal_len: 24.00], [latitude: 18.566227] [longitude: 73.771846] [rel_alt: 70.472 abs_alt: 607.273] [gb_yaw: -125.5 gb_pitch: -63.1 gb_roll: 0.0]

"""
    srt = tmp_path / "merged.srt"
    srt.write_text(
        block.format(index=1, frame=0)
        + block.format(index=2, frame=1)
        + block.format(index=3, frame=0)
        + block.format(index=4, frame=1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="select one explicitly"):
        parse_dji_srt(srt)
    selected = parse_dji_srt(srt, segment_index=1)
    assert [sample.frame for sample in selected] == [0, 1]


def test_single_missing_telemetry_frame_is_bounded_and_reported() -> None:
    start = TelemetrySample(
        frame=0,
        latitude=18.0,
        longitude=73.0,
        relative_altitude_m=100.0,
        gimbal_yaw_deg=0.0,
        gimbal_pitch_deg=-90.0,
        gimbal_roll_deg=0.0,
        focal_length_35mm=36.0,
    )
    end = TelemetrySample(
        frame=2,
        latitude=18.0,
        longitude=73.0,
        relative_altitude_m=100.0,
        gimbal_yaw_deg=0.0,
        gimbal_pitch_deg=-90.0,
        gimbal_roll_deg=0.0,
        focal_length_35mm=36.0,
    )
    projector = TelemetryGroundProjector(
        [start, end],
        frame_width=3600,
        frame_height=1800,
        max_interpolation_gap_frames=1,
    )
    center = np.asarray([[(3600 - 1) / 2, (1800 - 1) / 2]])
    points, reliable = projector.project(np.asarray([1]), center)
    assert reliable.all()
    assert np.allclose(points[0], [0.0, 0.0], atol=1e-6)
    report = projector.to_report()
    assert report["interpolated_frame_count"] == 1
    assert report["interpolated_frames"] == [1]
