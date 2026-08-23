# Dataset registry

Inventory date: 2026-08-24

## Local root

```text
C:\Users\PRATHAMESH\Documents\Codex\2026-08-22\kn\work\compressed-20260823T124814Z-1-001\compressed
```

This directory is outside the Git repository and must remain untracked.

## Media assets

| Asset | Bytes | Resolution | FPS | Frames / entries | Duration |
|---|---:|---:|---:|---:|---:|
| `Intersection_Merged_convert_4k.mp4` | 191,712,536 | 3840 x 2160 | 29.97003 | 11,971 frames | 399.432 s |
| `Intersection_1080p.srt` | 3,937,113 | telemetry | nominal 30 | 11,971 blocks | 399.432 s |
| `Multi_Road_Merged_convert_4k.mp4` | 99,029,786 | 3840 x 2160 | 29.97003 | 9,140 frames | 304.971 s |
| `Multi_Road_1080p.srt` | 3,003,741 | telemetry | nominal 30 | 9,140 blocks | 304.971 s |

The local compressed videos match the durations advertised by the hackathon resource page: approximately 6:39 and 5:05. Their sizes are much smaller than the original 6.0 GB and 4.6 GB files, so compression artifacts must be considered when evaluating tiny-object recall.

## SRT fields

Each telemetry block contains:

- subtitle timestamp and `FrameCnt`;
- wall-clock timestamp;
- ISO, shutter, aperture, exposure and focal length;
- latitude and longitude;
- relative and absolute altitude; and
- gimbal yaw, pitch, and roll.

## Frame-count segmentation

The SRT block count equals the compressed MP4 frame count, but raw `FrameCnt` is not globally unique.

| SRT | Segment | Entries | Raw `FrameCnt` range | Missing raw-count gaps |
|---|---:|---:|---:|---:|
| Intersection | 0 | 6,801 | 0-6,830 | 30 |
| Intersection | 1 | 5,170 | 0-5,193 | 24 |
| Multi-road | 0 | 6,802 | 0-6,832 | 31 |
| Multi-road | 1 | 2,338 | 0-2,347 | 10 |

The raw counter resets at the second source segment and skips some numbers inside each segment. Therefore:

- The merged MP4 output frame index aligns to SRT **block order**, not to a unique raw `FrameCnt` over the entire file.
- `FrameCnt` is still useful within an explicitly selected source segment.
- A full-video telemetry loader needs `output_frame`, `subtitle_index`, `source_segment`, and `source_frame_count` as separate fields.
- The current `parse_dji_srt(..., segment_index=...)` behavior is safe only when the processed clip and its frame offset are explicitly mapped to that segment.
- Do not interpolate across a segment reset or across a source-video boundary.

This is a Level-4 correctness gate and also affects Level-2 metric velocity.

## Dataset split policy

The final split must be stored with frame ranges and hashes. Until corrected annotations exist, use this provisional policy:

- annotation pilot/development: 89 consecutive frames;
- configuration development: a labelled 15-second clip;
- held-out verification: a different labelled 15- or 30-second interval;
- robustness test: longer unseen intervals from both intersection and multi-road videos.

Do not train or tune on the held-out interval. Do not call unlabelled video “test data”; it is only an unlabelled robustness review.

## Missing dataset assets

- No corrected COCO detection ground truth is currently registered.
- No corrected MOT identity ground truth is currently registered.
- No surveyed ground control points or lane centreline geometry are registered.
- No verified camera distortion/intrinsic calibration is registered.

These missing assets, not video duration, are the main blockers for formal Levels 1, 2, and 4 validation.
