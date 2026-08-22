from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="FlytBase Traffic Analysis Agent", layout="wide")
st.title("FlytBase Traffic Analysis Agent")
st.caption("Drone video -> persistent trajectories -> auditable object insight")

results_value = st.sidebar.text_input("Results directory", value="outputs/level1")
results_dir = Path(results_value)
level1_summary_path = results_dir / "summary.json"
level2_summary_path = results_dir / "level2_summary.json"

if not level1_summary_path.exists() and not level2_summary_path.exists():
    st.info("Run a pipeline, then point this dashboard to its output directory.")
    st.stop()

if level1_summary_path.exists():
    st.header("Level 1 - Detection and Tracking")
    summary = json.loads(level1_summary_path.read_text(encoding="utf-8"))
    quality = summary.get("tracking_quality_proxies", {})
    columns = st.columns(4)
    columns[0].metric("Unique tracks", quality.get("unique_tracks", 0))
    columns[1].metric("Median duration", f"{quality.get('median_track_duration_s', 0):.2f} s")
    columns[2].metric("Long-track ratio", f"{100 * quality.get('long_track_ratio', 0):.1f}%")
    columns[3].metric("Track continuity", f"{100 * quality.get('median_observed_ratio', 0):.1f}%")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Road users by class")
        class_counts = pd.Series(summary.get("unique_tracks_by_class", {}), name="tracks")
        if not class_counts.empty:
            st.bar_chart(class_counts)
    with right:
        st.subheader("Trajectory density")
        heatmap = results_dir / "density_heatmap.png"
        if heatmap.exists():
            st.image(str(heatmap), use_container_width=True)

    video = results_dir / "annotated_tracking.mp4"
    if video.exists():
        st.subheader("Detection, IDs, and trajectory evidence")
        st.video(str(video))

if level2_summary_path.exists():
    st.header("Level 2 - Object-Level Insight")
    summary = json.loads(level2_summary_path.read_text(encoding="utf-8"))
    appearance = summary.get("appearance", {})
    kinematics = summary.get("kinematics", {})
    columns = st.columns(3)
    columns[0].metric("Analysed tracks", summary.get("analysed_tracks", 0))
    columns[1].metric("Metric-reliable tracks", kinematics.get("metric_reliable_objects", 0))
    columns[2].metric("Metric status", kinematics.get("metric_status", "unavailable"))

    left, right = st.columns(2)
    with left:
        st.subheader("Coarse colour")
        color_counts = pd.Series(appearance.get("colour_distribution", {}), name="tracks")
        if not color_counts.empty:
            st.bar_chart(color_counts)
    with right:
        st.subheader("Relative size class")
        size_counts = pd.Series(appearance.get("relative_size_distribution", {}), name="tracks")
        if not size_counts.empty:
            st.bar_chart(size_counts)

    speed_plot = results_dir / "level2_speed_profiles.png"
    if speed_plot.exists():
        st.image(str(speed_plot), use_container_width=True)
    level2_video = results_dir / "level2_evidence.mp4"
    if level2_video.exists():
        st.video(str(level2_video))
    object_insights = results_dir / "level2_object_insights.csv"
    if object_insights.exists():
        st.dataframe(pd.read_csv(object_insights), use_container_width=True)

    for limitation in summary.get("limitations", []):
        st.warning(limitation)
