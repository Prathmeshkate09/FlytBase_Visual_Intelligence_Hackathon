from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="FlytBase Traffic Analysis Agent", layout="wide")
st.title("FlytBase Traffic Analysis Agent")
st.caption("Drone video → persistent trajectories → measurable traffic evidence")

results_value = st.sidebar.text_input("Results directory", value="outputs/level1")
results_dir = Path(results_value)

summary_path = results_dir / "summary.json"
if not summary_path.exists():
    st.info("Run the pipeline, then point this dashboard to its output directory.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
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

metrics_path = results_dir / "object_metrics.csv"
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    st.subheader("Object-level evidence")
    st.dataframe(metrics.sort_values("duration_s", ascending=False).head(100), use_container_width=True)

st.warning("Current speed values are in pixels/second. Do not claim km/h until ground-plane calibration is added.")

