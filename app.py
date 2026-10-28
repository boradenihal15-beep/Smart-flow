"""
SmartFlow: AI Traffic Brain - prototype UI
Run:  streamlit run app.py
"""
import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from smartflow_core import (
    DEFAULT_GREEN_TIMES,
    FIXED_BASELINE_GREEN,
    draw_roi,
    process_video,
    resize_keep_aspect,
    roi_from_fractions,
    save_log_csv,
    summarize,
)

st.set_page_config(page_title="SmartFlow: AI Traffic Brain", layout="wide")
st.title("SmartFlow: AI Traffic Brain")
st.caption("Upload a traffic video of ONE approach to a signal. The system detects vehicles waiting "
           "in the queue zone, classifies density and recommends the green-signal time.")

# ---------------- sidebar settings ----------------
st.sidebar.header("Settings")
conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
waiting_only = st.sidebar.checkbox("Count only waiting (stationary) vehicles", True)
window_s = st.sidebar.slider("Smoothing window (seconds)", 1, 15, 5)

st.sidebar.subheader("Density thresholds (vehicles)")
low_max = st.sidebar.number_input("LOW up to", 1, 50, 5)
med_max = st.sidebar.number_input("MEDIUM up to", int(low_max) + 1, 100, max(12, int(low_max) + 1))

st.sidebar.subheader("Green time (seconds)")
g_low = st.sidebar.number_input("LOW", 5, 120, DEFAULT_GREEN_TIMES["LOW"])
g_med = st.sidebar.number_input("MEDIUM", 5, 120, DEFAULT_GREEN_TIMES["MEDIUM"])
g_high = st.sidebar.number_input("HIGH", 5, 120, DEFAULT_GREEN_TIMES["HIGH"])
green_times = {"LOW": g_low, "MEDIUM": g_med, "HIGH": g_high}

st.sidebar.subheader("Queue zone (fraction of frame)")
zx1, zx2 = st.sidebar.slider("Left - Right", 0.0, 1.0, (0.0, 1.0), 0.01)
zy1, zy2 = st.sidebar.slider("Top - Bottom", 0.0, 1.0, (0.45, 0.95), 0.01)

# ---------------- upload ----------------
uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is None:
    st.info("Upload a video to begin.")
    st.stop()

tmp_dir = tempfile.mkdtemp()
video_path = os.path.join(tmp_dir, uploaded.name)
with open(video_path, "wb") as f:
    f.write(uploaded.getbuffer())

# preview of the first frame with the queue zone, so you can place it before running
cap = cv2.VideoCapture(video_path)
ok, first = cap.read()
cap.release()
if not ok:
    st.error("Could not read this video.")
    st.stop()

first = resize_keep_aspect(first)
fh, fw = first.shape[:2]
preview = draw_roi(first.copy(), roi_from_fractions(fw, fh, zx1, zy1, zx2, zy2))
st.subheader("1. Place the queue zone (just before the stop line)")
st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

# ---------------- run ----------------
st.subheader("2. Run detection")
if st.button("Run SmartFlow", type="primary"):
    out_video = os.path.join(tmp_dir, "smartflow_output.mp4")
    out_csv = os.path.join(tmp_dir, "smartflow_log.csv")
    log = []

    left, right = st.columns([3, 1])
    frame_box = left.empty()
    m_wait = right.empty()
    m_dens = right.empty()
    m_green = right.empty()
    bar = st.progress(0.0)

    with st.spinner("Loading YOLOv8 and processing (first run downloads the model)..."):
        for i, step in enumerate(process_video(
            video_path,
            roi_frac=(zx1, zy1, zx2, zy2),
            conf=conf,
            low_max=low_max,
            med_max=med_max,
            green_times=green_times,
            window_s=window_s,
            waiting_only=waiting_only,
            out_path=out_video,
            log=log,
        )):
            if i % 3 == 0:  # refresh the screen every 3rd frame (faster)
                frame_box.image(cv2.cvtColor(step["frame"], cv2.COLOR_BGR2RGB),
                                use_container_width=True)
                m_wait.metric("Waiting vehicles", step["waiting"])
                m_dens.metric("Density", step["density"])
                m_green.metric("Recommended green", f"{step['green_time']} s",
                               delta=f"{step['green_time'] - FIXED_BASELINE_GREEN:+d} s vs fixed")
                bar.progress(step["progress"])
    bar.progress(1.0)
    st.success("Done.")

    save_log_csv(log, out_csv)
    df = pd.DataFrame(log)
    summary = summarize(log)

    st.subheader("3. Results")
    if summary:
        c = st.columns(4)
        c[0].metric("Avg waiting vehicles", summary["avg_waiting"])
        c[1].metric("Max waiting vehicles", summary["max_waiting"])
        c[2].metric("Avg recommended green", f"{summary['avg_recommended_green_s']} s")
        c[3].metric("Fixed timer", f"{summary['fixed_green_s']} s")
        st.write(f"Time spent in each density level - LOW: {summary['pct_low']}%, "
                 f"MEDIUM: {summary['pct_medium']}%, HIGH: {summary['pct_high']}%")

        fig, ax1 = plt.subplots(figsize=(9, 3.5))
        ax1.plot(df["time_s"], df["waiting_vehicles"], label="Waiting vehicles", color="tab:blue")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Vehicles")
        ax2 = ax1.twinx()
        ax2.step(df["time_s"], df["recommended_green_s"], where="post",
                 label="Recommended green (s)", color="tab:green")
        ax2.axhline(FIXED_BASELINE_GREEN, color="grey", linestyle="--", label="Fixed timer")
        ax2.set_ylabel("Green time (s)")
        fig.legend(loc="upper right")
        fig.tight_layout()
        st.pyplot(fig)
        fig.savefig(os.path.join(tmp_dir, "smartflow_chart.png"), dpi=150)

        st.dataframe(df, use_container_width=True)

        d1, d2 = st.columns(2)
        with open(out_csv, "rb") as f:
            d1.download_button("Download log (CSV)", f, "smartflow_log.csv", "text/csv")
        with open(out_video, "rb") as f:
            d2.download_button("Download annotated video", f, "smartflow_output.mp4", "video/mp4")
