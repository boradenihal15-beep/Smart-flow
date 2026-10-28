"""
SmartFlow: AI Traffic Brain - core logic (single approach / one direction prototype)

Pipeline:
  video -> frames -> YOLOv8 detect + track vehicles -> keep vehicles inside the
  queue zone (ROI) near the stop line -> count the WAITING ones -> classify density
  (LOW / MEDIUM / HIGH) -> recommend green-signal time.

Run without the UI:
    python smartflow_core.py my_video.mp4 --out result.mp4
"""
import argparse
import csv
from collections import defaultdict, deque

import cv2
import numpy as np

# COCO class ids that YOLOv8 knows -> vehicle names
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

DEFAULT_GREEN_TIMES = {"LOW": 20, "MEDIUM": 40, "HIGH": 60}  # seconds
FIXED_BASELINE_GREEN = 30  # what a traditional fixed-timer signal would give

COLORS = {  # BGR
    "waiting": (0, 200, 0),    # green  : vehicle waiting in the queue zone
    "moving": (0, 215, 255),   # yellow : vehicle inside zone but moving
    "outside": (160, 160, 160),  # grey   : vehicle outside the zone
}


def classify_density(count, low_max=5, med_max=12):
    """Map a vehicle count to a density level."""
    if count <= low_max:
        return "LOW"
    if count <= med_max:
        return "MEDIUM"
    return "HIGH"


def roi_from_fractions(frame_w, frame_h, x1, y1, x2, y2):
    """Rectangle ROI given as fractions (0-1) of the frame -> polygon points."""
    pts = [
        (x1 * frame_w, y1 * frame_h),
        (x2 * frame_w, y1 * frame_h),
        (x2 * frame_w, y2 * frame_h),
        (x1 * frame_w, y2 * frame_h),
    ]
    return np.array(pts, dtype=np.int32)


def resize_keep_aspect(frame, max_width=960):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))


def draw_roi(frame, roi, alpha=0.18):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi], (255, 120, 0))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [roi], True, (255, 120, 0), 2)
    return frame


def process_video(
    video_path,
    roi_frac=(0.0, 0.45, 1.0, 0.95),
    model_name="yolov8n.pt",
    conf=0.35,
    low_max=5,
    med_max=12,
    green_times=None,
    window_s=5.0,
    waiting_only=True,
    still_ratio=0.30,
    max_width=960,
    out_path=None,
    log=None,
    model=None,
):
    """
    Generator. Yields one dict per processed frame:
        {frame, t, total_in_zone, waiting, avg_waiting, density, green_time, progress}
    Per-second rows are appended to `log` (a list) if given.

    waiting_only : count only (almost) stationary vehicles in the zone.
    still_ratio  : a vehicle is "waiting" if, over the last second, it moved less than
                   still_ratio * its own box size.
    window_s     : density is decided from the average count over the last window_s seconds,
                   so the timer does not flicker with every frame.
    """
    from ultralytics import YOLO  # imported here so the UI loads fast

    green_times = green_times or DEFAULT_GREEN_TIMES
    model = model or YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    writer = None
    roi = None
    history = defaultdict(lambda: deque(maxlen=max(2, int(fps))))  # id -> recent centres
    recent_counts = deque(maxlen=max(1, int(fps * window_s)))
    last_logged_sec = -1
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_keep_aspect(frame, max_width)
        h, w = frame.shape[:2]
        if roi is None:
            roi = roi_from_fractions(w, h, *roi_frac)
        if out_path and writer is None:
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        t = frame_idx / fps
        result = model.track(
            frame,
            persist=True,
            conf=conf,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
            tracker="bytetrack.yaml",
        )[0]

        draw_roi(frame, roi)
        in_zone = 0
        waiting = 0

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None
                else np.arange(len(xyxy)) + 100000
            )
            for (x1, y1, x2, y2), cls, tid in zip(xyxy, clses, ids):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                size = max(x2 - x1, y2 - y1)
                history[tid].append((cx, cy))

                inside = cv2.pointPolygonTest(roi, (float(cx), float(cy)), False) >= 0
                is_still = False
                hist = history[tid]
                if len(hist) == hist.maxlen:
                    moved = np.hypot(hist[-1][0] - hist[0][0], hist[-1][1] - hist[0][1])
                    is_still = moved < still_ratio * size

                if inside:
                    in_zone += 1
                    if is_still or not waiting_only:
                        waiting += 1
                        state = "waiting"
                    else:
                        state = "moving"
                else:
                    state = "outside"

                color = COLORS[state]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{VEHICLE_CLASSES.get(cls, 'veh')} #{tid}" if boxes.id is not None else VEHICLE_CLASSES.get(cls, "veh")
                cv2.putText(frame, label, (int(x1), max(15, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        recent_counts.append(waiting)
        avg_waiting = float(np.mean(recent_counts))
        density = classify_density(avg_waiting, low_max, med_max)
        green = green_times[density]

        # ---- info panel -------------------------------------------------
        panel = [
            f"Waiting in zone: {waiting}   (avg {avg_waiting:.1f})",
            f"Density: {density}",
            f"Recommended GREEN: {green}s  (fixed: {FIXED_BASELINE_GREEN}s)",
            f"t = {t:5.1f}s",
        ]
        cv2.rectangle(frame, (0, 0), (420, 24 * len(panel) + 10), (0, 0, 0), -1)
        for i, line in enumerate(panel):
            cv2.putText(frame, line, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)

        # ---- one log row per second -------------------------------------
        sec = int(t)
        if log is not None and sec != last_logged_sec:
            last_logged_sec = sec
            log.append({
                "time_s": sec,
                "vehicles_in_zone": in_zone,
                "waiting_vehicles": waiting,
                "avg_waiting": round(avg_waiting, 2),
                "density": density,
                "recommended_green_s": green,
                "fixed_green_s": FIXED_BASELINE_GREEN,
            })

        frame_idx += 1
        yield {
            "frame": frame, "t": t, "total_in_zone": in_zone, "waiting": waiting,
            "avg_waiting": avg_waiting, "density": density, "green_time": green,
            "progress": min(1.0, frame_idx / total_frames),
        }

    cap.release()
    if writer is not None:
        writer.release()


def summarize(log):
    """Short summary of a finished run (for the report / slides)."""
    if not log:
        return {}
    waiting = [r["waiting_vehicles"] for r in log]
    greens = [r["recommended_green_s"] for r in log]
    levels = [r["density"] for r in log]
    n = len(log)
    return {
        "seconds_analysed": n,
        "avg_waiting": round(float(np.mean(waiting)), 2),
        "max_waiting": int(max(waiting)),
        "avg_recommended_green_s": round(float(np.mean(greens)), 1),
        "fixed_green_s": FIXED_BASELINE_GREEN,
        "pct_low": round(100 * levels.count("LOW") / n, 1),
        "pct_medium": round(100 * levels.count("MEDIUM") / n, 1),
        "pct_high": round(100 * levels.count("HIGH") / n, 1),
    }


def save_log_csv(log, path):
    if not log:
        return
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        wr.writeheader()
        wr.writerows(log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SmartFlow prototype (one approach)")
    ap.add_argument("video")
    ap.add_argument("--out", default="smartflow_output.mp4")
    ap.add_argument("--csv", default="smartflow_log.csv")
    ap.add_argument("--roi", nargs=4, type=float, default=[0.0, 0.45, 1.0, 0.95],
                    metavar=("X1", "Y1", "X2", "Y2"), help="queue zone as fractions of the frame")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--show", action="store_true", help="show a live window")
    args = ap.parse_args()

    rows = []
    last_print = -1
    for step in process_video(args.video, roi_frac=tuple(args.roi), conf=args.conf,
                              out_path=args.out, log=rows):
        if args.show:
            cv2.imshow("SmartFlow", step["frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if int(step["t"]) % 5 == 0 and int(step["t"]) != last_print:
            last_print = int(step["t"])
            print(f"t={step['t']:.0f}s waiting={step['waiting']} "
                  f"density={step['density']} green={step['green_time']}s")
    save_log_csv(rows, args.csv)
    print("Summary:", summarize(rows))
    print(f"Saved: {args.out}, {args.csv}")
