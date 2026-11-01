"""Scratch: detection-only audit of the door corridor.

No tracking, no gating — just raw YOLO persons sampled every 2nd frame, classified
by signed distance from the gate. Produces:
  build/corridor_audit.json   per-sample anchors + side counts
  build/corridor_moments.txt  suspicious windows (change in inside side w/o event)
"""

import csv
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

HERE = r"C:\Users\irudh_eu7dknc\Downloads\MINE"
with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
P1, P2 = cfg["line"]
CORR_X = 950          # left edge of "door corridor" region of interest
STRIDE = 2


def signed_distance(pt):
    dx, dy = P2[0] - P1[0], P2[1] - P1[1]
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    return (dx * (pt[1] - P1[1]) - dy * (pt[0] - P1[0])) / n


model = YOLO(cfg["model"])
cap = cv2.VideoCapture(os.path.join(HERE, "video.f398.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)

samples = []
fi = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if fi % STRIDE == 0:
        r = model.predict(frame, classes=[0], conf=0.3, imgsz=cfg["imgsz"], verbose=False)[0]
        anchors = []
        try:
            b = r.boxes
            for bx, cf in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                x1, y1, x2, y2 = bx
                cx = (x1 + x2) / 2
                if cx < CORR_X or (y2 - y1) < 45:      # corridor + same size filter
                    continue
                d = signed_distance((cx, y2))
                anchors.append({"cx": int(cx), "by": int(y2), "h": int(y2 - y1),
                                "d": d, "conf": float(cf)})
        except Exception:
            anchors = []
        t = fi / fps
        samples.append({"t": round(t, 2), "anchors": anchors,
                        "n_out": int(sum(1 for a in anchors if a["d"] <= -30)),
                        "n_in": int(sum(1 for a in anchors if a["d"] >= 30))})
    fi += 1
cap.release()

with open(os.path.join(HERE, "build", "corridor_audit.json"), "w", encoding="utf-8") as f:
    json.dump({"fps": fps, "corridor_x": CORR_X, "stride": STRIDE, "samples": samples},
              f, indent=1, default=lambda o: float(o) if isinstance(o, np.generic) else str(o))

events = {}
for e in csv.DictReader(open(os.path.join(HERE, "output", "events.csv"), encoding="utf-8")):
    hh, mm, ss = e["timestamp"].split(":")
    t = int(hh) * 3600 + int(mm) * 60 + float(ss)
    events[t] = 1 if e["event"] == "IN" else -1


def ev_delta(t0, t1):
    return sum(delta for t, delta in events.items() if t0 - 1.0 <= t <= t1 + 1.0)


# smoothed side counts
win = 5
s_in = np.array([s["n_in"] for s in samples], float)
s_out = np.array([s["n_out"] for s in samples], float)
sm_in = np.convolve(s_in, np.ones(win) / win, mode="same")
sm_out = np.convolve(s_out, np.ones(win) / win, mode="same")

print(f"events: {len(events)}   samples: {len(samples)}")
print("\nSUSPICIOUS WINDOWS — people appearing/moving on the inside side with no")
print("matching committed event within +-1 s of the entire rise window:")
HORIZON = int(6.0 * fps / STRIDE)          # 6 seconds of samples
found = []
for i in range(len(samples) - HORIZON):
    j = i + HORIZON
    t0, t1 = samples[i]["t"], samples[j]["t"]
    d_in = sm_in[j] - sm_in[i]
    if d_in < 1.0:                       # inside population did not RISE
        continue
    evd = ev_delta(t0, t1)
    if evd >= 1:                          # an IN event explains it
        continue
    found.append((t0, t1, round(float(sm_in[i]), 2), round(float(sm_in[j]), 2), evd))

# merge overlapping windows
merged = []
for f in found:
    if merged and f[0] - merged[-1][1] < 4:
        continue
    merged.append(f)

print(f"{'window':>16} {'inside before':>14} {'inside after':>13} {'events':>7}")
for t0, t1, a, b, evd in merged[:24]:
    print(f"{t0:7.2f} - {t1:7.2f} {a:>14.2f} {b:>13.2f} {evd:>7d}")
print(f"\n{len(merged)} suspicious windows (showing first 24)")
