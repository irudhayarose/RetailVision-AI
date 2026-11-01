"""Scratch diagnostic: find every near-threshold motion near the gate, whether or not
an event was committed, so missed/phantom IN-OUTs can be identified by a human.

Writes build/gate_candidates.json and snapshot jpgs for the strongest candidates.
"""

import json
import os
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(HERE, "video.f398.mp4")
OUTDIR = os.path.join(HERE, "build")
os.makedirs(OUTDIR, exist_ok=True)

with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
P1, P2 = cfg["line"]
COMMIT = float(cfg["commit_px"])
INSIDE_SIGN = int(cfg["inside_sign"])
STRIDE = 2


def signed_distance(pt, p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    return (dx * (pt[1] - p1[1]) - dy * (pt[0] - p1[0])) / n


model = YOLO(cfg["model"])
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{total} frames @ {fps:.2f} fps; gate {P1}->{P2}, commit {COMMIT}")

history = {}          # tid -> last {frame, d}
candidates = []
snap_saved = 0

fi = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if fi % STRIDE == 0:
        r = model.track(frame, persist=True, classes=[0], conf=cfg["conf"],
                        imgsz=cfg["imgsz"], verbose=False, tracker="bytetrack.yaml")[0]
        b = r.boxes
        if b is not None and b.id is not None:
            for box, tid in zip(b.xyxy.cpu().numpy(),
                                b.id.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box
                d = signed_distance(((x1 + x2) / 2, y2), P1, P2)
                tid = int(tid)
                prev = history.get(tid)
                history[tid] = {"frame": fi, "d": d}
                if prev:
                    pd = prev["d"]
                    # a genuine sign flip through the gate layer = candidate crossing
                    if d * pd < 0 and abs(d) < 200 and abs(pd) < 200:
                        t = fi / fps
                        strong_enough = min(abs(d), abs(pd)) > 25   # both sides at least 25px past
                        c = {
                            "tid": tid,
                            "t_prev": round(prev["frame"] / fps, 2),
                            "t_cross": round(t, 2),
                            "d_before": round(pd, 1),
                            "d_after": round(d, 1),
                            "travel": round(abs(d - pd), 1),
                            "strong": strong_enough,
                        }
                        candidates.append(c)
                        if strong_enough and snap_saved < 40:
                            vis = frame.copy()
                            cv2.line(vis, P1, P2, (40, 235, 255), 4, cv2.LINE_AA)
                            cx, cy = int((x1 + x2) / 2), int(y2)
                            cv2.circle(vis, (cx, cy), 7, (0, 0, 255), -1, cv2.LINE_AA)
                            cv2.putText(vis, f"ID{tid}  d:{pd:+.0f}->{d:+.0f}",
                                        (cx - 60, cy - 14), cv2.FONT_HERSHEY_DUPLEX,
                                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(vis, f"t={t:.2f}s",
                                        (cx - 60, cy + 26), cv2.FONT_HERSHEY_DUPLEX,
                                        0.6, (40, 235, 255), 2, cv2.LINE_AA)
                            cv2.imwrite(os.path.join(OUTDIR, f"gate_t{t:.2f}_id{tid}.jpg"),
                                        vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            snap_saved += 1
    fi += 1
    if fi % 120 == 0:
        print(f"  ... frame {fi}/{total}", flush=True)

cap.release()

with open(os.path.join(OUTDIR, "gate_candidates.json"), "w", encoding="utf-8") as f:
    json.dump({"gate": [P1, P2], "commit": COMMIT, "n_candidates": len(candidates),
               "candidates": candidates}, f, indent=2,
              default=lambda o: float(o) if isinstance(o, np.generic) else str(o))

strong = [c for c in candidates if c["strong"]]
print(f"\ncandidates total {len(candidates)} | strong {len(strong)}")
for c in strong:
    dirn = "OUT->IN" if (c["d_before"] < 0) == (INSIDE_SIGN > 0) else "IN->OUT"
    print(f"  t={c['t_cross']:.2f} ID{c['tid']:3d} d {c['d_before']:+.0f} -> {c['d_after']:+.0f}  {dirn}")
print(f"snapshots saved: {snap_saved} in {OUTDIR}")
