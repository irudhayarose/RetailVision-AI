"""Did anybody cross the gate *during* the warm-up window? (warm-up boundary check)"""
import json
import os
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retail.geometry import signed_distance
from retail.reid import signature
from retail.tracks import TrackManager

cfg = json.load(open("config.json", encoding="utf-8"))
cap = cv2.VideoCapture(cfg["video"])
fps = cap.get(cv2.CAP_PROP_FPS)
model = YOLO(cfg["model"])
tm = TrackManager(cfg, fps)
trace = {}

for f in range(40):
    ok, fr = cap.read()
    if not ok:
        break
    res = model.track(fr, persist=True, classes=[0], conf=0.35, imgsz=640,
                      verbose=False, tracker="bytetrack.yaml")
    dets = []
    b = res[0].boxes
    if b is not None and b.id is not None:
        for box, tid in zip(b.xyxy.cpu().numpy(), b.id.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = [float(v) for v in box]
            if y2 - y1 < 45:
                continue
            bb = (x1, y1, x2, y2)
            dets.append({"ext_id": int(tid), "box": bb, "sig": signature(fr, bb)})
    tm.ingest(f, dets)
    tm.tick(f)
    for d in dets:
        trace.setdefault(d["ext_id"], {})[f] = ((d["box"][0] + d["box"][2]) / 2,
                                                d["box"][3])
cap.release()

p1, p2 = tm.p1, tm.p2
print("external-ID trajectories over the first 3 s (d>0 = inside):")
for ext in sorted(trace):
    fr = trace[ext]
    ks = sorted(fr)
    x0, y0 = fr[ks[0]]
    x1_, y1_ = fr[ks[-1]]
    d0 = signed_distance((x0, y0), p1, p2)
    d1 = signed_distance((x1_, y1_), p1, p2)
    note = ""
    if d0 < -40 and d1 > 40:
        note = "CROSSES IN during warm-up"
    elif d0 > 40 and d1 < -40:
        note = "CROSSES OUT during warm-up"
    elif d0 < -40:
        note = "started outside, still outside"
    print(f"e{ext:<4} frames {ks[0]:>2}-{ks[-1]:<2} n={len(ks):<2} "
          f"x {x0:5.0f}->{x1_:5.0f}  d {d0:5.0f}->{d1:5.0f}  {note}")
