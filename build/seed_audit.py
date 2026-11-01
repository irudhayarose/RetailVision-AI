"""Audit the initial-occupancy seed: who is counted inside at the warm-up boundary?

Answers two questions the README must not fudge:
  1. How many *distinct* people are visibly inside at the seed frame?
  2. Are any of the seeded tracks duplicates of the same person (two ByteTrack
     IDs for one body), which would inflate occupancy permanently?

Usage:
    python build/seed_audit.py [seed_frame]
"""
import json
import math
import os
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retail.geometry import signed_distance
from retail.reid import signature
from retail.tracks import TrackManager

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
SEED_FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 39

cap = cv2.VideoCapture(os.path.join(HERE, cfg["video"]))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
model = YOLO(os.path.join(HERE, cfg["model"]))
tm = TrackManager(cfg, fps)


def frame_boxes(frame):
    res = model.track(frame, persist=True, classes=[0], conf=cfg.get("conf", 0.35),
                      imgsz=cfg.get("imgsz", 640), verbose=False,
                      tracker="bytetrack.yaml")
    boxes = res[0].boxes
    dets = []
    if boxes is not None and boxes.id is not None:
        for box, tid in zip(boxes.xyxy.cpu().numpy(),
                            boxes.id.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = [float(v) for v in box]
            if y2 - y1 < float(cfg.get("min_box_h", 45)):
                continue
            b = (x1, y1, x2, y2)
            dets.append({"ext_id": int(tid), "box": b, "sig": signature(frame, b)})
    return dets


snapshot = None
for frame_idx in range(SEED_FRAME + 1):
    ok, frame = cap.read()
    if not ok:
        break
    dets = frame_boxes(frame)
    tm.ingest(frame_idx, dets)
    tm.tick(frame_idx)
    if frame_idx == SEED_FRAME:
        snapshot = frame.copy()

p1, p2 = tm.p1, tm.p2
inside = [t for t in tm.tracks.values()
          if t.last_side == tm.inside_sign and SEED_FRAME - t.last_seen <= 2]

print(f"seed frame {SEED_FRAME} (t={SEED_FRAME / fps:.2f}s): "
      f"{len(inside)} tracks with inside side\n")
print(f"{'pid':>4} {'ext':>5} {'x':>5} {'y':>5} {'w':>4} {'h':>4} {'d':>5}")
info = []
for t in sorted(inside, key=lambda q: q.anchor()[0]):
    x1, y1, x2, y2 = t.box
    ax, ay = t.anchor()
    d = signed_distance((ax, ay), p1, p2)
    print(f"P{t.pid:<3} {t.ext_id:>5} {ax:5.0f} {ay:5.0f} {x2 - x1:4.0f} "
          f"{y2 - y1:4.0f} {d:5.0f}")
    info.append((t, ax, ay, x2 - x1, y2 - y1))

print("\nclosest pairs (a duplicate would be two boxes on one body):")
pairs = []
for i in range(len(info)):
    for j in range(i + 1, len(info)):
        (ta, ax1, ay1, wa, ha), (tb, ax2, ay2, wb, hb) = info[i], info[j]
        dist = math.hypot(ax1 - ax2, ay1 - ay2)
        # horizontal overlap of the boxes' x-ranges, as a fraction of width
        xa1, xa2 = ta.box[0], ta.box[2]
        xb1, xb2 = tb.box[0], tb.box[2]
        ov = max(0.0, min(xa2, xb2) - max(xa1, xb1)) / max(1.0, min(wa, wb))
        pairs.append((dist, ov, ta.pid, tb.pid))
for dist, ov, a, b in sorted(pairs)[:5]:
    print(f"  P{a} / P{b}: anchor distance {dist:5.0f} px, x-overlap {ov:.2f}")
suspects = [p for p in pairs if p[0] < 45 and p[1] > 0.55]
print(f"\nsuspected duplicates: {len(suspects)}")
for dist, ov, a, b in suspects:
    print(f"  P{a} / P{b}  dist={dist:.0f} overlap={ov:.2f}")

tm.seed_initial_occupancy(SEED_FRAME, [t.pid for t in inside])
print(f"\nseeded occupancy = {tm.occupancy}")

if snapshot is not None:
    for i, (t, ax, ay, w, h) in enumerate(info, 1):
        x1, y1, x2, y2 = [int(v) for v in t.box]
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), (60, 220, 255), 2)
        cv2.putText(snapshot, f"#{i} P{t.pid}", (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (60, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(snapshot, (int(ax), int(ay)), 5, (0, 0, 255), -1)
    cv2.line(snapshot, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
             (0, 235, 255), 3)
    cv2.putText(snapshot, f"seeded inside at t={SEED_FRAME / fps:.1f}s: {len(info)}",
                (500, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2,
                cv2.LINE_AA)
    outp = os.path.join(HERE, "build", "seed_audit.jpg")
    cv2.imwrite(outp, snapshot, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", outp)
cap.release()
