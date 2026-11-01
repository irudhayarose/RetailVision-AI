"""Scratch: probe a timestamp by labelling every tracked person as INSIDE/OUTSIDE
by the same gate the pipeline uses, and stack it against the annotated.mp4 frame.

    python probe_time.py 62 81 100
"""

import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

HERE = r"C:\Users\irudh_eu7dknc\Downloads\MINE"
import json
with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
P1, P2 = cfg["line"]
INSIDE_SIGN = int(cfg["inside_sign"])


def signed_distance(pt):
    dx, dy = P2[0] - P1[0], P2[1] - P1[1]
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    return (dx * (pt[1] - P1[1]) - dy * (pt[0] - P1[0])) / n


times = []
for a in sys.argv[1:]:
    for part in a.split(","):
        part = part.strip().replace(":", ".")
        if not part:
            continue
        try:
            if "." in part:
                mm, ss = part.split(".")
                times.append(int(mm) * 60 + float(ss))
            else:
                times.append(float(part))
        except ValueError:
            print("skipping bad time:", a)
times = sorted(set(t for t in times if 0 <= t <= 110))

model = YOLO(cfg["model"])
for t in times:
    src = cv2.VideoCapture(os.path.join(HERE, cfg["video"]))
    src.set(cv2.CAP_PROP_POS_MSEC, max(0, (t - 1.9)) * 1000)
    last = None
    idx = 0
    while idx < 25:
        ok, fr = src.read()
        if not ok:
            break
        r = model.track(fr, persist=True, classes=[0], conf=cfg["conf"],
                        imgsz=cfg["imgsz"], verbose=False, tracker="bytetrack.yaml")[0]
        b = r.boxes
        ids = b.id.cpu().numpy().astype(int) if b is not None and b.id is not None else []
        xyxy = b.xyxy.cpu().numpy() if b is not None and b.id is not None else []
        for (bx, tid) in zip(xyxy, ids):
            x1, y1, x2, y2 = bx
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            d = signed_distance(((x1 + x2) / 2, y2))
            inside = (1 if d > 0 else -1) == INSIDE_SIGN and abs(d) >= 10
            col = (255, 230, 70) if inside else (64, 180, 255)
            cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), col, 2, cv2.LINE_AA)
            cv2.putText(fr, f"ID{tid} {'IN' if inside else 'OUT'}",
                        (int(x1), max(12, int(y1) - 6)), cv2.FONT_HERSHEY_DUPLEX,
                        0.55, col, 1, cv2.LINE_AA)
            cv2.circle(fr, (int((x1 + x2) / 2), int(y2)), 5, col, -1, cv2.LINE_AA)
            cv2.putText(fr, f"{d:+.0f}", (cx - 20, int(y2) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        last = fr
        idx += 1
    src.release()

    if last is None:
        print("could not read frame at", t)
        continue

    cv2.line(last, tuple(P1), tuple(P2), (40, 235, 255), 3, cv2.LINE_AA)
    cv2.putText(last, "GATE", (P2[0] - 42, P2[1] - 12 if P2[1] < 700 else P2[1] - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (40, 235, 255), 2, cv2.LINE_AA)

    src = cv2.VideoCapture(os.path.join(HERE, "output", "annotated.mp4"))
    src.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, anno = src.read()
    src.release()

    if ok:
        h = min(last.shape[0], anno.shape[0])
        last_r = cv2.resize(last, (int(last.shape[1] * h / last.shape[0]), h))
        anno_r = cv2.resize(anno, (int(anno.shape[1] * h / anno.shape[0]), h))
        canvas = np.vstack([anno_r, last_r])
        cv2.putText(canvas, f"TOP: annotated.mp4 @ {t:.2f}s  |  BOTTOM: gate-relative "
                            f"classification (yellow=INSIDE, blue=OUTSIDE)",
                    (24, canvas.shape[0] - 16), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (30, 30, 30), 6, cv2.LINE_AA)
        cv2.putText(canvas, f"TOP: annotated.mp4 @ {t:.2f}s  |  BOTTOM: gate-relative "
                            f"classification (yellow=INSIDE, blue=OUTSIDE)",
                    (24, canvas.shape[0] - 16), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (240, 240, 240), 1, cv2.LINE_AA)
        outp = os.path.join(HERE, "build", f"probe_{t:.2f}.jpg")
        cv2.imwrite(outp, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", outp)
    else:
        print("annotated.mp4 read failed for t =", t)
