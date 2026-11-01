"""Verify every appearance event visually: for each fired timestamp, show the
annotated frame at term-2 s (before) and +2 s (after),so a human can judge
whether a person genuinely materialised there.

Usage:
    python build/verify_appearances.py 8.71 59.88 90.28
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(HERE, "output", "annotated.mp4")

times = [float(a) for a in sys.argv[1:]]
if not times:
    # derive from events.csv: appearance rows (person_id >= 5000)
    for e in csv.DictReader(open(os.path.join(HERE, "output", "events.csv"),
                                 encoding="utf-8")):
        if int(e["person_id"]) >= 5000:
            hh, mm, ss = e["timestamp"].split(":")
            times.append(int(hh) * 3600 + int(mm) * 60 + float(ss))

cap = cv2.VideoCapture(ANN)
rows = []
for t in sorted(times):
    tiles = []
    for dt, label in ((-2.0, "before"), (2.0, "after")):
        tt = max(0.0, t + dt)
        cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (620, 351))
        cv2.putText(fr, f"appearance @ {t:.1f}s  {label} ({tt:.1f}s)",
                    (10, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255), 2,
                    cv2.LINE_AA)
        tiles.append(fr)
    if len(tiles) == 2:
        rows.append(np.hstack(tiles))
cap.release()

if rows:
    sheet = np.vstack(rows)
    outp = os.path.join(HERE, "build", "appearance_check.jpg")
    cv2.imwrite(outp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", outp)
else:
    print("no frames extracted")
