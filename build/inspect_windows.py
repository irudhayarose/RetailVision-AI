"""Scratch: inspect the suspicious corridor windows - what did the raw anchors do?
For each window, dump the anchors present (position, size) at the start and end, plus
snapshots from annotated.mp4 so a human can verify whether someone really walked in."""

import csv
import json
import os

import cv2
import numpy as np

HERE = r"C:\Users\irudh_eu7dknc\Downloads\MINE"
audit = json.load(open(HERE + r"\build\corridor_audit.json", encoding="utf-8"))
samples = audit["samples"]

WINDOWS = [(11.61, 17.57), (40.02, 45.98), (54.23, 60.18), (84.63, 90.58)]

rows = []
for w0, w1 in WINDOWS:
    a0 = [s for s in samples if abs(s["t"] - w0) < 0.4]
    a1 = [s for s in samples if abs(s["t"] - w1) < 0.4]
    print(f"\n=== window {w0:.1f} - {w1:.1f} ===")
    for label, lst in (("start", a0), ("end", a1)):
        print(f"  {label}:")
        for s in lst[:2]:
            for a in s["anchors"]:
                side = "OUT" if a["d"] <= -30 else ("IN " if a["d"] >= 30 else "mid")
                print(f"    t={s['t']:6.2f} cx={a['cx']:4} bottom_y={a['by']:3} h={a['h']:3} "
                      f"d={a['d']:+7.1f} {side} conf={a['conf']:.2f}")

# montage: two frames per window (start+end) from annotated.mp4
cap = cv2.VideoCapture(os.path.join(HERE, "output", "annotated.mp4"))
tiles = []
for w0, w1 in WINDOWS:
    tile_row = []
    for t in (w0 + 0.5, w1 - 0.5):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (620, 351))
        cv2.putText(fr, f"t={t:.1f}s", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                    (0, 255, 255), 2, cv2.LINE_AA)
        tile_row.append(fr)
    if len(tile_row) == 2:
        tiles.append(np.hstack(tile_row))
cap.release()
if tiles:
    sheet = np.vstack(tiles)
    cv2.imwrite(os.path.join(HERE, "build", "suspect_windows.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("\nwrote build/suspect_windows.jpg")
