"""Verify every appearance-counted entry visually: before/after frames from the
v2 annotated video, so a human can judge whether a person genuinely materialised
at the door.

Usage:
    python build/verify_appearances_v2.py
"""
import csv
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "output")
ANN = os.path.join(OUT, "annotated.mp4")


def secs(ts):
    hh, mm, ss = ts.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def main():
    with open(os.path.join(OUT, "events.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    events = [r for r in rows
              if r.get("detail", "").startswith("appearance-no-crossing")]
    if not events:
        print("no appearance entries in this run")
        return

    cap = cv2.VideoCapture(ANN)
    tiles = []
    for ev in events:
        t = secs(ev["timestamp"])
        pair = []
        for dt, label in ((-2.0, "before"), (2.0, "after")):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t + dt) * 1000)
            ok, fr = cap.read()
            if not ok:
                continue
            fr = cv2.resize(fr, (620, 351))
            cv2.putText(fr, f"appearance P{ev['person_id']} @ {t:.1f}s  {label}",
                        (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255),
                        2, cv2.LINE_AA)
            pair.append(fr)
        if len(pair) == 2:
            tiles.append(np.hstack(pair))
    cap.release()

    if not tiles:
        print("no frames extracted")
        return
    sheet = np.vstack(tiles)
    outp = os.path.join(HERE, "build", "appearance_check_v2.jpg")
    cv2.imwrite(outp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", outp)


if __name__ == "__main__":
    main()
