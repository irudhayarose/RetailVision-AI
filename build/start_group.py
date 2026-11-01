"""Scratch: extract the first seconds of the RAW video with the gate marked,
so the simultaneous entry group can be seen and counted by eye."""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
P1, P2 = cfg["line"]

times = [0.2, 0.8, 1.4, 2.0, 2.6, 3.2, 3.8, 4.4]
cap = cv2.VideoCapture(os.path.join(HERE, "video.f398.mp4"))
tiles = []
for t in times:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    if not ok:
        continue
    cv2.line(fr, tuple(P1), tuple(P2), (40, 235, 255), 3, cv2.LINE_AA)
    cv2.putText(fr, f"t={t:.1f}s", (14, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1,
                (0, 255, 255), 2, cv2.LINE_AA)
    tiles.append(cv2.resize(fr, (620, 351)))
cap.release()

while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
sheet = np.vstack(rows)
outp = os.path.join(HERE, "build", "start_group.jpg")
cv2.imwrite(outp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("wrote", outp)
