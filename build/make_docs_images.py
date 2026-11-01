"""Regenerate docs images from the v2 render.

Produces:
  docs/hud.png     dashboard panel + event ticker, cropped and enlarged
  docs/event.png   full frame at the first verified entry (flash visible)
  docs/states.png  full frame showing occluded-ghost markers
"""
import csv
import os

import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "output")
DOCS = os.path.join(HERE, "docs")
os.makedirs(DOCS, exist_ok=True)


def secs(ts):
    hh, mm, ss = ts.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


with open(os.path.join(OUT, "events.csv"), encoding="utf-8") as f:
    events = list(csv.DictReader(f))
with open(os.path.join(OUT, "timeline.csv"), encoding="utf-8") as f:
    timeline = list(csv.DictReader(f))

entry = next((e for e in events if e["event"] in ("ENTRY", "REENTRY")), None)
t_event = (secs(entry["timestamp"]) + 0.35) if entry else 5.0

ghost = next((r for r in timeline
              if int(r["occluded"]) > 0 and float(r["t"]) > 8.0), None)
t_ghost = (float(ghost["t"]) + 0.2) if ghost else 20.0

cap = cv2.VideoCapture(os.path.join(OUT, "annotated.mp4"))


def frame_at(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    return fr if ok else None


ev = frame_at(t_event)
if ev is not None:
    cv2.imwrite(os.path.join(DOCS, "event.png"), ev,
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print("wrote docs/event.png")

st = frame_at(t_ghost)
if st is not None:
    cv2.imwrite(os.path.join(DOCS, "states.png"), st,
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print("wrote docs/states.png")

mid = frame_at(max(0.0, float(timeline[-1]["t"]) - 2.0))
if mid is not None:
    hud = mid[10:600, 10:436].copy()
    hud = cv2.resize(hud, (724, 1001), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(DOCS, "hud.png"), hud,
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print("wrote docs/hud.png")

cap.release()
