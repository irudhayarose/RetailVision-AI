"""Scratch: why did (or didn't) the 8.7 s appearance cluster fire?"""
import json
import os
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retail.appearance import StripAppearance
from retail.geometry import signed_distance
from retail.reid import signature
from retail.tracks import TrackManager

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
cap = cv2.VideoCapture(os.path.join(HERE, cfg["video"]))
fps = cap.get(cv2.CAP_PROP_FPS)
model = YOLO(os.path.join(HERE, cfg["model"]))
tm = TrackManager(cfg, fps)
app = StripAppearance(cfg, fps, int(3.0 * fps))
p1, p2 = tm.p1, tm.p2

limit = int(12 * fps)
for frame_idx in range(limit):
    ok, frame = cap.read()
    if not ok:
        break
    res = model.track(frame, persist=True, classes=[0], conf=0.35, imgsz=640,
                      verbose=False, tracker="bytetrack.yaml")
    boxes = res[0].boxes
    dets = []
    if boxes is not None and boxes.id is not None:
        for box, tid in zip(boxes.xyxy.cpu().numpy(),
                            boxes.id.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = [float(v) for v in box]
            if y2 - y1 < 45:
                continue
            b = (x1, y1, x2, y2)
            dets.append({"ext_id": int(tid), "box": b, "sig": signature(frame, b)})
    tm.ingest(frame_idx, dets)
    tm.tick(frame_idx)
    app.update(frame_idx, dets, tm)

    if frame_idx >= int(6.5 * fps) and frame_idx % 3 == 0:
        t = frame_idx / fps
        det_line = " ".join(
            f"e{d['ext_id']}@x{int((d['box'][0]+d['box'][2])/2)}"
            f"(d{int(signed_distance(((d['box'][0]+d['box'][2])/2, d['box'][3]), p1, p2))})"
            for d in dets if (d["box"][0] + d["box"][2]) / 2 > 900)
        tr_line = " ".join(
            f"P{tr.pid}/e{tr.ext_id}@{int(tr.anchor()[0])}"
            f"({tr.state.value[:5]}{'I' if tr.inside else ''})"
            for tr in tm.tracks.values()
            if tr.trail and tr.anchor()[0] > 900)
        cl_line = " ".join(
            f"c{cl['cid']}[{cl['state']} born{cl['born']/fps:.1f} x{int(cl['cx'])} "
            f"mind{int(cl['min_d'])} na{int(cl['new_area'])} t{len(cl['tids'])}]"
            for cl in app.clusters)
        print(f"t={t:5.2f} det {det_line}")
        print(f"        trk {tr_line}")
        print(f"        clu {cl_line}")
cap.release()
