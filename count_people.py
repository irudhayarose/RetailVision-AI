

import argparse
import csv
import json
import math
import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "video": "video.f398.mp4",
    "model": "yolov8n.pt",
    "output_dir": "output",
    "line": [[250, 60], [250, 700]],
    "inside_sign": 1,
    "commit_px": 40.0,
    "init_band_px": 10.0,
    "warmup_s": 3.0,
    "band_px": 12.0,
    "min_event_gap_s": 2.0,
    "appear_link_px": 110,
    "appear_confirm_frames": 8,
    "appear_id_base": 5000,
    "conf": 0.35,
    "imgsz": 640,
    "min_box_h": 45,
    "max_gap_frames": 45,
}


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def signed_distance(pt, p1, p2):
    """Signed perpendicular distance from pt to the line p1->p2 (pixels)."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = math.hypot(dx, dy) or 1.0
    cross = dx * (pt[1] - p1[1]) - dy * (pt[0] - p1[0])
    return cross / norm


def side_of(pt, p1, p2, band):
    """+1 / -1 for clearly one side, 0 when inside the dead-band."""
    d = signed_distance(pt, p1, p2)
    if abs(d) < band:
        return 0
    return 1 if d > 0 else -1


def anchor(box):
    """Feet position: bottom-centre of the bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


# --------------------------------------------------------------------------- #
# per-track state
# --------------------------------------------------------------------------- #
class Track:
    __slots__ = ("tid", "last_side", "seen_frame", "trail", "did_in", "did_out",
                 "last_dir", "flash_until", "last_event_frame")

    def __init__(self, tid):
        self.tid = tid
        self.last_side = 0          # last *committed* side (never 0)
        self.seen_frame = -1
        self.trail = deque(maxlen=45)
        self.did_in = False
        self.did_out = False
        self.last_dir = None
        self.flash_until = -1
        self.last_event_frame = -10 ** 9

COL_INT = (80, 220, 80)      # green
COL_OUT = (70, 70, 235)      # red
COL_IDLE = (200, 190, 90)    # teal-ish
FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_hud(frame, entered, left, inside_now, flow, fps_live, frame_idx):
    """ENTERED/LEFT are crossing events; INSIDE is the live tracked population inside
    the gate; FLOW is entered-left (the audit counter, seed-independent)."""
    h_panel, w_panel = 180, 320
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (18 + w_panel, 18 + h_panel), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (18, 18), (18 + w_panel, 18 + h_panel), (110, 110, 110), 2)

    cv2.putText(frame, "STORE FOOTFALL", (34, 48), FONT, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(frame, (34, 58), (18 + w_panel - 16, 58), (110, 110, 110), 1)

    rows = [("ENTERED", entered, COL_INT), ("LEFT", left, COL_OUT),
            ("INSIDE", inside_now, (60, 210, 250)), ("FLOW", flow, (150, 150, 150))]
    y = 90
    for label, value, col in rows:
        cv2.putText(frame, label, (34, y), FONT, 0.6, col, 1, cv2.LINE_AA)
        txt = str(value)
        cv2.putText(frame, txt, (18 + w_panel - 30 - 20 * len(txt), y), FONT, 0.95, col, 2, cv2.LINE_AA)
        y += 32

    cv2.putText(frame, f"frame {frame_idx}  {fps_live:4.1f} fps",
                (34, 18 + h_panel - 12), FONT, 0.42, (150, 150, 150), 1, cv2.LINE_AA)


def draw_line(frame, p1, p2, inside_sign):
    p1 = tuple(int(v) for v in p1)
    p2 = tuple(int(v) for v in p2)
    cv2.line(frame, p1, p2, (0, 0, 0), 9, cv2.LINE_AA)
    cv2.line(frame, p1, p2, (40, 235, 255), 4, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(frame, p, 7, (40, 235, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 7, (0, 0, 0), 1, cv2.LINE_AA)

    # small arrow marking the IN direction
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    n = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / n * 34 * inside_sign, dx / n * 34 * inside_sign
    cv2.arrowedLine(frame, (int(mx), int(my)), (int(mx + nx), int(my + ny)),
                    (0, 255, 120), 3, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(frame, "IN", (int(mx + nx) + 6, int(my + ny) + 5), FONT, 0.5,
                (0, 255, 120), 1, cv2.LINE_AA)


def draw_track(frame, tr, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    if tr.last_dir == "IN":
        col = COL_INT
    elif tr.last_dir == "OUT":
        col = COL_OUT
    else:
        col = COL_IDLE

    # trail
    if len(tr.trail) > 1:
        pts = np.array([[int(a), int(b)] for a, b in tr.trail], np.int32)
        cv2.polylines(frame, [pts], False, col, 2, cv2.LINE_AA)

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

    label = f"ID {tr.tid}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 12, y1), col, -1)
    cv2.putText(frame, label, (x1 + 6, y1 - 6), FONT, 0.55, (15, 15, 15), 1, cv2.LINE_AA)

    cx = (x1 + x2) // 2
    cv2.circle(frame, (cx, y2), 4, col, -1, cv2.LINE_AA)


def draw_flash(frame, tr, event, frame_idx):
    if frame_idx > tr.flash_until:
        return
    col = COL_INT if event == "IN" else COL_OUT
    txt = f"ID {tr.tid}  {event}"
    (tw, th), _ = cv2.getTextSize(txt, FONT, 0.8, 2)
    x = int(tr.trail[-1][0]) - tw // 2 if tr.trail else 60
    y = int(tr.trail[-1][1]) - 40 if tr.trail else 120
    x = max(10, min(x, frame.shape[1] - tw - 10))
    y = max(60, min(y, frame.shape[0] - 10))
    cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), col, -1)
    cv2.putText(frame, txt, (x, y), FONT, 0.8, (10, 10, 10), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def load_config(path):
    cfg = dict(DEFAULTS)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def analyze(model, cap, frames_total, stride=3):
    """Report where track anchors first appear / last vanish -> finds the portal."""
    starts, ends = [], []
    frame_idx = 0
    first, last = {}, {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            res = model.track(frame, persist=True, classes=[0], conf=0.35,
                              imgsz=640, verbose=False, tracker="bytetrack.yaml")
            boxes = res[0].boxes
            if boxes is not None and boxes.id is not None:
                for box, tid in zip(boxes.xyxy.cpu().numpy(),
                                    boxes.id.cpu().numpy().astype(int)):
                    a = anchor(box)
                    first.setdefault(tid, a)
                    last[tid] = a
        frame_idx += 1
    for tid in first:
        starts.append(first[tid])
        ends.append(last[tid])

    def summarise(name, pts):
        if not pts:
            print(f"  {name}: none")
            return
        a = np.array(pts)
        print(f"  {name}: n={len(a)}  x med={int(np.median(a[:,0]))} "
              f"p10={int(np.percentile(a[:,0],10))} p90={int(np.percentile(a[:,0],90))}")
        print(f"           y med={int(np.median(a[:,1]))} "
              f"p10={int(np.percentile(a[:,1],10))} p90={int(np.percentile(a[:,1],90))}")

    print("track apparition / disappearance zones:")
    summarise("first-seen", starts)
    summarise("last-seen", ends)
    print("\nUse these to place the line between the two zones.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--seconds", type=float, default=None, help="only process N seconds")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="skip rendering")
    ap.add_argument("--out-dir", default="output_v1",
                    help="artifact directory for this legacy build "
                         "(default: output_v1, so it cannot overwrite the v2 "
                         "system's output/)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    video_path = os.path.join(HERE, cfg["video"])
    out_dir = os.path.join(HERE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO(cfg["model"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = int(args.seconds * fps) if args.seconds else total
    print(f"video {width}x{height} @ {fps:.2f} fps, {total} frames "
          f"({total / fps:.1f}s) -> processing {limit}")

    if args.analyze:
        analyze(model, cap, total)
        return

    p1, p2 = cfg["line"]
    band = float(cfg["band_px"])
    commit_px = float(cfg.get("commit_px", 40.0))
    init_band = float(cfg.get("init_band_px", 10.0))
    warmup_frames = int(round(float(cfg.get("warmup_s", 3.0)) * fps))
    min_gap_frames = int(round(float(cfg.get("min_event_gap_s", 2.0)) * fps))
    appear_confirm = int(cfg.get("appear_confirm_frames", 8))
    appear_cancel = int(round(float(cfg.get("appear_cancel_s", 2.0)) * fps))
    appear_credit = int(round(float(cfg.get("appear_credit_s", 2.5)) * fps))
    strip_lo, strip_hi = cfg.get("appear_strip_x", [950, 1078])
    strip_d_min = float(cfg.get("appear_strip_d_min", 12.0))
    link_r = float(cfg.get("appear_link_px", 90.0))
    new_r = float(cfg.get("appear_new_area_px", 85.0))
    new_look = int(round(float(cfg.get("appear_new_area_s", 4.0)) * fps))
    appear_id_base = int(cfg.get("appear_id_base", 5000))
    inside_sign = int(cfg["inside_sign"])   # which side counts as "inside"
    conf = float(cfg["conf"])
    imgsz = int(cfg["imgsz"])
    min_h = float(cfg["min_box_h"])

    suffix = f"_{int(args.seconds)}s" if args.seconds else ""
    video_out = os.path.join(out_dir, f"annotated{suffix}.mp4")
    csv_out = os.path.join(out_dir, f"events{suffix}.csv")
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise SystemExit("VideoWriter failed to open")

    tracks = {}
    events = []
    entered = 0            # IN crossings only — the answer to "how many entered"
    left = 0               # OUT crossings only
    inside_now = 0         # instantaneous detector-verified population inside the gate
    flow_val = 0           # entered - left, the audit counter
    peak = 0               # max of inside_now over the clip
    seeded = False
    seed_value = 0
    strip_clusters = []            # door-strip presence clusters (appearance layer)
    strip_mem = deque(maxlen=600)  # recent inside-side strip anchors (frame, x, y)
    in_frames = deque(maxlen=120)  # frames of committed crossing-INs (for credit)
    trace = []                     # per-frame [t, inside_now] for offline audit
    appear_in = 0                  # appearance-counted entries (subset of `entered`)
    t0 = time.time()
    frame_idx = 0

    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.track(frame, persist=True, classes=[0], conf=conf,
                          imgsz=imgsz, verbose=False, tracker="bytetrack.yaml")
        boxes = res[0].boxes

        current = []
        pending = []
        if boxes is not None and boxes.id is not None:
            for box, tid in zip(boxes.xyxy.cpu().numpy(),
                                boxes.id.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box
                if (y2 - y1) < min_h:
                    continue
                tid = int(tid)
                tr = tracks.get(tid) or tracks.setdefault(tid, Track(tid))
                a = anchor(box)
                tr.seen_frame = frame_idx
                tr.trail.append(a)
                current.append(tr)

                d = signed_distance(a, p1, p2)
                if tr.last_side == 0:
                    # establish a side as soon as the person is clearly off the line,
                    # otherwise a track that starts near the door can never count
                    if abs(d) >= init_band:
                        tr.last_side = 1 if d > 0 else -1
                    s = 0
                else:
                    s = 1 if d >= commit_px else (-1 if d <= -commit_px else 0)
                if s != 0 and s != tr.last_side:
                    prev_side = tr.last_side
                    tr.last_side = s                 # commit the side either way
                    if prev_side != 0:
                        event = "IN" if s == inside_sign else "OUT"
                        duplicate = (event == "IN" and tr.did_in) or (event == "OUT" and tr.did_out)
                        too_soon = (frame_idx - tr.last_event_frame) < min_gap_frames
                        if not duplicate and not too_soon:
                            if event == "IN":
                                entered += 1
                                tr.did_in = True
                                in_frames.append(frame_idx)   # used by credit checks
                                # the crosser's own strip cluster (born right after
                                # they pass the line) is their crossing explained —
                                # cancel it so the same person is not counted twice
                                a_now = tr.trail[-1]
                                for cl in strip_clusters:
                                    if cl["state"] in ("new", "pending") and math.hypot(
                                            a_now[0] - cl["cx"],
                                            a_now[1] - cl["cy"]) < 110.0:
                                        cl["state"] = "crossing-explained"
                            else:
                                left += 1
                                tr.did_out = True
                                # a committed OUT near a pending strip cluster cancels
                                # it: the cluster's person just walked out through the
                                # gate, so their presence was exit-in-progress, not a
                                # late-entrant materialise (cluster anchors no longer
                                # reach the cluster once d < strip_d_min, which is why
                                # the min_d dip check alone cannot veto this)
                                a_now = tr.trail[-1]
                                for cl in strip_clusters:
                                    if cl["state"] in ("new", "pending") and math.hypot(
                                            a_now[0] - cl["cx"],
                                            a_now[1] - cl["cy"]) < 110.0:
                                        cl["state"] = "cancelled-by-OUT"
                            tr.last_event_frame = frame_idx
                            tr.last_dir = event
                            tr.flash_until = frame_idx + int(fps * 0.9)
                            ts = frame_idx / fps
                            # headcount is filled in AFTER this frame's detection pass,
                            # from the live in-gate population — see below
                            events.append({
                                "timestamp": f"{int(ts // 3600):02d}:{int(ts % 3600 // 60):02d}:{ts % 60:06.3f}",
                                "person_id": tid,
                                "event": event,
                                "headcount": None,
                            })
                            pending.append(events[-1])

                if writer is not None:
                    draw_track(frame, tr, box)

        inside_now = sum(1 for tr in current if tr.last_side == inside_sign)
        if not seeded and frame_idx >= warmup_frames:
            # recorded for audit only: how many people the detector saw inside at the
            # end of warm-up. This is the reference the flow counter starts from.
            seed_value = inside_now
            seeded = True
            print(f"  [warmup {frame_idx / fps:.1f}s] audit seed recorded = {seed_value}")

      
        for tr in current:
            a = tr.trail[-1]
            d = signed_distance(a, p1, p2)
            x = a[0]
            if d >= strip_d_min and strip_lo <= x <= strip_hi:
                strip_mem.append((frame_idx, x, a[1]))
                best = None
                for cl in strip_clusters:
                    if frame_idx - cl["last"] > 4:
                        continue
                    hit = math.hypot(x - cl["cx"], a[1] - cl["cy"]) < link_r
                    # associate against the cluster's whole recent trail too, so a
                    # walking person who flickers out of the detector is not
                    # re-clustered as a fresh presence ahead of their own trail
                    for tf, tx, ty in cl["trail"]:
                        r = link_r if frame_idx - tf <= int(1.5 * fps) else 160.0
                        if math.hypot(x - tx, a[1] - ty) < r:
                            hit = True
                            break
                    if hit:
                        best = cl
                        break
                if best is None:
                    new_area = True
                    for mf, mx, my in strip_mem:
                        if mf < frame_idx and frame_idx - mf <= new_look and \
                                math.hypot(x - mx, a[1] - my) < new_r:
                            new_area = False
                            break
                    cl = {"cid": len(strip_clusters) + 1, "cx": x, "cy": a[1],
                          "born": frame_idx, "born_d": d, "seen": 0, "_fr": -1,
                          "last": frame_idx, "min_d": d, "state": "new",
                          "new_area": new_area, "trail": [], "tids": {tr.tid}}
                    strip_clusters.append(cl)
                else:
                    cl = best
                cl["cx"], cl["cy"] = x, a[1]
                cl["last"] = frame_idx
                cl["min_d"] = min(cl["min_d"], d)
                cl["tids"].add(tr.tid)
                cl["trail"].append((frame_idx, x, a[1]))
                if len(cl["trail"]) > 80:
                    del cl["trail"][:len(cl["trail"]) - 80]
                if cl["_fr"] != frame_idx:
                    cl["_fr"] = frame_idx
                    cl["seen"] += 1

        for cl in strip_clusters:
            if cl["state"] != "new" or frame_idx - cl["last"] > 4:
                continue
            if cl["seen"] < appear_confirm:
                continue
            # candidate conditions, evaluated in order of my confidence in each
            if cl["born"] <= warmup_frames:
                cl["state"] = "initial-occupancy"
                continue
            if not cl["new_area"]:
                cl["state"] = "occupied-area"
                continue
            if any(cl["born"] - appear_credit <= f < cl["born"] for f in in_frames):
                cl["state"] = "crossing-explained"
                continue
            cl["state"] = "pending"
            cl["confirm_at"] = cl["born"] + appear_cancel

        for cl in strip_clusters:
            if cl["state"] != "pending" or frame_idx < cl["confirm_at"]:
                continue
            veto = cl["min_d"] < -10
            if not veto:
                # any track that fed this cluster whose own trail later dips OUTSIDE
                # the gate near the cluster was a leaver in progress: cancel, because
                # firing here would convert an undetected exit into a phantom entry.
                # (The per-cluster min_d check cannot see this: cluster anchors stop
                # updating once the person's `d` drops below strip_d_min, so the dip
                # never lands on the cluster itself.)
                for tid in cl["tids"]:
                    tr = tracks.get(tid)
                    if tr is None:
                        continue
                    for tx, ty in tr.trail:
                        if signed_distance((tx, ty), p1, p2) < -10 and \
                                math.hypot(tx - cl["cx"], ty - cl["cy"]) < 160.0:
                            veto = True
                            break
                    if veto:
                        break
            if veto:
                cl["state"] = "cancelled-exiting"
                continue
            cl["state"] = "fired"
            appear_in += 1
            entered += 1
            ts = cl["born"] / fps
            events.append({
                "timestamp": f"{int(ts // 3600):02d}:{int(ts % 3600 // 60):02d}:{ts % 60:06.3f}",
                "person_id": appear_id_base + cl["cid"],
                "event": "IN",
                "headcount": None,
            })
            pending.append(events[-1])
            print(f"  [appear {ts:7.2f}s] cluster {cl['cid']:3d} appeared inside the "
                  f"door strip, persists {cl['seen']} frames, no committed crossing "
                  f"-> counted as IN")

        # live headcount INSIDE is a measurement, not an account: the detector-verified
        # population inside the gate, right now. FLOW (entered-left) is the audit number.
        # Any event recorded on this frame gets its headcount from this snapshot.
        flow_val = max(0, entered - left)
        peak = max(peak, inside_now)
        trace.append([round(frame_idx / fps, 2), inside_now])
        for ev in pending:
            ev["headcount"] = inside_now
            print(f"  [{ev['timestamp']}] ID {ev['person_id']:3d} "
                  f"{ev['event']:3s} -> INSIDE {inside_now} (FLOW {flow_val})")
        pending.clear()

        if writer is not None:
            draw_line(frame, p1, p2, inside_sign)
            for tr in tracks.values():
                if frame_idx - tr.seen_frame <= 2 and tr.last_dir:
                    draw_flash(frame, tr, tr.last_dir, frame_idx)
            elapsed = max(time.time() - t0, 1e-6)
            draw_hud(frame, entered, left, inside_now, flow_val,
                     (frame_idx + 1) / elapsed, frame_idx)
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 60 == 0:
            el = time.time() - t0
            print(f"  ... {frame_idx}/{limit} frames, {frame_idx / el:.1f} fps", flush=True)

    cap.release()
    if writer is not None:
        writer.release()

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "person_id", "event", "headcount"])
        w.writeheader()
        w.writerows(events)

    with open(os.path.join(HERE, "build", "inside_trace.json"), "w", encoding="utf-8") as f:
        json.dump({"fps": fps, "trace": trace}, f)

    summary = {
        "video": os.path.basename(video_path),
        "frames_processed": frame_idx,
        "duration_s": round(frame_idx / fps, 2),
        "entered_total": entered,
        "entered_crossings": entered - appear_in,
        "left_crossings": left,
        "appearance_entries": appear_in,
        "audit_seed_recorded": seed_value,
        "final_live_headcount": inside_now,
        "peak_live_headcount": peak,
        "final_flow_delta": flow_val,
        "flow_note": "FLOW = entered - left across the clip (audit only; live INSIDE is "
                     "the detector-verified population, events.csv headcount is that "
                     "live value at the event instant)",
        "unique_tracks": len(tracks),
        "events": len(events),
        "annotated_video": os.path.relpath(video_out, HERE) if writer is not None else None,
        "events_csv": os.path.relpath(csv_out, HERE),
        "config": {"line": [p1, p2], "band_px": band, "commit_px": commit_px,
                   "min_event_gap_s": round(min_gap_frames / fps, 2),
                   "inside_sign": inside_sign,
                   "conf": conf, "model": cfg["model"]},
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, f"summary{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DONE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
