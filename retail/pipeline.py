"""End-to-end pipeline: detect -> track -> state machine -> dashboard -> logs.

This is the v2 system.  ``count_people.py`` remains in the repository as the v1
threshold-based build for comparison; here every occupancy change is tied to a
verified event and a per-person state machine.
"""
import csv
import json
import os
import time
from collections import Counter

import cv2
from ultralytics import YOLO

from .appearance import StripAppearance
from .camera import camera_status
from .dashboard import (draw_entrance, draw_ghost, draw_hud, draw_track,
                        draw_ticker)
from .reid import signature
from .state import Event, State
from .tracks import TrackManager

CSV_FIELDS = ["timestamp", "person_id", "event", "headcount", "state",
              "confidence", "detail"]


def run(cfg, base_dir, seconds=None, render=True, out_dir=None, progress=True):
    video_path = os.path.join(base_dir, cfg["video"])
    out_dir = os.path.join(base_dir, out_dir or cfg.get("output_dir", "output"))
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO(cfg["model"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = int(seconds * fps) if seconds else total
    conf = float(cfg.get("conf", 0.35))
    imgsz = int(cfg.get("imgsz", 640))
    min_h = float(cfg.get("min_box_h", 45))
    warmup_frames = int(round(float(cfg.get("warmup_s", 3.0)) * fps))

    if progress:
        print(f"video {width}x{height} @ {fps:.2f} fps, {total} frames "
              f"({total / fps:.1f}s) -> processing {limit}", flush=True)

    suffix = f"_{int(seconds)}s" if seconds else ""
    video_out = os.path.join(out_dir, f"annotated{suffix}.mp4")
    csv_out = os.path.join(out_dir, f"events{suffix}.csv")
    trace_out = os.path.join(out_dir, f"timeline{suffix}.csv")
    summary_out = os.path.join(out_dir, f"summary{suffix}.json")

    writer = None
    if render:
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise SystemExit("VideoWriter failed to open")

    tm = TrackManager(cfg, fps)
    appearance = StripAppearance(cfg, fps, warmup_frames)
    cam = camera_status(cfg, (height, width))
    for w in cam["warnings"]:
        tm.warnings.append(w)

    seeded = False
    seeded_at = None
    timeline = []
    t0 = time.time()
    frame_idx = 0
    printed_notes = 0
    printed_events = 0

    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.track(frame, persist=True, classes=[0], conf=conf,
                          imgsz=imgsz, verbose=False, tracker="bytetrack.yaml")
        boxes = res[0].boxes

        dets = []
        if boxes is not None and boxes.id is not None:
            for box, tid in zip(boxes.xyxy.cpu().numpy(),
                                boxes.id.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = [float(v) for v in box]
                if (y2 - y1) < min_h:
                    continue
                b = (x1, y1, x2, y2)
                dets.append({"ext_id": int(tid), "box": b,
                             "sig": signature(frame, b)})

        tm.ingest(frame_idx, dets)
        tm.tick(frame_idx)

        # ---- initial occupancy: people already inside at the end of warm-up ---
        if not seeded and frame_idx >= warmup_frames:
            pids = [t.pid for t in tm.tracks.values()
                    if t.last_side == tm.inside_sign
                    and frame_idx - t.last_seen <= 2]
            tm.seed_initial_occupancy(frame_idx, pids)
            seeded = True
            seeded_at = frame_idx
            if progress:
                print(f"  [warmup {frame_idx / fps:.1f}s] initial occupancy seeded"
                      f" = {tm.seeded_initial} (inside before the clip can evidence"
                      f" a crossing)", flush=True)

        # ---- appearance layer (simultaneous / unhidden entries) --------------
        fired = appearance.update(frame_idx, dets, tm)
        if progress:
            for note in appearance.notes[printed_notes:]:
                print(f"  [{note}]", flush=True)
            printed_notes = len(appearance.notes)
            for ev in tm.events[printed_events:]:
                print(f"  [{ev['timestamp']}] P{ev['person_id']:4d} "
                      f"{ev['event']:9s} inside={ev['headcount']} "
                      f"state={ev['state']}  ({ev['detail']})", flush=True)
            printed_events = len(tm.events)

        m = tm.metrics()
        visible_now = tm.visible_inside_count(dets)
        timeline.append((round(frame_idx / fps, 3), m["occupancy_confirmed"],
                         visible_now, m["occluded_now"], m["uncertain_now"]))

        if writer is not None:
            draw_entrance(frame, tm.p1, tm.p2, tm.inside_sign, cam)
            for t in tm.tracks.values():
                if t.missed == 0 and t.box is not None:
                    draw_track(frame, t, frame_idx, fps)
            for t in tm.tracks.values():
                if t.inside and t.state in (State.OCCLUDED, State.POSSIBLY_EXITED) \
                        and 2 < t.missed <= int(10 * fps):
                    draw_ghost(frame, t)
            draw_hud(frame, m, visible_now, (frame_idx + 1) / max(time.time() - t0, 1e-6),
                     frame_idx, cam, tm.events,
                     recent=(sum(1 for c in tm.recent_crossings
                                 if c["kind"] == "IN"
                                 and frame_idx - c["frame"] <= 60 * fps),
                             sum(1 for c in tm.recent_crossings
                                 if c["kind"] == "OUT"
                                 and frame_idx - c["frame"] <= 60 * fps)))
            writer.write(frame)

        frame_idx += 1
        if progress and frame_idx % 60 == 0:
            el = max(time.time() - t0, 1e-6)
            print(f"  ... {frame_idx}/{limit} frames, {frame_idx / el:.1f} fps, "
                  f"inside={m['occupancy_confirmed']} visible={visible_now} "
                  f"entries={m['entries_total']} exits={m['exits_total']}",
                  flush=True)

    cap.release()
    if writer is not None:
        writer.release()

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(tm.events)

    with open(trace_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "occupancy_confirmed", "visible_now", "occluded",
                    "uncertain"])
        w.writerows(timeline)

    counts = Counter(e["event"] for e in tm.events)
    entry_crossings = sum(1 for e in tm.events
                          if e["event"] in (Event.ENTRY.value, Event.REENTRY.value)
                          and e["detail"].startswith("committed"))
    entry_appearance = sum(1 for e in tm.events
                           if e["detail"].startswith("appearance-no-crossing"))
    entry_reid = sum(1 for e in tm.events
                     if e["event"] in (Event.ENTRY.value, Event.REENTRY.value)
                     and e["detail"].startswith("reid"))
    m = tm.metrics()
    runtime = round(time.time() - t0, 1)
    summary = {
        "system": "retail-occupancy-v2 (state machine, verified events)",
        "video": os.path.basename(video_path),
        "frames_processed": frame_idx,
        "duration_s": round(frame_idx / fps, 2),
        "fps_source": round(fps, 2),
        "occupancy_confirmed_final": m["occupancy_confirmed"],
        "visible_now_final": timeline[-1][2] if timeline else 0,
        "peak_occupancy": m["peak_occupancy"],
        "initial_occupancy_seeded": m["initial_seeded"],
        "seeded_at_frame": seeded_at,
        "seeded_pids": m["seeded_pids"],
        "entries_total": m["entries_total"],
        "entry_crossings": entry_crossings,
        "entry_appearance": entry_appearance,
        "entry_reid": entry_reid,
        "exits_total": m["exits_total"],
        "reentries_total": m["reentries_total"],
        "unique_visitors": m["unique_visitors"],
        "unique_visitors_seen_total": m["unique_visitors"] + m["initial_seeded"],
        "staff_inside": m["staff_inside"],
        "customers_estimate": m["customers_estimate"],
        "customers_reliable": m["customers_reliable"],
        "occluded_now": m["occluded_now"],
        "uncertain_now": m["uncertain_now"],
        "tracks_total": m["tracks_total"],
        "events": len(tm.events),
        "event_counts": dict(counts),
        "appearance": {
            "clusters": len(appearance.clusters),
            "fired": appearance.fired,
            "states": dict(Counter(cl["state"] for cl in appearance.clusters)),
        },
        "camera": cam,
        "state_legend": [s.value for s in State],
        "event_legend": [e.value for e in Event],
        "annotated_video": os.path.relpath(video_out, base_dir) if writer else None,
        "events_csv": os.path.relpath(csv_out, base_dir),
        "timeline_csv": os.path.relpath(trace_out, base_dir),
        "runtime_s": runtime,
        "throughput_fps": round(frame_idx / runtime, 2) if runtime else None,
        "config": {"line": [list(tm.p1), list(tm.p2)],
                   "inside_sign": tm.inside_sign,
                   "commit_px": tm.commit,
                   "min_event_gap_s": round(tm.min_gap / fps, 2),
                   "occlusion_hold_s": round(tm.hold / fps, 2),
                   "reid_sim": tm.reid_thr,
                   "conf": conf, "model": cfg["model"]},
        "notes": [
            "INSIDE is the verified-event account: seeded initial occupancy plus "
            "verified entries minus verified exits. It never drops because a box "
            "disappeared.",
            "VISIBLE NOW is the detector-visible population inside the line at that "
            "instant; it is a measurement and fluctuates with occlusion.",
            "Rows with detail 'appearance-no-crossing' are entries the gate never "
            "witnessed; they keep the person's identity and are not unique-visitor "
            "duplicates.",
        ],
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if progress:
        print("\n=== DONE ===")
        print(json.dumps(summary, indent=2))
    return summary
