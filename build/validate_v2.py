"""Validate the v2 event log, timeline and summary against each other.

Checks (all must hold):
  * schema and monotonic timestamps
  * headcount changes by exactly +1 on ENTRY/REENTRY, -1 on EXIT, 0 otherwise
  * final headcount equals summary.occupancy_confirmed_final
  * no EXIT for a person who was not inside (entries/seed tracked)
  * no duplicate ENTRY for the same person without an intervening EXIT
  * unique_visitors + reentries_total == entries_total
  * timeline occupancy changes exactly match the events on that frame
  * occupancy is never negative and never drops without an EXIT
  * occluded <= uncertain; visible count >= 0
  * peak occupancy matches the timeline maximum

Usage:
    python build/validate_v2.py [artifact_dir]
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output")

ENTRY = "ENTRY"
REENTRY = "REENTRY"
EXIT = "EXIT"
UNCERTAIN = "UNCERTAIN"
REAPPEARED = "REAPPEARED"
ALLOWED = {ENTRY, REENTRY, EXIT, UNCERTAIN, REAPPEARED}
FIELDS = ["timestamp", "person_id", "event", "headcount", "state",
          "confidence", "detail"]


def secs(ts):
    hh, mm, ss = ts.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    errors = []

    def check(cond, msg):
        if not cond:
            errors.append(msg)

    rows = load(os.path.join(OUT, "events.csv"))
    timeline = load(os.path.join(OUT, "timeline.csv"))
    with open(os.path.join(OUT, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    # ---- schema ---------------------------------------------------------- #
    check(rows and list(rows[0].keys()) == FIELDS,
          f"events.csv header {list(rows[0].keys()) if rows else None} != {FIELDS}")
    check(timeline and list(timeline[0].keys()) ==
          ["t", "occupancy_confirmed", "visible_now", "occluded", "uncertain"],
          "timeline.csv header mismatch")

    # ---- timestamps and event names -------------------------------------- #
    prev_t = -1.0
    for i, r in enumerate(rows):
        t = secs(r["timestamp"])
        check(t >= prev_t, f"row {i}: timestamp went backwards")
        prev_t = t
        check(r["event"] in ALLOWED, f"row {i}: unknown event {r['event']}")
        check(float(r["confidence"]) >= 0, f"row {i}: negative confidence")

    # ---- per-event accounting -------------------------------------------- #
    baseline = int(summary.get("initial_occupancy_seeded", 0))
    h = baseline
    entered_pids = {}          # pid -> True while inside
    seen_entry = set()
    seeded = set()
    n_entry = n_reentry = n_exit = 0

    for i, r in enumerate(rows):
        pid = int(r["person_id"])
        ev = r["event"]
        expected = h
        if ev in (ENTRY, REENTRY):
            expected = h + 1
            check(pid not in entered_pids,
                  f"row {i}: duplicate {ev} for P{pid} without an exit")
            if ev == REENTRY:
                n_reentry += 1
                check(pid in seen_entry,
                      f"row {i}: REENTRY for P{pid} with no earlier entry")
            else:
                n_entry += 1
            seen_entry.add(pid)
            entered_pids[pid] = True
        elif ev == EXIT:
            expected = h - 1
            n_exit += 1
            if pid not in entered_pids:
                # a seeded person leaving; only allowed once
                check(pid not in seeded,
                      f"row {i}: duplicate EXIT for seeded P{pid}")
                seeded.add(pid)
            entered_pids.pop(pid, None)
        actual = int(r["headcount"])
        check(actual == expected,
              f"row {i} ({ev} P{pid}): headcount {actual} != expected {expected}")
        check(actual >= 0, f"row {i}: negative headcount")
        h = actual

    check(h == summary["occupancy_confirmed_final"],
          f"final headcount {h} != summary {summary['occupancy_confirmed_final']}")
    check(n_entry + n_reentry == summary["entries_total"],
          f"entries {n_entry}+{n_reentry} != summary {summary['entries_total']}")
    check(n_exit == summary["exits_total"],
          f"exits {n_exit} != summary {summary['exits_total']}")
    check(n_reentry == summary["reentries_total"],
          f"reentries {n_reentry} != summary {summary['reentries_total']}")
    check(n_entry == summary["unique_visitors"],
          f"unique {n_entry} != summary {summary['unique_visitors']}")
    check(summary["unique_visitors"] + summary["reentries_total"] ==
          summary["entries_total"], "unique + reentries != entries")

    n_committed = sum(1 for r in rows
                      if r["detail"].startswith("committed")
                      and r["event"] in (ENTRY, REENTRY))
    n_appear = sum(1 for r in rows
                   if r["detail"].startswith("appearance-no-crossing"))
    check(n_committed == summary["entry_crossings"],
          f"committed crossings {n_committed} != summary "
          f"{summary['entry_crossings']}")
    check(n_appear == summary["entry_appearance"],
          f"appearance entries {n_appear} != summary "
          f"{summary['entry_appearance']}")
    check(n_committed + n_appear <= summary["entries_total"],
          "entry breakdown exceeds entries_total")

    # ---- timeline vs events ---------------------------------------------- #
    # appearance rows are backdated to the materialisation frame; their detail
    # carries the verification time, which is when occupancy actually changed.
    delta = {}
    for r in rows:
        m = re.search(r"verified_at=(\d\d:\d\d:\d\d\.\d+)", r["detail"])
        t = round(secs(m.group(1) if m else r["timestamp"]), 2)
        d = 1 if r["event"] in (ENTRY, REENTRY) else (
            -1 if r["event"] == EXIT else 0)
        delta[t] = delta.get(t, 0) + d

    # the initial occupancy is seeded once, by definition without a crossing:
    # that single step is accounted for here and nowhere else
    seeded_at = summary.get("seeded_at_frame")
    if seeded_at is not None:
        seed_t = round(seeded_at / summary["fps_source"], 2)
        delta[seed_t] = delta.get(seed_t, 0) + int(summary["initial_occupancy_seeded"])

    prev = None
    peak = 0
    for i, r in enumerate(timeline):
        t = float(r["t"])
        occ = int(r["occupancy_confirmed"])
        visible = int(r["visible_now"])
        occ_now = int(r["occluded"])
        unc = int(r["uncertain"])
        check(occ >= 0, f"timeline row {i}: negative occupancy")
        check(visible >= 0, f"timeline row {i}: negative visible count")
        check(unc >= occ_now, f"timeline row {i}: uncertain < occluded")
        peak = max(peak, occ)
        if prev is not None:
            actual = occ - prev
            expected = 0
            for et, d in delta.items():
                if abs(et - t) <= 0.05:
                    expected += d
            check(actual == expected,
                  f"timeline row {i} (t={t}): delta {actual} != events {expected}")
        prev = occ

    check(peak == summary["peak_occupancy"],
          f"timeline peak {peak} != summary {summary['peak_occupancy']}")
    check(timeline and int(timeline[-1]["occupancy_confirmed"]) ==
          summary["occupancy_confirmed_final"], "timeline end != summary")

    # ---- report ----------------------------------------------------------- #
    print(f"rows: {len(rows)}  ENTRY: {n_entry}  REENTRY: {n_reentry}  "
          f"EXIT: {n_exit}  appearance: {n_appear}")
    for r in rows:
        print(f"   {r['timestamp']}  P{int(r['person_id']):4d}  "
              f"{r['event']:9s} hc={r['headcount']:>2s}  "
              f"{r['state']:<16s} {r['detail']}")
    if errors:
        print("\nFAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("\nAll v2 invariants hold.")


if __name__ == "__main__":
    main()
