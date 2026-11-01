"""Validate output/events.csv + output/summary.json against provable invariants.

Checks:
  1. CSV schema: exactly the 4 required columns.
  2. Timestamps parse and are monotonically non-decreasing.
  3. event is IN or OUT.
  4. person_id: numeric; appearance rows must be >= appear_id_base (5000).
  5. Every IN row has a headcount >= 0 (a relative timestamp outside the clip
     would be a bug, not a count).
  6. summary.json: entered_crossings == #  IN rows with id < 5000 +
     appearance_entries == # rows with id >= 5000; left_crossings == # OUT rows.
  7. FLOW consistency: final_flow_delta == entered - left.
  8. Total rows == rows in CSV.

Exit code 0 = all invariants hold.
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(HERE, "output", "events.csv")
SUM_PATH = os.path.join(HERE, "output", "summary.json")

errors = []
warnings = []


def err(msg):
    errors.append(msg)


rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
if not rows:
    err("events.csv is empty")
summary = json.load(open(SUM_PATH, encoding="utf-8"))


def t_seconds(ts):
    hh, mm, ss = ts.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


prev_t = -1.0
for i, r in enumerate(rows, 1):
    if set(r.keys()) != {"timestamp", "person_id", "event", "headcount"}:
        err(f"row {i}: bad columns {sorted(r.keys())}")
        break
    try:
        t = t_seconds(r["timestamp"])
    except Exception as e:
        err(f"row {i}: unparseable timestamp {r['timestamp']!r} ({e})")
        continue
    if t < prev_t:
        err(f"row {i}: timestamp {r['timestamp']} goes backwards (prev {prev_t:.3f})")
    prev_t = t
    if r["event"] not in ("IN", "OUT"):
        err(f"row {i}: event must be IN/OUT, got {r['event']!r}")
    try:
        pid = int(r["person_id"])
    except ValueError:
        err(f"row {i}: person_id not numeric: {r['person_id']!r}")
        continue
    try:
        hc = int(r["headcount"])
    except (TypeError, ValueError):
        err(f"row {i}: headcount not numeric: {r['headcount']!r}")
        continue
    if hc < 0:
        err(f"row {i}: negative headcount {hc}")

n_in_cross = sum(1 for r in rows if r["event"] == "IN" and int(r["person_id"]) < 5000)
n_appear = sum(1 for r in rows if r["event"] == "IN" and int(r["person_id"]) >= 5000)
n_out = sum(1 for r in rows if r["event"] == "OUT")

checks = [
    ("entered_crossings", summary["entered_crossings"], n_in_cross),
    ("appearance_entries", summary.get("appearance_entries"), n_appear),
    ("entered_total", summary.get("entered_total"),
     summary["entered_crossings"] + summary.get("appearance_entries", 0)),
    ("left_crossings", summary["left_crossings"], n_out),
    ("events", summary["events"], len(rows)),
    ("final_flow_delta", summary["final_flow_delta"],
     summary.get("entered_total", summary["entered_crossings"])
     - summary["left_crossings"]),
]
for name, a, b in checks:
    if a != b:
        err(f"summary.{name} = {a} but events.csv says {b}")

if summary.get("audit_seed_recorded", 0) < 0:
    err("audit_seed_recorded is negative")

print(f"rows: {len(rows)}  "
      f"IN-crossings: {n_in_cross}  appearance-INs: {n_appear}  OUT: {n_out}")
for r in rows:
    print(f"   {r['timestamp']}  ID {r['person_id']:>5}  {r['event']:<3} headcount {r['headcount']}")

if warnings:
    for w in warnings:
        print("WARN:", w)
if errors:
    print("\nFAILED:")
    for e in errors:
        print("  -", e)
    sys.exit(1)
print("\nAll invariants hold.")
