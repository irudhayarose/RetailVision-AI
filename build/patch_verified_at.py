"""Backfill ``verified_at=`` on appearance rows written before that field existed.

An appearance entry is timestamped when the person materialised, but the
decision needs the cancel window, so occupancy actually changes ~4 s later.
Newer runs annotate that in ``detail``; this script repairs an older
``events.csv`` by reading the *actual* change out of ``timeline.csv`` (the first
frame after the appearance where the verified occupancy rises).  Nothing is
assumed about the delay.

Usage:
    python build/patch_verified_at.py [artifact_dir]
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output")


def secs(ts):
    hh, mm, ss = ts.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def stamp(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


def main():
    ev_path = os.path.join(OUT, "events.csv")
    tl_path = os.path.join(OUT, "timeline.csv")
    with open(ev_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    with open(tl_path, encoding="utf-8") as f:
        timeline = [(float(r["t"]), int(r["occupancy_confirmed"]))
                    for r in csv.DictReader(f)]

    patched = 0
    for i, r in enumerate(rows):
        d = r["detail"]
        if not d.startswith("appearance-no-crossing") or "verified_at=" in d:
            continue
        born = secs(r["timestamp"])
        # the first occupancy rise strictly after the birth is the verification
        verified = None
        for j in range(1, len(timeline)):
            t, occ = timeline[j]
            if t <= born:
                continue
            if occ > timeline[j - 1][1]:
                verified = t
                break
        if verified is None:
            print(f"  ! row {i} P{r['person_id']}: no occupancy rise found")
            continue
        rows[i]["detail"] = f"{d} verified_at={stamp(verified)}"
        patched += 1
        print(f"  P{r['person_id']}: materialised {r['timestamp']} "
              f"-> verified {stamp(verified)}")

    if not patched:
        print("nothing to patch")
        return

    with open(ev_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"patched {patched} row(s) in {ev_path}")


if __name__ == "__main__":
    main()
