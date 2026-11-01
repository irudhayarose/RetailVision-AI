"""Scratch: classify the diagnostic's gate candidates against committed events."""
import csv
import json

HERE = r"C:\Users\irudh_eu7dknc\Downloads\MINE"

diag = json.load(open(HERE + r"\build\gate_candidates.json", encoding="utf-8"))
events = list(csv.DictReader(open(HERE + r"\output\events.csv", encoding="utf-8")))

print("committed events (events.csv):")
for e in events:
    print(f"   {e['timestamp']}  ID {e['person_id']:>3}  {e['event']:<3} headcount {e['headcount']}")

print("\nall sign-flip candidates seen by the diagnostic (stride 2 sampling):")
print(f"{'time(s)':>9} {'ID':>4} {'d_before':>10} {'d_after':>10} {'max|d|':>8}  class")
for c in diag["candidates"]:
    m = max(abs(c["d_before"]), abs(c["d_after"]))
    tag = "COMMIT-SIZED" if m >= 40 else "SUB-COMMIT (would NOT fire with commit=40)"
    print(f"{c['t_cross']:>9.2f} {c['tid']:>4} {c['d_before']:>+10.1f} {c['d_after']:>+10.1f} "
          f"{m:>8.1f}  {tag}")

# direction decode: inside_sign=+1 on a p1->p2 = (0,477) line means d<0 = outside (x>1080)
print("\nnote: d<0 = outside (door side), d>0 = inside (shop side) for this line")
