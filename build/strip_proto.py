"""Scratch: prototype a localized "appearance entry" detector on corridor_audit.json.

Signal: door-strip population (people on the INSIDE side of the gate within the
corridor strip, x >= CORR_X), smoothed. A genuine entrant must appear in the strip;
regains of people lost behind shelving happen deep inside, away from the door.

Rule to test: strip count exceeding its trailing-min baseline (a strip that was
empty/minimal for a while and then holds someone) fires a candidate, with credit
subtraction for committed IN crossings.

Also prints the strip timeline so the fires can be judged against the 4 audited
windows (11.6-17.6, 40-46, 54-60, 84.6-90.6 s).
"""
import csv
import json

HERE = r"C:\Users\irudh_eu7dknc\Downloads\MINE"
audit = json.load(open(HERE + r"\build\corridor_audit.json", encoding="utf-8"))
samples = audit["samples"]          # every 2nd frame, ~6.5 Hz
sr = 2 / audit["fps"]              # sample period in seconds

# committed events from the CURRENT render (crossing ground truth)
ev_in = []
for e in csv.DictReader(open(HERE + r"\output\events.csv", encoding="utf-8")):
    hh, mm, ss = e["timestamp"].split(":")
    t = int(hh) * 3600 + int(mm) * 60 + float(ss)
    if e["event"] == "IN" and int(e["person_id"]) < 5000:
        ev_in.append(t)

# per-sample strip-inside count (d >= 25: inner side, x <= 1055)
strip = [0.0] * len(samples)
for i, s in enumerate(samples):
    n = sum(1 for a in s["anchors"] if a["d"] >= 25)
    strip[i] = n

# smooth over ~1.2 s
W = max(1, int(round(1.2 / sr)))
sm = [sum(strip[max(0, i - W + 1):i + 1]) / len(strip[max(0, i - W + 1):i + 1])
      for i in range(len(strip))]

# rolling trailing-min over REF seconds, ending at i-1 (exclusive of current)
REF_S = 10.0
REFW = int(round(REF_S / sr))
MINP = 0.0

candidates = []
win_lo = 0
for i in range(len(samples)):
    lo = max(0, i - REFW)
    window = sm[lo:i]
    baseline = min(window) if window else 0.0
    if baseline <= MINP and sm[i] >= 1.0:
        candidates.append(i)

# require sustained: next REFW-consistency: candidate counts kept where sm>=1
# for the next 4 samples (~0.8 s)
fires = []
for i in candidates:
    if all(sm[j] >= 1.0 for j in range(i, min(len(sm), i + 4))):
        fires.append(i)

# merge fires closer than 2 s, record first
merged = []
for i in fires:
    t = samples[i]["t"]
    if merged and t - merged[-1][0] < 2.0:
        continue
    merged.append((t, sm[i]))

print(f"committed IN crossings: {[round(t, 2) for t in ev_in]}")
print(f"\nlocalized strip-appearance candidate fires ({len(merged)}):")
for t, v in merged:
    # credit: was there a committed IN within +-4 s?
    near = [round(x, 2) for x in ev_in if abs(x - t) <= 4.0]
    tag = f"  explained by crossing IN at {near}" if near else ""
    print(f"  t={t:7.2f}  strip_sm={v:.2f}{tag}")

audited = [(11.6, 17.6), (40.0, 46.0), (54.2, 60.2), (84.6, 90.6)]
print("\naudited genuine windows:")
for a, b in audited:
    hit = any(a - 1 <= t <= b + 1 for t, _ in merged)
    print(f"  {a:6.1f}-{b:6.1f}: {'FIRED' if hit else 'MISSED'}")
