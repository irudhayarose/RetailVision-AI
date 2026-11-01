"""Scratch: strip-presence timeline around the missed audit windows."""
import json

audit = json.load(open(r"C:\Users\irudh_eu7dknc\Downloads\MINE\build\corridor_audit.json"))
samples = audit["samples"]
for s in samples:
    t = s["t"]
    if (30 <= t <= 50) or (80 <= t <= 93):
        n_in = sum(1 for a in s["anchors"] if a["d"] >= 25)
        detail = " ".join("x{}d{:+.0f}".format(a["cx"], a["d"]) for a in s["anchors"])
        if n_in or t % 1 < 0.2:
            print("t={:7.2f} strip_in={} | {}".format(t, n_in, detail))
