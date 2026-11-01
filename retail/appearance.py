"""Door-strip appearance counting.

Some people never produce a committed gate crossing: they walk in as part of a
tight group, the detector only picks them up once they are already inside, or
occlusion hides the crossing itself.  The gate counter cannot see those, so the
system would under-count simultaneous entries -- exactly the failure this layer
fixes.

The rule is deliberately LOCAL.  Genuine late-detected entrants materialise on
the inside side **of the door strip**.  Regains of people who were lost behind
shelving happen deep inside the store, far from the door, and are ignored.

A presence cluster is confirmed only when it is

* sustained for ``appear_confirm_frames``,
* born after warm-up (people already inside are initial occupancy),
* born into an area no inside anchor occupied in the last
  ``appear_new_area_s`` seconds (kills ByteTrack churn regains),
* not explained by a committed inward crossing just before it was born,
* not an already-counted track wandering back through the strip.

Confirmation is delayed by ``appear_cancel_s`` and vetoed when the person is
actually a **leaver**: the cluster's own minimum distance dips outward, a
committed outward crossing happens next to it, or a track that fed the cluster
later dips outside the gate nearby.  Without that veto an undetected exit would
be converted into a phantom entry.
"""
import math
from collections import deque

from .geometry import signed_distance


def _anchor(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


class StripAppearance:
    """Door-strip presence clusters -> verified appearance entries."""

    def __init__(self, cfg, fps, warmup_frames):
        self.fps = float(fps) or 13.0
        self.warmup_frames = int(warmup_frames)
        self.strip_lo, self.strip_hi = cfg.get("appear_strip_x", [950, 1078])
        self.d_min = float(cfg.get("appear_strip_d_min", 12.0))
        self.link_r = float(cfg.get("appear_link_px", 90.0))
        self.new_r = float(cfg.get("appear_new_area_px", 85.0))
        self.new_look = int(round(float(cfg.get("appear_new_area_s", 4.0)) * self.fps))
        self.confirm = int(cfg.get("appear_confirm_frames", 8))
        self.cancel = int(round(float(cfg.get("appear_cancel_s", 4.0)) * self.fps))
        self.credit = int(round(float(cfg.get("appear_credit_s", 2.5)) * self.fps))

        self.clusters = []
        self.mem = deque(maxlen=800)     # recent inside-side strip anchors
        self.fired = 0
        self.notes = []

    # ------------------------------------------------------------------ #
    def _crossing_near(self, tm, kind, ref_frame, pos, before=0.0, after=0.0,
                       radius=160.0):
        """A crossing of `kind` near `pos`, within [ref-before, ref+after]."""
        for c in tm.recent_crossings:
            if c["kind"] != kind:
                continue
            dt = c["frame"] - ref_frame
            if dt < -before or dt > after:
                continue
            if math.hypot(c["pos"][0] - pos[0], c["pos"][1] - pos[1]) <= radius:
                return c
        return None

    def _already_inside(self, tm, tids):
        for ext in tids:
            t = tm.tracks.get(tm.by_ext.get(ext))
            if t is not None and t.inside:
                return True
        return False

    def _born_at_door(self, tm, tids):
        """An entrant's track must have started at/near the door strip.

        A track that began deep in the store and merely walked into the strip
        was already inside; counting it would be a false entry.
        """
        for ext in tids:
            t = tm.tracks.get(tm.by_ext.get(ext))
            if t is None or not t.trail:
                continue
            if t.trail[0][0] < self.strip_lo - 30:
                return False
        return True

    # ------------------------------------------------------------------ #
    def update(self, frame_idx, dets, tm):
        """Consume this frame's detections; return clusters fired this frame."""
        p1, p2 = tm.p1, tm.p2

        # ---- 1. maintain presence clusters in the door strip --------------
        for det in dets:
            box = det["box"]
            a = _anchor(box)
            d = signed_distance(a, p1, p2)
            x, y = a
            if d < self.d_min or not (self.strip_lo <= x <= self.strip_hi):
                continue
            self.mem.append((frame_idx, x, y))

            # 1. identity continuity: this external ID already fed a cluster
            best = None
            for cl in reversed(self.clusters):
                if det["ext_id"] in cl["tids"] and frame_idx - cl["last"] <= 4:
                    best = cl
                    break
            # 2. a nearby cluster updated in the last two frames bridges tracker
            #    ID churn -- but the radius is tight so two people entering side
            #    by side are NOT merged into one presence
            if best is None:
                bd = 55.0
                for cl in self.clusters:
                    if frame_idx - cl["last"] > 2:
                        continue
                    dd = math.hypot(x - cl["cx"], y - cl["cy"])
                    if dd < bd:
                        best, bd = cl, dd

            if best is None:
                new_area = True
                for mf, mx, my in self.mem:
                    if mf < frame_idx and frame_idx - mf <= self.new_look and \
                            math.hypot(x - mx, y - my) < self.new_r:
                        new_area = False
                        break
                best = {"cid": len(self.clusters) + 1, "cx": x, "cy": y,
                        "born": frame_idx, "born_d": d, "seen": 0, "_fr": -1,
                        "last": frame_idx, "min_d": d, "state": "new",
                        "new_area": new_area, "trail": [],
                        "tids": {det["ext_id"]}}
                self.clusters.append(best)

            cl = best
            cl["cx"], cl["cy"] = x, y
            cl["last"] = frame_idx
            cl["min_d"] = min(cl["min_d"], d)
            cl["tids"].add(det["ext_id"])
            cl["trail"].append((frame_idx, x, y))
            if len(cl["trail"]) > 80:
                del cl["trail"][:len(cl["trail"]) - 80]
            if cl["_fr"] != frame_idx:
                cl["_fr"] = frame_idx
                cl["seen"] += 1

        # ---- 2. candidate screening ---------------------------------------
        for cl in self.clusters:
            if cl["state"] != "new" or frame_idx - cl["last"] > 4:
                continue
            if cl["seen"] < self.confirm:
                continue
            if cl["born"] < self.warmup_frames:
                cl["state"] = "initial-occupancy"
                continue
            if not cl["new_area"]:
                cl["state"] = "occupied-area"
                continue
            if not self._born_at_door(tm, cl["tids"]):
                cl["state"] = "occupied-area"
                continue
            if self._crossing_near(tm, "IN", cl["born"], (cl["cx"], cl["cy"]),
                                   before=self.credit) is not None:
                cl["state"] = "crossing-explained"
                continue
            if self._already_inside(tm, cl["tids"]):
                cl["state"] = "already-inside"
                continue
            cl["state"] = "pending"
            cl["confirm_at"] = cl["born"] + self.cancel

        # ---- 3. delayed confirmation with leaver veto ----------------------
        fired = []
        for cl in self.clusters:
            if cl["state"] != "pending" or frame_idx < cl["confirm_at"]:
                continue

            veto = cl["min_d"] < -10
            pos = (cl["cx"], cl["cy"])
            if not veto and self._crossing_near(
                    tm, "OUT", cl["born"], pos, before=self.cancel,
                    after=10 ** 9) is not None:
                # somebody walked out through the gate right where this cluster
                # was born: the presence was an exit in progress
                veto = True
            if not veto:
                # the cluster's own anchors stop updating once the person leaves
                # the strip, so the feeding track's trail is the only witness to
                # a later exit -- check it near the cluster
                for ext in cl["tids"]:
                    t = tm.tracks.get(tm.by_ext.get(ext))
                    if t is None:
                        continue
                    for tx, ty in t.trail:
                        if signed_distance((tx, ty), p1, p2) < -10 and \
                                math.hypot(tx - pos[0], ty - pos[1]) < 160.0:
                            veto = True
                            break
                    if veto:
                        break
            if veto:
                cl["state"] = "cancelled-exiting"
                continue

            cl["state"] = "fired"
            pid = tm.appearance_entry(cl["born"], pos, ext_ids=cl["tids"])
            # the row is timestamped when the person actually appeared, but the
            # decision needed the cancel window; record when it was verified so
            # the occupancy column is auditable (it is the post-verification value)
            tm.events[-1]["detail"] += f" verified_at={tm._ts(frame_idx)}"
            self.fired += 1
            note = (f"appear {cl['born'] / self.fps:7.2f}s cluster {cl['cid']:3d} "
                    f"materialised in door strip, seen {cl['seen']} frames, "
                    f"no committed crossing -> entry as person {pid}")
            self.notes.append(note)
            fired.append({"frame": cl["born"], "cid": cl["cid"], "pid": pid,
                          "seen": cl["seen"]})
        return fired
