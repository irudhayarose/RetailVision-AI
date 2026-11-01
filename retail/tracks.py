"""Per-person track lifecycle, verified entry/exit, occlusion and re-ID.

Design rules (from the brief):

* Occupancy changes ONLY on verified events. A bounding box disappearing never
  removes anybody from the store: it becomes OCCLUDED, then POSSIBLY_EXITED,
  and is logged as UNCERTAIN, but occupancy is retained.
* A verified EXIT requires a committed outward crossing of the entrance line.
* Re-identification may *propose* an association (appearance signature), but the
  decision is corroborated by time and position; a re-entering customer produces
  a new entry event and does not become a duplicate unique visitor.
* People are customers by default; staff is assigned only with zone evidence.
"""
import math
from collections import deque

from .geometry import in_polygon, signed_distance
from .reid import similarity
from .state import Event, State

CUSTOMER = "CUSTOMER"
UNKNOWN = "UNKNOWN"


class PersonTrack:
    """One physical person, independent of the tracker's (reusable) external ID."""

    def __init__(self, pid, frame_idx):
        self.pid = pid
        self.ext_id = None
        self.trail = deque(maxlen=60)          # recent foot anchors (x, y)
        self.box = None                        # last bounding box
        self.first_seen = frame_idx
        self.last_seen = frame_idx
        self.missed = 0
        self.state = State.UNKNOWN
        self.confidence = 0.5
        self.inside = False                    # verified occupancy membership
        self.seeded = False                    # counted at warm-up, not an ENTRY
        self.group_id = None
        self.classification = CUSTOMER         # CUSTOMER / STAFF
        self.staff_dwell = 0
        self.sig = None
        self.last_side = 0                     # committed side (never 0 after init)
        self.last_event = None
        self.last_event_frame = -10 ** 9
        self.entry_count = 0
        self.exit_count = 0
        self.exit_anchor = None
        self.occlusion_logged = False
        self.reappeared_until = 0

    def anchor(self):
        return self.trail[-1] if self.trail else (0.0, 0.0)

    def predicted_anchor(self):
        if len(self.trail) < 2:
            return self.anchor()
        (x0, y0), (x1, y1) = self.trail[-2], self.trail[-1]
        return (x1 + (x1 - x0), y1 + (y1 - y0))

    def observe(self, frame_idx, box, sig=None):
        x1, y1, x2, y2 = box
        self.box = box
        self.trail.append(((x1 + x2) / 2.0, y2))
        self.last_seen = frame_idx
        self.missed = 0
        if sig is not None:
            self.sig = sig


class TrackManager:
    """Owns identities, the state machine and the verified occupancy account."""

    def __init__(self, cfg, fps):
        self.cfg = cfg
        self.fps = float(fps) or 13.0
        self.p1 = tuple(cfg["line"][0])
        self.p2 = tuple(cfg["line"][1])
        self.inside_sign = int(cfg.get("inside_sign", 1))
        self.commit = float(cfg.get("commit_px", 40.0))
        self.init_band = float(cfg.get("init_band_px", 10.0))
        self.min_gap = int(round(float(cfg.get("min_event_gap_s", 4.0)) * self.fps))
        self.hold = int(round(float(cfg.get("occlusion_hold_s", 8.0)) * self.fps))
        self.reid_thr = float(cfg.get("reid_sim", 0.55))
        self.reid_gap = int(round(float(cfg.get("reid_max_gap_s", 120.0)) * self.fps))
        self.reid_radius = float(cfg.get("reid_radius_px", 260.0))
        self.group_window = int(round(float(cfg.get("group_window_s", 2.5)) * self.fps))
        self.group_radius = float(cfg.get("group_radius_px", 220.0))
        self.staff_zone = cfg.get("staff_zone") or None
        self.staff_dwell_frames = int(round(float(cfg.get("staff_dwell_s", 20.0)) * self.fps))
        self.late_entry_frames = int(round(
            float(cfg.get("late_entry_s", 1.0)) * self.fps))
        self.warmup_frames = int(round(
            float(cfg.get("warmup_s", 3.0)) * self.fps))
        self.late_entry_distance = float(
            cfg.get("late_entry_distance_px", 220.0))
        self.approach_px = float(cfg.get("approach_px", 140.0))
        self.strip_lo, self.strip_hi = cfg.get("appear_strip_x", [950, 1078])
        self.strip_d_min = float(cfg.get("appear_strip_d_min", 12.0))

        self.tracks = {}          # pid -> PersonTrack
        self.by_ext = {}          # external tracker id -> pid
        self.next_pid = 1
        self.next_gid = 1
        self.recent_entries = deque(maxlen=16)    # (frame, pos, pid)
        self.recent_crossings = deque(maxlen=64)  # {frame, kind, pos, pid}

        # verified-event occupancy
        self.occupancy = 0
        self.peak = 0
        self.entries = 0
        self.exits = 0
        self.reentries = 0
        self.unique_visitors = 0
        self.seeded_initial = 0
        self.seeded_pids = []      # captured at seed time, never recomputed

        self.events = []          # dict rows, written to CSV
        self.warnings = []

    # ------------------------------------------------------------------ #
    # logging
    # ------------------------------------------------------------------ #
    def _ts(self, frame_idx):
        ts = frame_idx / self.fps
        return f"{int(ts // 3600):02d}:{int(ts % 3600 // 60):02d}:{ts % 60:06.3f}"

    def log(self, frame_idx, pid, event, detail="", confidence=1.0):
        t = self.tracks.get(pid)
        state = t.state.value if t is not None else UNKNOWN
        self.events.append({
            "timestamp": self._ts(frame_idx),
            "person_id": pid,
            "event": event,
            "headcount": self.occupancy,
            "state": state,
            "confidence": round(confidence, 2),
            "detail": detail,
        })

    # ------------------------------------------------------------------ #
    # group association
    # ------------------------------------------------------------------ #
    def _assign_group(self, frame_idx, pos, pid):
        for f0, pos0, pid0 in self.recent_entries:
            if frame_idx - f0 > self.group_window or pid0 == pid:
                continue
            if math.hypot(pos[0] - pos0[0], pos[1] - pos0[1]) <= self.group_radius:
                other = self.tracks.get(pid0)
                if other is None:
                    continue
                gid = other.group_id
                if gid is None:
                    gid = self.next_gid
                    self.next_gid += 1
                    other.group_id = gid
                self.tracks[pid].group_id = gid
                return gid
        gid = self.next_gid
        self.next_gid += 1
        self.tracks[pid].group_id = gid
        return gid

    # ------------------------------------------------------------------ #
    # occupancy-verified events
    # ------------------------------------------------------------------ #
    def _verify_entry(self, t, frame_idx, detail):
        reentry = t.exit_count > 0
        t.inside = True
        t.seeded = False
        t.state = State.ENTERED
        t.confidence = 0.9 if detail.startswith("committed") else 0.6
        t.entry_count += 1
        t.occlusion_logged = False
        self.occupancy += 1
        self.peak = max(self.peak, self.occupancy)
        self.entries += 1
        if reentry:
            self.reentries += 1
        else:
            self.unique_visitors += 1
        self._assign_group(frame_idx, t.anchor(), t.pid)
        t.last_event = Event.REENTRY.value if reentry else Event.ENTRY.value
        t.last_event_frame = frame_idx
        self.recent_entries.append((frame_idx, t.anchor(), t.pid))
        self.recent_crossings.append({"frame": frame_idx, "kind": "IN",
                                      "pos": t.anchor(), "pid": t.pid,
                                      "appearance": not detail.startswith("committed")})
        self.log(frame_idx, t.pid, t.last_event, detail=detail, confidence=t.confidence)

    def _verify_exit(self, t, frame_idx):
        t.inside = False
        t.seeded = False
        t.state = State.EXIT_CONFIRMED
        t.confidence = 0.9
        t.exit_count += 1
        self.occupancy = max(0, self.occupancy - 1)
        self.exits += 1
        t.last_event = Event.EXIT.value
        t.last_event_frame = frame_idx
        t.exit_anchor = t.anchor()
        t.occlusion_logged = False
        self.recent_crossings.append({"frame": frame_idx, "kind": "OUT",
                                      "pos": t.anchor(), "pid": t.pid,
                                      "appearance": False})
        self.log(frame_idx, t.pid, Event.EXIT.value,
                 detail="committed-outward-crossing", confidence=t.confidence)

    # ------------------------------------------------------------------ #
    # appearance-counted entry (person the gate never witnessed crossing)
    # ------------------------------------------------------------------ #
    def appearance_entry(self, frame_idx, pos, ext_ids=None, confidence=0.6):
        """Verify an entry for a person the gate never saw cross.

        Prefer the track the appearance cluster was actually built from, so the
        person keeps one identity across the rest of the clip. Otherwise fall
        back to the nearest track that is not yet counted, and only as a last
        resort create a synthetic person.
        """
        best = None
        for ext in (ext_ids or ()):
            t = self.tracks.get(self.by_ext.get(ext))
            if t is not None and not t.inside:
                best = t
                break
        if best is None:
            bd = 120.0
            for t in self.tracks.values():
                if t.inside or not t.trail:
                    continue
                if t.state not in (State.UNKNOWN, State.APPROACHING_ENTRY):
                    continue
                dd = math.hypot(t.anchor()[0] - pos[0], t.anchor()[1] - pos[1])
                if dd < bd:
                    best, bd = t, dd
        if best is None:
            best = PersonTrack(self.next_pid, frame_idx)
            best.trail.append(tuple(pos))
            best.last_side = self.inside_sign
            self.tracks[best.pid] = best
            self.next_pid += 1
        self._verify_entry(best, frame_idx, detail="appearance-no-crossing")
        return best.pid

    # ------------------------------------------------------------------ #
    # re-identification
    # ------------------------------------------------------------------ #
    def _in_strip(self, x, d):
        margin = 15.0
        return (d >= self.strip_d_min and
                self.strip_lo - margin <= x <= self.strip_hi + margin)

    def _match_ghost(self, frame_idx, box, sig):
        """Propose a ghost for a detection that lost its external ID.

        Inside ghosts (OCCLUDED / POSSIBLY_EXITED) must reappear near where they
        vanished; a verified exit only allows a match near the exit point.  The
        appearance signature is the proposal, position/time is the corroboration.
        Returns (track, similarity) or None.
        """
        if sig is None:
            return None
        best, bs = None, self.reid_thr
        cx, cy = (box[0] + box[2]) / 2.0, box[3]
        for t in self.tracks.values():
            if t.state not in (State.OCCLUDED, State.POSSIBLY_EXITED,
                               State.EXIT_CONFIRMED):
                continue
            if frame_idx - t.last_seen > self.reid_gap:
                continue
            if t.state == State.EXIT_CONFIRMED:
                if t.exit_anchor is None:
                    continue
                ex, ey = t.exit_anchor
                if math.hypot(cx - ex, cy - ey) > self.reid_radius:
                    continue
            else:
                ax, ay = t.anchor()
                if math.hypot(cx - ax, cy - ay) > self.reid_radius:
                    continue
                # A person materialising inside the door strip cannot be the
                # same identity as someone who vanished deep in the store: the
                # interior ghost is an occlusion, the strip detection is an
                # entrant. Let the appearance layer count the entrant instead of
                # silently absorbing them into an existing identity.
                d_new = signed_distance((cx, cy), self.p1, self.p2)
                d_old = signed_distance((ax, ay), self.p1, self.p2)
                if self._in_strip(cx, d_new) and not self._in_strip(ax, d_old):
                    continue
            s = similarity(sig, t.sig)
            if s < bs:
                continue
            best, bs = t, s
        return (best, bs) if best is not None else None

    # ------------------------------------------------------------------ #
    # position helpers
    # ------------------------------------------------------------------ #
    def _resume(self, t, frame_idx, detail):
        """Reattach a previously occluded person.

        Short occlusions are routine and stay out of the event log; only a
        person who had already been reported UNCERTAIN produces a REAPPEARED
        row (that is the recovery of a recorded uncertainty).
        """
        was_uncertain = t.state == State.POSSIBLY_EXITED
        t.state = State.REAPPEARED
        t.reappeared_until = frame_idx + int(2.0 * self.fps)
        t.occlusion_logged = False
        if was_uncertain:
            t.confidence = 0.8
            self.log(frame_idx, t.pid, Event.REAPPEARED.value, detail=detail,
                     confidence=0.8)

    # ------------------------------------------------------------------ #
    # main per-frame ingest
    # ------------------------------------------------------------------ #
    def ingest(self, frame_idx, dets):
        """dets: list of dicts {ext_id, box, sig}."""
        for det in dets:
            ext = det["ext_id"]
            box = det["box"]
            sig = det.get("sig")
            pid = self.by_ext.get(ext)
            t = self.tracks.get(pid) if pid is not None else None

            if t is None:
                match = self._match_ghost(frame_idx, box, sig)
                if match is not None:
                    t, score = match
                    self.by_ext[ext] = t.pid
                    t.ext_id = ext
                    t.observe(frame_idx, box, sig)
                    if t.state == State.EXIT_CONFIRMED:
                        d = signed_distance(t.anchor(), self.p1, self.p2)
                        if d * self.inside_sign > 0 and \
                                frame_idx - t.last_event_frame >= self.min_gap:
                            self._verify_entry(
                                t, frame_idx,
                                detail=f"reid-reentry-inside sim={score:.2f}")
                        else:
                            # identity is reattached, but a new entry still needs
                            # inward evidence (or the flicker gap to elapse)
                            t.state = State.APPROACHING_ENTRY
                            t.confidence = 0.6
                    else:
                        self._resume(
                            t, frame_idx,
                            f"reid-occlusion-recovery sim={score:.2f}")
                    continue
                t = PersonTrack(self.next_pid, frame_idx)
                self.tracks[t.pid] = t
                self.next_pid += 1
                self.by_ext[ext] = t.pid
                t.ext_id = ext

            resumed = t.inside and t.missed > 2 and t.state in (
                State.OCCLUDED, State.POSSIBLY_EXITED)
            t.observe(frame_idx, box, sig)
            if resumed:
                # same external ID came back: identity was never lost, so this
                # is an occlusion recovery, not an entry.
                self._resume(t, frame_idx, "track-resumed-after-occlusion")

            a = t.anchor()
            d = signed_distance(a, self.p1, self.p2)

            # initialise the committed side only once the person is clearly on
            # one side; inside the band the side is unknown (prevents flicker)
            if t.last_side == 0:
                if d * self.inside_sign >= self.commit:
                    t.last_side = self.inside_sign
                elif d * self.inside_sign <= -self.commit:
                    t.last_side = -self.inside_sign
                if t.last_side == 0:
                    continue

            if t.state in (State.UNKNOWN, State.APPROACHING_ENTRY) and not t.inside:
                if d * self.inside_sign <= -self.commit:
                    t.state = State.APPROACHING_ENTRY
                elif d * self.inside_sign >= self.commit * 2:
                    t.state = State.UNKNOWN

                # If tracking began after warm-up and the person remains
                # clearly inside, the gate crossing may have been hidden by
                # another person or an occlusion. Count this late detection
                # once it persists, rather than leaving it uncounted.
                if (t.state == State.UNKNOWN
                        and frame_idx - t.first_seen + 1 >= self.late_entry_frames
                        and t.first_seen > self.warmup_frames
                        and t.trail
                        and abs(signed_distance(
                            t.trail[0], self.p1, self.p2)) <=
                        self.late_entry_distance
                        and abs(t.anchor()[0] - t.trail[0][0]) <= 45.0):
                    self._verify_entry(
                        t, frame_idx, detail="late-detected-inside")
                    continue

            s = 0
            if d >= self.commit:
                s = 1
            elif d <= -self.commit:
                s = -1
            if s == 0 or s == t.last_side:
                continue
            if (frame_idx - t.last_event_frame) < self.min_gap:
                # flicker guard: keep the committed side so a suppressed jitter
                # does not swallow the next genuine crossing
                continue
            inward = (s == self.inside_sign)
            if inward and t.inside:
                t.last_side = s             # already counted inside
                continue
            if (not inward) and (not t.inside):
                t.last_side = s             # already counted outside
                continue
            t.last_side = s
            if inward:
                self._verify_entry(t, frame_idx, detail="committed-inward-crossing")
            else:
                self._verify_exit(t, frame_idx)

    # ------------------------------------------------------------------ #
    # per-frame lifecycle
    # ------------------------------------------------------------------ #
    def tick(self, frame_idx):
        for t in self.tracks.values():
            missed = frame_idx - t.last_seen
            t.missed = missed

            if missed == 0:
                # observed this frame: settle momentary states and gather evidence
                if t.state == State.REAPPEARED and frame_idx >= t.reappeared_until:
                    t.state = State.INSIDE if t.inside else State.EXIT_CONFIRMED
                elif t.inside and t.state == State.ENTERED and \
                        frame_idx > t.last_event_frame:
                    t.state = State.INSIDE
                if self.staff_zone and t.inside:
                    if in_polygon(t.anchor(), self.staff_zone):
                        t.staff_dwell += 1
                    else:
                        t.staff_dwell = 0
                    if t.staff_dwell >= self.staff_dwell_frames and \
                            t.classification != "STAFF":
                        t.classification = "STAFF"
                        t.confidence = 0.8
                        self.log(frame_idx, t.pid, Event.UNCERTAIN.value,
                                 detail="staff-classified-dwell-in-staff-zone",
                                 confidence=0.8)
                continue

            if missed <= 2:
                continue

            if t.inside:
                if missed <= self.hold:
                    if t.state in (State.ENTERED, State.INSIDE, State.REAPPEARED):
                        t.state = State.OCCLUDED
                elif t.state != State.POSSIBLY_EXITED:
                    t.state = State.POSSIBLY_EXITED
                    t.confidence = 0.4
                    if not t.occlusion_logged:
                        t.occlusion_logged = True
                        self.log(frame_idx, t.pid, Event.UNCERTAIN.value,
                                 detail="missing-inside-no-exit-evidence",
                                 confidence=0.4)
                # occupancy retained: NEVER decrement on disappearance

    # ------------------------------------------------------------------ #
    # seeding / metrics
    # ------------------------------------------------------------------ #
    def seed_initial_occupancy(self, frame_idx, pids):
        """People visibly inside at the end of warm-up: initial occupancy.

        They are inside before the clip can evidence a crossing, so they are
        counted in occupancy but can never produce an ENTRY event.
        """
        for pid in pids:
            t = self.tracks.get(pid)
            if t is None or t.inside:
                continue
            t.inside = True
            t.seeded = True
            t.state = State.INSIDE
            t.confidence = 0.6
            t.last_event = None
            self.occupancy += 1
            self.seeded_initial += 1
            self.seeded_pids.append(pid)
        self.peak = max(self.peak, self.occupancy)

    @property
    def tracks_total(self):
        return len(self.tracks)

    def visible_inside_count(self, dets):
        """Detector-visible population inside the line right now (a measurement)."""
        n = 0
        for det in dets:
            x1, y1, x2, y2 = det["box"]
            a = ((x1 + x2) / 2.0, y2)
            if signed_distance(a, self.p1, self.p2) * self.inside_sign > 0:
                n += 1
        return n

    def metrics(self):
        inside = [t for t in self.tracks.values() if t.inside]
        staff = [t for t in inside if t.classification == "STAFF"]
        unknown = [t for t in inside if t.classification == UNKNOWN]
        occluded = sum(1 for t in inside if t.state == State.OCCLUDED)
        possibly = sum(1 for t in inside if t.state == State.POSSIBLY_EXITED)
        avg_conf = (sum(t.confidence for t in inside) / len(inside)) if inside else 0.0
        return {
            "occupancy_confirmed": self.occupancy,
            "seeded": self.seeded_initial > 0,
            "peak_occupancy": self.peak,
            "entries_total": self.entries,
            "exits_total": self.exits,
            "reentries_total": self.reentries,
            "unique_visitors": self.unique_visitors,
            "initial_seeded": self.seeded_initial,
            "seeded_pids": list(self.seeded_pids),
            "staff_inside": len(staff),
            "unknown_inside": len(unknown),
            "customers_estimate": len(inside) - len(staff),
            "customers_reliable": len(unknown) == 0,
            "occluded_now": occluded,
            "possibly_exited_now": possibly,
            "uncertain_now": occluded + possibly,
            "active_ids": len(self.by_ext),
            "tracks_total": len(self.tracks),
            "avg_confidence": round(avg_conf, 2),
        }
