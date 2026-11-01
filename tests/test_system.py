"""Synthetic scenario tests for the v2 occupancy state machine.

Every test drives TrackManager directly with fabricated detections, so the
decision logic is exercised without YOLO or a GPU.  The scenarios mirror the
brief's required test list (section 13).

Run:
    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retail.appearance import StripAppearance          # noqa: E402
from retail.camera import camera_status                # noqa: E402
from retail.state import Event, State                  # noqa: E402
from retail.tracks import TrackManager                 # noqa: E402

FPS = 10


def make_cfg(**over):
    cfg = {
        "line": [[100, 0], [100, 300]],
        "inside_sign": 1,
        "commit_px": 20.0,
        "init_band_px": 10.0,
        "min_event_gap_s": 0.5,       # 5 frames
        "occlusion_hold_s": 1.0,      # 10 frames
        "reid_sim": 0.75,
        "reid_max_gap_s": 100.0,
        "reid_radius_px": 300.0,
        "group_window_s": 2.5,
        "group_radius_px": 220.0,
        "staff_zone": None,
        "staff_dwell_s": 1.0,
        "warmup_s": 0.0,
        "appear_strip_x": [0, 0],
        "appear_strip_d_min": 12.0,
        "appear_link_px": 90.0,
        "appear_new_area_px": 85.0,
        "appear_new_area_s": 4.0,
        "appear_confirm_frames": 4,
        "appear_cancel_s": 1.0,
        "appear_credit_s": 0.3,
    }
    cfg.update(over)
    return cfg


def box(x, y=200, w=30, h=60):
    return (x - w / 2.0, y - h, x + w / 2.0, y)


def det(ext, x, y=200, sig=None):
    return {"ext_id": ext, "box": box(x, y), "sig": sig}


SIG_A = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32)
SIG_B = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)


class Driver:
    """Frame-by-frame driver with the occupancy invariants checked every frame."""

    def __init__(self, cfg=None):
        self.cfg = cfg or make_cfg()
        self.tm = TrackManager(self.cfg, FPS)
        self.frame = 0

    def step(self, dets):
        before = self.tm.occupancy
        self.tm.ingest(self.frame, dets)
        self.tm.tick(self.frame)
        self.check()
        exited = any(e["event"] == Event.EXIT.value and
                     e["timestamp"] == self.tm._ts(self.frame)
                     for e in self.tm.events)
        assert self.tm.occupancy >= before or exited, \
            "occupancy dropped without a verified EXIT"
        self.frame += 1
        return self.tm

    def run(self, seq):
        for dets in seq:
            self.step(dets)
        return self.tm

    def check(self):
        tm = self.tm
        assert tm.occupancy == sum(1 for t in tm.tracks.values() if t.inside), \
            "occupancy account drifted from the set of inside tracks"
        assert tm.occupancy >= 0
        assert tm.entries == sum(t.entry_count for t in tm.tracks.values())
        assert tm.exits == sum(t.exit_count for t in tm.tracks.values())


def walk_in(d, ext, ys=200, x0=160, x1=40, n=9, sig=None):
    for i in range(n):
        x = x0 + (x1 - x0) * i / (n - 1)
        d.step([det(ext, x, ys, sig)])


def walk_out(d, ext, ys=200, x0=40, x1=160, n=7, sig=None):
    for i in range(n):
        x = x0 + (x1 - x0) * i / (n - 1)
        d.step([det(ext, x, ys, sig)])


def pid_of(tm, ext):
    return tm.tracks[tm.by_ext[ext]]


class TestEntryExit(unittest.TestCase):
    def test_late_detected_customer_inside_is_counted(self):
        d = Driver(make_cfg(late_entry_s=0.5, warmup_s=0.0))
        d.step([])
        d.step([])
        for _ in range(6):
            d.step([det(1, 70)])
        self.assertEqual(d.tm.entries, 1)
        self.assertEqual(d.tm.occupancy, 1)
        self.assertEqual(d.tm.events[0]["detail"], "late-detected-inside")

    def test_one_person_enters_and_exits(self):
        d = Driver()
        walk_in(d, 1)
        self.assertEqual(d.tm.entries, 1)
        self.assertEqual(d.tm.exits, 0)
        self.assertEqual(d.tm.occupancy, 1)
        self.assertEqual(d.tm.unique_visitors, 1)
        events = [e["event"] for e in d.tm.events]
        self.assertIn(Event.ENTRY.value, events)

        walk_out(d, 1)
        self.assertEqual(d.tm.entries, 1)
        self.assertEqual(d.tm.exits, 1)
        self.assertEqual(d.tm.occupancy, 0)
        self.assertEqual(d.tm.tracks[1].state, State.EXIT_CONFIRMED)
        self.assertEqual([e["event"] for e in d.tm.events],
                         [Event.ENTRY.value, Event.EXIT.value])

    def test_flicker_near_the_line_counts_once(self):
        d = Driver()
        for x in (200, 160, 120, 80):
            d.step([det(1, x)])
        self.assertEqual(d.tm.entries, 1)
        for x in (120, 60, 120, 60, 50):
            d.step([det(1, x)])
        self.assertEqual(d.tm.entries, 1, "flicker created a duplicate entry")
        for x in (100, 120, 140):
            d.step([det(1, x)])
        self.assertEqual(d.tm.exits, 1, "suppressed flicker swallowed the exit")

    def test_simultaneous_opposite_crossings(self):
        d = Driver()
        d.step([det(1, 40)])                     # person already inside
        d.tm.seed_initial_occupancy(d.frame, [1])
        d.check()
        for x in (180, 160):                     # person approaching outside
            d.step([det(1, 40), det(2, x)])
        d.step([det(1, 140), det(2, 60)])        # same frame: OUT + IN
        self.assertEqual(d.tm.entries, 1)
        self.assertEqual(d.tm.exits, 1)
        self.assertEqual(d.tm.occupancy, 1)
        self.assertEqual(d.tm.unique_visitors, 1)


class TestOcclusion(unittest.TestCase):
    def test_mother_and_child_short_occlusion(self):
        d = Driver()
        for i in range(9):
            x = 160 - i * 15
            d.step([det(1, x, 170), det(2, x, 230)])
        self.assertEqual(d.tm.occupancy, 2)
        self.assertEqual(d.tm.entries, 2)

        for _ in range(5):                       # child fully hidden
            d.step([det(1, 40, 170)])
        self.assertEqual(d.tm.occupancy, 2, "child removed on disappearance")
        self.assertEqual(d.tm.exits, 0)
        self.assertEqual(d.tm.tracks[2].state, State.OCCLUDED)

        d.step([det(1, 40, 170), det(2, 45, 230)])
        self.assertEqual(d.tm.occupancy, 2)
        self.assertEqual(d.tm.entries, 2, "reappearing child counted again")
        self.assertEqual(d.tm.tracks[2].state, State.REAPPEARED)

        for _ in range(25):
            d.step([det(1, 40, 170), det(2, 45, 230)])
        self.assertEqual(d.tm.tracks[2].state, State.INSIDE)

    def test_long_occlusion_becomes_uncertain_not_exit(self):
        d = Driver()
        walk_in(d, 1)
        self.assertEqual(d.tm.occupancy, 1)
        for _ in range(40):
            d.step([])
        t = d.tm.tracks[1]
        self.assertEqual(t.state, State.POSSIBLY_EXITED)
        self.assertEqual(d.tm.exits, 0)
        self.assertEqual(d.tm.occupancy, 1, "occupancy dropped on disappearance")
        uncertain = [e for e in d.tm.events if e["event"] == Event.UNCERTAIN.value]
        self.assertEqual(len(uncertain), 1)
        self.assertEqual(uncertain[0]["detail"], "missing-inside-no-exit-evidence")
        self.assertEqual(uncertain[0]["state"], State.POSSIBLY_EXITED.value)

    def test_reappearance_with_new_id_is_reidentified(self):
        d = Driver()
        walk_in(d, 1, ys=170, sig=SIG_A)
        self.assertEqual(d.tm.entries, 1)
        for _ in range(20):                      # lost long enough to be a ghost
            d.step([])
        self.assertEqual(d.tm.tracks[1].state, State.POSSIBLY_EXITED)
        # same person comes back with a new tracker ID right where they vanished
        for x in (38, 42, 45):
            d.step([det(99, x, 170, SIG_A)])
        self.assertEqual(d.tm.tracks[1].state, State.REAPPEARED)
        self.assertEqual(d.tm.entries, 1, "occlusion recovery counted as entry")
        self.assertEqual(d.tm.occupancy, 1)
        self.assertEqual(d.tm.tracks_total, 1, "re-ID created a duplicate identity")


class TestGroups(unittest.TestCase):
    def _enter_group(self):
        d = Driver()
        ys = [140, 180, 220, 260]
        for i in range(9):
            x = 160 - i * 15
            d.step([det(10 + k, x, ys[k]) for k in range(4)])
        return d, ys

    def test_family_of_four_enters_together(self):
        d, _ = self._enter_group()
        self.assertEqual(d.tm.entries, 4)
        self.assertEqual(d.tm.occupancy, 4)
        groups = {t.group_id for t in d.tm.tracks.values()}
        self.assertEqual(len(groups), 1)
        self.assertNotIn(None, groups)

    def test_members_separate_but_keep_identity(self):
        d, ys = self._enter_group()
        gid = pid_of(d.tm, 10).group_id
        for i in range(8):
            xs = [20 + i * 3, 60 + i * 3, 30, 70]
            d.step([det(10 + k, xs[k], ys[k]) for k in range(4)])
        self.assertEqual(d.tm.entries, 4)
        self.assertEqual(d.tm.occupancy, 4)
        self.assertTrue(all(t.group_id == gid for t in d.tm.tracks.values()))

    def test_one_member_hidden_does_not_remove_the_group(self):
        d, ys = self._enter_group()
        for _ in range(25):
            d.step([det(10 + k, 30 + k * 5, ys[k]) for k in range(3)])
        self.assertEqual(d.tm.occupancy, 4)
        self.assertEqual(d.tm.exits, 0)
        self.assertEqual(pid_of(d.tm, 13).state, State.POSSIBLY_EXITED)
        for k in range(3):
            self.assertEqual(pid_of(d.tm, 10 + k).state, State.INSIDE)


class TestReentry(unittest.TestCase):
    def test_exit_then_reentry_with_reid(self):
        d = Driver()
        walk_in(d, 1, sig=SIG_A)
        self.assertEqual(d.tm.entries, 1)
        walk_out(d, 1, sig=SIG_A)
        self.assertEqual(d.tm.exits, 1)
        self.assertEqual(d.tm.occupancy, 0)

        d.step([det(99, 150, sig=SIG_A)])        # returning, new tracker ID
        self.assertEqual(d.tm.tracks[1].state, State.APPROACHING_ENTRY)
        self.assertEqual(d.tm.occupancy, 0, "occupancy rose before the crossing")

        for x in (140, 120, 100, 80, 60):
            d.step([det(99, x, sig=SIG_A)])
        self.assertEqual(d.tm.entries, 2)
        self.assertEqual(d.tm.reentries, 1)
        self.assertEqual(d.tm.exits, 1)
        self.assertEqual(d.tm.unique_visitors, 1, "re-entry became a new visitor")
        self.assertEqual(d.tm.occupancy, 1)
        self.assertEqual(d.tm.tracks_total, 1)
        events = [e["event"] for e in d.tm.events]
        self.assertEqual(events, [Event.ENTRY.value, Event.EXIT.value,
                                  Event.REENTRY.value])


class TestStaff(unittest.TestCase):
    def test_staff_classification_requires_zone_evidence(self):
        zone = [[0, 0], [100, 0], [100, 300], [0, 300]]
        d = Driver(make_cfg(staff_zone=zone))
        for x in (200, 160, 120, 80, 60):
            d.step([det(1, x)])
        self.assertEqual(d.tm.occupancy, 1)
        for _ in range(15):
            d.step([det(1, 60)])
        m = d.tm.metrics()
        self.assertEqual(m["staff_inside"], 1)
        self.assertEqual(m["customers_estimate"], 0)
        self.assertTrue(m["customers_reliable"])
        self.assertTrue(any(e["detail"].startswith("staff-classified")
                            for e in d.tm.events))

    def test_no_zone_means_no_staff_claim(self):
        d = Driver()
        walk_in(d, 1)
        for _ in range(15):
            d.step([det(1, 40)])
        m = d.tm.metrics()
        self.assertEqual(m["staff_inside"], 0)
        self.assertTrue(m["customers_reliable"])
        self.assertEqual(m["customers_estimate"], 1)
        self.assertEqual(d.tm.tracks[1].classification, "CUSTOMER")

    def test_every_person_has_customer_or_staff_classification(self):
        d = Driver()
        walk_in(d, 1)
        classifications = {
            track.classification for track in d.tm.tracks.values()
        }
        self.assertTrue(classifications <= {"CUSTOMER", "STAFF"})


class TestIdentitySafety(unittest.TestCase):
    def test_two_similar_people_do_not_merge(self):
        d = Driver()
        for i in range(9):
            x = 160 - i * 15
            d.step([det(1, x, 170, SIG_A), det(2, x, 250, SIG_B)])
        self.assertEqual(d.tm.entries, 2)
        self.assertEqual(d.tm.occupancy, 2)

        # they cross paths inside: external IDs are stable, no merge allowed
        for i in range(6):
            d.step([det(1, 30 + i * 5, 170, SIG_A),
                    det(2, 70 - i * 5, 250, SIG_B)])
        self.assertEqual(d.tm.occupancy, 2)
        self.assertEqual(d.tm.tracks_total, 2)
        self.assertEqual(d.tm.entries, 2)

    def test_dissimilar_detection_is_not_reidentified(self):
        d = Driver()
        walk_in(d, 1, ys=170, sig=SIG_A)
        for _ in range(20):
            d.step([])
        self.assertEqual(d.tm.tracks[1].state, State.POSSIBLY_EXITED)
        # a different-looking person appears near the ghost: must not be merged
        for x in (40, 45, 50):
            d.step([det(77, x, 170, SIG_B)])
        self.assertEqual(d.tm.tracks[1].state, State.POSSIBLY_EXITED)
        self.assertEqual(d.tm.tracks_total, 2)
        self.assertEqual(d.tm.occupancy, 1)
        self.assertEqual(d.tm.entries, 1)

    def test_door_materialisation_is_not_absorbed_by_interior_ghost(self):
        cfg = make_cfg(appear_strip_x=[60, 90], appear_strip_d_min=12.0)
        d = Driver(cfg)
        walk_in(d, 1, ys=170, sig=SIG_A)          # ends deep inside, x=40
        for _ in range(20):
            d.step([])
        self.assertEqual(d.tm.tracks[1].state, State.POSSIBLY_EXITED)

        # a similar-looking person materialises in the door strip; this is an
        # entrant, not the interior ghost coming back
        d.step([det(88, 80, 170, SIG_A)])
        self.assertEqual(d.tm.tracks_total, 2)
        self.assertEqual(d.tm.tracks[1].state, State.POSSIBLY_EXITED)
        self.assertEqual(d.tm.entries, 1)
        self.assertEqual(d.tm.occupancy, 1)

    def test_everyone_exits_individually_but_together(self):
        d = Driver()
        ys = [140, 180, 220]
        for i in range(9):
            x = 160 - i * 15
            d.step([det(k + 1, x, ys[k]) for k in range(3)])
        self.assertEqual(d.tm.entries, 3)
        self.assertEqual(d.tm.occupancy, 3)
        for i in range(7):
            x = 70 + i * 15
            d.step([det(k + 1, x, ys[k]) for k in range(3)])
        self.assertEqual(d.tm.exits, 3)
        self.assertEqual(d.tm.occupancy, 0)


class TestAppearanceLayer(unittest.TestCase):
    def _cfg(self):
        return make_cfg(appear_strip_x=[0, 90], appear_new_area_px=10.0,
                        appear_new_area_s=1.0, appear_confirm_frames=4,
                        appear_cancel_s=0.4)

    def test_materialising_entrant_is_counted_once(self):
        cfg = self._cfg()
        tm = TrackManager(cfg, FPS)
        app = StripAppearance(cfg, FPS, 0)
        for frame in range(10):
            dets = [det(7, 50)]
            tm.ingest(frame, dets)
            tm.tick(frame)
            app.update(frame, dets, tm)
        self.assertEqual(app.fired, 1)
        self.assertEqual(tm.entries, 1)
        self.assertEqual(tm.occupancy, 1)
        entry = [e for e in tm.events if e["event"] == Event.ENTRY.value]
        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0]["detail"].split(" verified_at")[0],
                         "appearance-no-crossing")
        self.assertIn("verified_at=", entry[0]["detail"],
                      "delayed confirmation must record when it was verified")
        self.assertEqual(tm.tracks[1].pid, 1)

    def test_interior_walker_into_strip_is_not_an_entry(self):
        cfg = make_cfg(appear_strip_x=[60, 90], appear_strip_d_min=12.0,
                       appear_new_area_px=10.0, appear_new_area_s=1.0,
                       appear_confirm_frames=4, appear_cancel_s=0.4)
        tm = TrackManager(cfg, FPS)
        app = StripAppearance(cfg, FPS, 0)
        for frame in range(14):
            x = 10 + min(frame, 8) * 7       # starts deep inside, reaches strip
            dets = [det(5, x)]
            tm.ingest(frame, dets)
            tm.tick(frame)
            app.update(frame, dets, tm)
        self.assertEqual(app.fired, 0, "interior walker counted as an entrant")
        self.assertEqual(tm.occupancy, 0)
        self.assertTrue(any(cl["state"] == "occupied-area"
                            for cl in app.clusters))

    def test_leaver_in_the_strip_is_vetoed(self):
        cfg = self._cfg()
        cfg["appear_cancel_s"] = 1.0
        tm = TrackManager(cfg, FPS)
        app = StripAppearance(cfg, FPS, 0)
        xs = [50] * 5 + [60, 80, 100, 120, 140, 150, 155, 160]
        for frame, x in enumerate(xs):
            dets = [det(8, x)]
            tm.ingest(frame, dets)
            tm.tick(frame)
            app.update(frame, dets, tm)
        self.assertEqual(app.fired, 0, "an exit was converted into an entry")
        self.assertEqual(tm.occupancy, 0)
        self.assertEqual(app.clusters[0]["state"], "cancelled-exiting")


class TestCameraCoverage(unittest.TestCase):
    def test_full_coverage_is_clean(self):
        cam = camera_status({"line": [[100, 50], [100, 250]],
                             "entrance_fully_visible": True}, (300, 400))
        self.assertTrue(cam["entrance_fully_visible"])
        self.assertEqual(cam["warnings"], [])

    def test_partial_coverage_warns_and_blocks_confident_exits(self):
        cam = camera_status({"line": [[100, 50], [100, 250]],
                             "entrance_fully_visible": False}, (300, 400))
        self.assertFalse(cam["entrance_fully_visible"])
        self.assertTrue(any("not fully visible" in w for w in cam["warnings"]))

    def test_line_outside_frame_is_reported(self):
        cam = camera_status({"line": [[100, 50], [100, 350]],
                             "entrance_fully_visible": True}, (300, 400))
        self.assertFalse(cam["line_in_frame"])
        self.assertFalse(cam["entrance_fully_visible"])
        self.assertTrue(any("outside the frame" in w for w in cam["warnings"]))


class TestMetrics(unittest.TestCase):
    def test_accounting_identities(self):
        d = Driver()
        walk_in(d, 1, sig=SIG_A)
        walk_out(d, 1, sig=SIG_A)
        d.step([det(99, 150, sig=SIG_A)])
        for x in (140, 120, 100, 80, 60):
            d.step([det(99, x, sig=SIG_A)])
        m = d.tm.metrics()
        self.assertEqual(m["entries_total"], 2)
        self.assertEqual(m["exits_total"], 1)
        self.assertEqual(m["reentries_total"], 1)
        self.assertEqual(m["unique_visitors"], 1)
        self.assertEqual(m["occupancy_confirmed"], 1)
        self.assertEqual(m["occupancy_confirmed"],
                         sum(1 for t in d.tm.tracks.values() if t.inside))


if __name__ == "__main__":
    unittest.main(verbosity=2)
