"""Dashboard: bounding boxes, states, occupancy panel and event ticker."""
import math

import cv2

from .state import Event, State

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE = (238, 238, 238)
GREY = (168, 168, 168)
DARK = (22, 22, 22)
ACCENT = (60, 210, 250)
GREEN = (90, 220, 90)
RED = (70, 70, 235)
AMBER = (40, 190, 250)
ORANGE = (40, 140, 250)
BLUE = (235, 180, 60)
CYAN = (230, 220, 60)
GREY_BOX = (175, 175, 175)

STATE_COLOR = {
    State.INSIDE: GREEN,
    State.ENTERED: GREEN,
    State.REAPPEARED: CYAN,
    State.OCCLUDED: AMBER,
    State.POSSIBLY_EXITED: ORANGE,
    State.EXIT_CONFIRMED: RED,
    State.APPROACHING_ENTRY: BLUE,
    State.UNKNOWN: GREY_BOX,
}

STATE_SHORT = {
    State.INSIDE: "INSIDE",
    State.ENTERED: "ENTERED",
    State.REAPPEARED: "REAPPEARED",
    State.OCCLUDED: "OCCLUDED",
    State.POSSIBLY_EXITED: "POSS-EXIT",
    State.EXIT_CONFIRMED: "EXITED",
    State.APPROACHING_ENTRY: "APPROACH",
    State.UNKNOWN: "UNKNOWN",
}

EVENT_COLOR = {
    Event.ENTRY.value: GREEN,
    Event.REENTRY.value: CYAN,
    Event.EXIT.value: RED,
    Event.UNCERTAIN.value: AMBER,
    Event.REAPPEARED.value: BLUE,
}


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def draw_entrance(frame, p1, p2, inside_sign, cam):
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    col = (40, 235, 255) if cam.get("entrance_fully_visible", True) else (0, 165, 255)
    cv2.line(frame, p1, p2, (0, 0, 0), 9, cv2.LINE_AA)
    cv2.line(frame, p1, p2, col, 4, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(frame, p, 7, col, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 7, (0, 0, 0), 1, cv2.LINE_AA)

    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    n = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / n * 36 * inside_sign, dx / n * 36 * inside_sign
    cv2.arrowedLine(frame, (int(mx), int(my)), (int(mx + nx), int(my + ny)),
                    GREEN, 3, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(frame, "ENTRY", (int(mx + nx) + 8, int(my + ny) + 5), FONT, 0.5,
                GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, "GATE", (p1[0] + 10, p1[1] - 6), FONT, 0.45, col, 1,
                cv2.LINE_AA)

    if not cam.get("entrance_fully_visible", True):
        cv2.putText(frame, "! ENTRANCE NOT FULLY VISIBLE", (p1[0] - 210, p2[1] + 26),
                    FONT, 0.55, ORANGE, 2, cv2.LINE_AA)


def draw_track(frame, t, frame_idx=None, fps=13.0):
    """Draw one person: state colour, ID, group, classification and trail."""
    if t.box is None:
        return
    col = STATE_COLOR.get(t.state, GREY_BOX)
    x1, y1, x2, y2 = (int(v) for v in t.box)

    if len(t.trail) > 1:
        pts = [[int(a), int(b)] for a, b in t.trail]
        for a, b in zip(pts[:-1], pts[1:]):
            # a jump larger than this is a re-attachment, not a walk: drawing
            # it would strew a false trail across the frame
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 90:
                continue
            cv2.line(frame, tuple(a), tuple(b), col, 2, cv2.LINE_AA)

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

    label = f"P{t.pid} {t.classification}"
    if t.group_id is not None:
        label += f" G{t.group_id}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 12, y1), col, -1)
    cv2.putText(frame, label, (x1 + 6, y1 - 6), FONT, 0.5, DARK, 1, cv2.LINE_AA)

    cx = (x1 + x2) // 2
    cv2.circle(frame, (cx, y2), 4, col, -1, cv2.LINE_AA)

    if frame_idx is not None and t.last_event and \
            frame_idx - t.last_event_frame <= int(0.9 * fps):
        flash(frame, t, frame_idx)


def _dashed_circle(frame, center, radius, colour, thickness=2):
    for a in range(0, 360, 30):
        cv2.ellipse(frame, center, (radius, radius), 0, a, a + 16, colour,
                    thickness, cv2.LINE_AA)


def draw_ghost(frame, t):
    """Occluded / possibly-exited person: predicted position, not a detection."""
    col = STATE_COLOR.get(t.state, AMBER)
    ax, ay = t.predicted_anchor()
    c = (int(ax), int(ay))
    _dashed_circle(frame, c, 18, col, 2)
    cv2.line(frame, (c[0] - 8, c[1]), (c[0] + 8, c[1]), col, 2, cv2.LINE_AA)
    cv2.line(frame, (c[0], c[1] - 8), (c[0], c[1] + 8), col, 2, cv2.LINE_AA)
    label = f"P{t.pid} {STATE_SHORT.get(t.state, 'HIDDEN')}"
    cv2.putText(frame, label, (c[0] - 46, c[1] - 26), FONT, 0.45, col, 1,
                cv2.LINE_AA)


def flash(frame, t, frame_idx):
    col = EVENT_COLOR.get(str(t.last_event), GREEN)
    ax, ay = t.anchor()
    txt = f"P{t.pid} {t.last_event}"
    (tw, th), _ = cv2.getTextSize(txt, FONT, 0.65, 2)
    x = max(8, min(int(ax) - tw // 2, frame.shape[1] - tw - 12))
    y = max(56, min(int(ay) - 52, frame.shape[0] - 12))
    cv2.rectangle(frame, (x - 8, y - th - 10), (x + tw + 8, y + 8), col, -1)
    cv2.putText(frame, txt, (x, y), FONT, 0.65, DARK, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
def _panel(frame, x, y, w, h, alpha=0.86):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (24, 24, 24), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (105, 105, 105), 1)


def draw_hud(frame, m, visible_now, fps_live, frame_idx, cam, events_tail,
             recent=None):
    x, y, w, h = 18, 16, 408, 424
    _panel(frame, x, y, w, h)

    seeded = m.get("seeded", True)
    cv2.putText(frame, "RETAIL OCCUPANCY", (x + 16, y + 30), FONT, 0.66, WHITE,
                1, cv2.LINE_AA)
    cv2.putText(frame,
                "verified-event account, not a raw box count" if seeded
                else "seeding initial occupancy...",
                (x + 16, y + 48), FONT, 0.36, GREY if seeded else AMBER, 1,
                cv2.LINE_AA)
    cv2.line(frame, (x + 16, y + 58), (x + w - 16, y + 58), (100, 100, 100), 1)

    reliable = m["customers_reliable"]
    cam_ok = cam.get("entrance_fully_visible", True)
    rows = [
        ("INSIDE (confirmed)", m["occupancy_confirmed"], GREEN),
        ("VISIBLE NOW (detector)", visible_now, ACCENT),
        ("CUSTOMERS" + ("" if reliable else " (unverified)"),
         m["customers_estimate"], GREEN if reliable else GREY),
        ("STAFF", m["staff_inside"] if reliable or m["staff_inside"] else "-",
         GREEN if reliable else GREY),
        ("ENTERED", m["entries_total"], GREEN),
        ("LEFT", m["exits_total"], RED),
        ("UNIQUE VISITORS", m["unique_visitors"], WHITE),
        ("RE-ENTRIES", m["reentries_total"], CYAN),
        ("INITIAL (seeded)", m["initial_seeded"], GREY),
        ("IDENTITIES", m["tracks_total"], WHITE),
        ("OCCLUDED", m["occluded_now"], AMBER),
        ("UNCERTAIN", m["possibly_exited_now"], ORANGE),
        ("TRACK CONFIDENCE", f"{m['avg_confidence']:.2f}", WHITE),
        ("CAMERA", "OK" if cam_ok else "LIMITED", GREEN if cam_ok else ORANGE),
    ]
    # before the warm-up boundary the account is not seeded yet: showing 0 would
    # read as "the shop is empty", which is the opposite of what the frame shows
    live_only = {"VISIBLE NOW (detector)", "STAFF", "CAMERA", "IDENTITIES"}
    yy = y + 82
    for label, value, col in rows:
        cv2.putText(frame, label, (x + 16, yy), FONT, 0.46, GREY, 1, cv2.LINE_AA)
        txt = str(value) if seeded or label in live_only else "--"
        (tw, _), _ = cv2.getTextSize(txt, FONT, 0.66, 1)
        cv2.putText(frame, txt, (x + w - 20 - tw, yy), FONT, 0.66, col, 1,
                    cv2.LINE_AA)
        yy += 23

    cv2.putText(frame, f"frame {frame_idx}  {fps_live:4.1f} fps", (x + 16, y + h - 10),
                FONT, 0.38, GREY, 1, cv2.LINE_AA)

    if cam.get("warnings"):
        wy = y + h + 18
        cv2.putText(frame, "CAMERA WARNING", (x + 2, wy), FONT, 0.42, ORANGE, 1,
                    cv2.LINE_AA)
        for i, msg in enumerate(cam["warnings"][:2]):
            cv2.putText(frame, msg[:58], (x + 2, wy + 18 + i * 16), FONT, 0.36,
                        ORANGE, 1, cv2.LINE_AA)

    draw_ticker(frame, events_tail,
                y + h + 54 if cam.get("warnings") else y + h + 16, recent)


def draw_ticker(frame, events, top, recent=None):
    x, w, h = 18, 408, 132
    _panel(frame, x, top, w, h, alpha=0.6)
    cv2.putText(frame, "EVENT LOG (verified)", (x + 14, top + 22), FONT, 0.44,
                WHITE, 1, cv2.LINE_AA)
    if recent is not None:
        txt = f"last 60s  +{recent[0]} in  -{recent[1]} out"
        (tw, _), _ = cv2.getTextSize(txt, FONT, 0.36, 1)
        cv2.putText(frame, txt, (x + w - 14 - tw, top + 22), FONT, 0.36,
                    ACCENT, 1, cv2.LINE_AA)
    cv2.line(frame, (x + 14, top + 30), (x + w - 14, top + 30), (95, 95, 95), 1)
    yy = top + 48
    for ev in events[-5:]:
        col = EVENT_COLOR.get(ev["event"], GREY)
        txt = (f"{ev['timestamp']}  P{ev['person_id']}  {ev['event']}"
               f"  inside={ev['headcount']}")
        cv2.putText(frame, txt, (x + 14, yy), FONT, 0.38, col, 1, cv2.LINE_AA)
        yy += 17
    if not events:
        cv2.putText(frame, "no verified events yet", (x + 14, yy), FONT, 0.38,
                    GREY, 1, cv2.LINE_AA)
