

import json
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config.json")

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = json.load(f)

video = os.path.join(HERE, cfg["video"])
cap = cv2.VideoCapture(video)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit(f"cannot read {video}")

points = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))


win = "click the door threshold: 2 points  |  ENTER = save, r = reset, ESC = quit"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1100, 620)
cv2.setMouseCallback(win, on_mouse)

while True:
    view = frame.copy()
    for i, p in enumerate(points):
        cv2.circle(view, p, 8, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(view, f"P{i + 1} {p[0]},{p[1]}", (p[0] + 12, p[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if len(points) == 2:
        cv2.line(view, points[0], points[1], (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(view, "ENTER to save", (30, 60), cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(view, "click 2 points along the doorway", (30, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(win, view)
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        print("cancelled, config unchanged")
        break
    if key == ord("r"):
        points.clear()
    if key in (13, 10, ord("s")):
        if len(points) == 2:
            cfg["line"] = [list(points[0]), list(points[1])]
            with open(CONFIG, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"saved line {cfg['line']} -> {CONFIG}")
            break
        print("need 2 points first")

cv2.destroyAllWindows()
