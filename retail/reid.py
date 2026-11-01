"""Lightweight appearance signatures for re-identification.

HSV 2-D histogram of the person crop. Cheap enough for CPU-only real time and
good enough to distinguish differently-dressed people; it is explicitly NOT a
learned re-ID embedding, so the matcher is used only to propose an association
and the decision is always corroborated by position/time (see TrackManager).
"""
import cv2
import numpy as np


def signature(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 20:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


def similarity(a, b):
    """1 - Bhattacharyya distance; 1.0 = identical."""
    if a is None or b is None:
        return 0.0
    return float(1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
