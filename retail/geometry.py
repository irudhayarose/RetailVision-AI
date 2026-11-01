"""Geometry helpers: signed distance to the entrance line, zone membership."""
import cv2
import numpy as np


def signed_distance(pt, p1, p2):
    """Signed perpendicular distance of `pt` from the directed line p1->p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    return (dx * (pt[1] - p1[1]) - dy * (pt[0] - p1[0])) / n


def in_polygon(pt, poly):
    if not poly:
        return False
    return cv2.pointPolygonTest(np.array(poly, dtype=np.float32),
                                (float(pt[0]), float(pt[1])), False) >= 0
