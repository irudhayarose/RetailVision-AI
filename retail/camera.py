"""Camera / entrance coverage checks.

The brief is explicit: if the entrance is not fully visible, disappearance must
never be treated as a confirmed exit.  These checks make that limitation visible
on the dashboard and in the summary instead of silently producing a confident
number.
"""


def camera_status(cfg, frame_shape):
    """Return {ok, line_in_frame, warnings} for the configured entrance line."""
    h, w = frame_shape[0], frame_shape[1]
    p1, p2 = cfg["line"]
    warnings = []

    configured = bool(cfg.get("entrance_fully_visible", True))
    line_in_frame = all(0 <= p[0] < w and 0 <= p[1] < h for p in (p1, p2))

    if not configured:
        warnings.append("entrance not fully visible: exits cannot be confirmed "
                        "from disappearance alone")
    if not line_in_frame:
        warnings.append("gate line extends outside the frame: crossings may be "
                        "missed")
    if not warnings and not configured:
        warnings.append("entrance coverage not configured")

    return {
        "entrance_fully_visible": configured and line_in_frame,
        "line_in_frame": line_in_frame,
        "warnings": warnings,
    }
