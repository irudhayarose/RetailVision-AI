"""CLI entry point for the v2 retail occupancy system.

Usage
    python store_monitor.py                     # full clip, render + logs
    python store_monitor.py --seconds 30        # quick verification render
    python store_monitor.py --no-video          # metrics only (no output video)
    python store_monitor.py --out-dir output    # where artifacts are written
"""
import argparse
import json
import os

from retail.pipeline import run

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "video": "video.f398.mp4",
    "model": "yolov8n.pt",
    "output_dir": "output",
    "line": [[1080, 238], [1080, 715]],
    "inside_sign": 1,
    "conf": 0.35,
    "imgsz": 640,
    "min_box_h": 45,
    "warmup_s": 3.0,
}


def load_config(path):
    cfg = dict(DEFAULTS)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Retail occupancy monitor (v2)")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--seconds", type=float, default=None,
                    help="only process the first N seconds")
    ap.add_argument("--no-video", action="store_true",
                    help="skip rendering the annotated video")
    ap.add_argument("--out-dir", default=None,
                    help="artifact directory (default: config output_dir)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg, HERE, seconds=args.seconds, render=not args.no_video,
        out_dir=args.out_dir, progress=True)


if __name__ == "__main__":
    main()
