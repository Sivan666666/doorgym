#!/usr/bin/env python3
"""Measure dual-D435 realtime camera FPS with door_act_shadow RealSense pipeline.

This test does not load ACT and does not publish robot commands.  It starts the
same RealSense camera pair used by deployment, reads the latest policy-depth
frames for a fixed duration, and reports worker frame-count deltas.
"""

from __future__ import annotations

import argparse
import json
import time

from door_act_shadow import (
    DEFAULT_DEPTH_INPAINT_MODE,
    DEFAULT_DEPTH_GAUSSIAN_BLUR_KSIZE,
    DEFAULT_DEPTH_GAUSSIAN_BLUR_SIGMA,
    DEFAULT_FRONT_REALSENSE_SERIAL,
    DEFAULT_WRIST_REALSENSE_SERIAL,
    DEPTH_FPS,
    DepthCrop,
    RealSenseDepthPair,
    validate_gaussian_blur_args,
    validate_depth_inpaint_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration_s", type=float, default=10.0)
    parser.add_argument("--poll_hz", type=float, default=30.0)
    parser.add_argument("--depth_inpaint_mode", default=DEFAULT_DEPTH_INPAINT_MODE)
    parser.add_argument("--wrist_serial", default=DEFAULT_WRIST_REALSENSE_SERIAL)
    parser.add_argument("--front_serial", default=DEFAULT_FRONT_REALSENSE_SERIAL)
    parser.add_argument("--camera_worker_mode", choices=["auto", "thread", "process"], default="process")
    parser.add_argument("--camera_warmup_frames", type=int, default=10)
    parser.add_argument("--rs_filters", dest="rs_filters", action="store_true")
    parser.add_argument("--no_rs_filters", dest="rs_filters", action="store_false")
    parser.set_defaults(rs_filters=None)
    parser.add_argument("--rs_spatial_magnitude", type=int, default=2)
    parser.add_argument("--depth_lower_m", type=float, default=0.2)
    parser.add_argument("--depth_far_m", type=float, default=1.5)
    parser.add_argument("--crop_left", type=int, default=0)
    parser.add_argument("--crop_right", type=int, default=0)
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    parser.add_argument("--depth_gaussian_blur_ksize", type=int, default=DEFAULT_DEPTH_GAUSSIAN_BLUR_KSIZE)
    parser.add_argument("--depth_gaussian_blur_sigma", type=float, default=DEFAULT_DEPTH_GAUSSIAN_BLUR_SIGMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.depth_inpaint_mode = validate_depth_inpaint_mode(args.depth_inpaint_mode)
    if args.rs_filters is None:
        args.rs_filters = args.depth_inpaint_mode != "opencv_k_no_rs"
    elif args.depth_inpaint_mode == "opencv_k_no_rs" and bool(args.rs_filters):
        print(
            "warning: --depth_inpaint_mode opencv_k_no_rs always disables RealSense filters; ignoring --rs_filters.",
            flush=True,
        )
        args.rs_filters = False
    args.depth_gaussian_blur_ksize, args.depth_gaussian_blur_sigma = validate_gaussian_blur_args(
        args.depth_gaussian_blur_ksize,
        args.depth_gaussian_blur_sigma,
    )
    camera = RealSenseDepthPair(
        wrist_serial=args.wrist_serial,
        front_serial=args.front_serial,
        use_filters=bool(args.rs_filters),
        warmup_frames=args.camera_warmup_frames,
        align_depth_to_color=True,
        crop=DepthCrop(args.crop_left, args.crop_right, args.crop_top, args.crop_bottom),
        depth_lower_m=args.depth_lower_m,
        depth_far_m=args.depth_far_m,
        worker_mode=args.camera_worker_mode,
        spatial_magnitude=args.rs_spatial_magnitude,
        depth_inpaint_mode=args.depth_inpaint_mode,
        depth_gaussian_blur_ksize=args.depth_gaussian_blur_ksize,
        depth_gaussian_blur_sigma=args.depth_gaussian_blur_sigma,
    )
    poll_dt = 1.0 / max(float(args.poll_hz), 1.0)
    samples = []
    try:
        camera.start()
        # Prime once after startup.
        _, _, meta0 = camera.read()
        start = time.time()
        first_counts = {
            "wrist": int(meta0["wrist"]["frame_count"]),
            "front": int(meta0["front"]["frame_count"]),
        }
        last_meta = meta0
        read_count = 0
        deadline = start + max(float(args.duration_s), 0.1)
        next_t = time.monotonic()
        while time.time() < deadline:
            _, _, meta = camera.read()
            last_meta = meta
            samples.append(
                {
                    "t": time.time(),
                    "wrist_count": int(meta["wrist"]["frame_count"]),
                    "front_count": int(meta["front"]["frame_count"]),
                    "dt_ms": float(meta.get("dt_ms", 0.0)),
                    "wrist_age_ms": float(meta["wrist"].get("age_ms") or 0.0),
                    "front_age_ms": float(meta["front"].get("age_ms") or 0.0),
                    "wrist_opencv_k": meta["wrist"].get("opencv_k"),
                    "front_opencv_k": meta["front"].get("opencv_k"),
                }
            )
            read_count += 1
            next_t += poll_dt
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
        elapsed = max(time.time() - start, 1.0e-6)
        final_counts = {
            "wrist": int(last_meta["wrist"]["frame_count"]),
            "front": int(last_meta["front"]["frame_count"]),
        }
        summary = {
            "duration_s": elapsed,
            "target_camera_fps": DEPTH_FPS,
            "poll_hz": float(args.poll_hz),
            "read_count": int(read_count),
            "read_hz": float(read_count) / elapsed,
            "depth_inpaint_mode": args.depth_inpaint_mode,
            "depth_gaussian_blur": {
                "enabled": args.depth_gaussian_blur_ksize > 0,
                "ksize": int(args.depth_gaussian_blur_ksize),
                "sigma": float(args.depth_gaussian_blur_sigma),
            },
            "worker_mode": last_meta.get("worker_mode"),
            "first_counts": first_counts,
            "final_counts": final_counts,
            "camera_fps": {
                "wrist": (final_counts["wrist"] - first_counts["wrist"]) / elapsed,
                "front": (final_counts["front"] - first_counts["front"]) / elapsed,
            },
            "last_meta": {
                "wrist": last_meta["wrist"],
                "front": last_meta["front"],
                "dt_ms": last_meta.get("dt_ms"),
                "device_timestamp_dt_ms": last_meta.get("device_timestamp_dt_ms"),
            },
        }
        print("camera_hz_summary " + json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
