#!/usr/bin/env python3
"""Benchmark ZED RGBD capture throughput for different depth modes.

This intentionally does not save any frames.  Each measured frame does:

  grab()
  retrieve_image(LEFT)
  retrieve_measure(DEPTH)
  get_data() for both RGB and depth

So the reported rate is closer to the Python-side RGBD acquisition loop used by
deployment code than a raw camera grab rate.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import numpy as np
import pyzed.sl as sl


RESOLUTION_MAP = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1200": sl.RESOLUTION.HD1200,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "SVGA": sl.RESOLUTION.SVGA,
    "VGA": sl.RESOLUTION.VGA,
}


def parse_res_spec(spec: str) -> tuple[str, Any, int]:
    """Parse RES@FPS, e.g. HD2K@15."""
    if "@" in spec:
        name, fps_s = spec.split("@", 1)
        fps = int(fps_s)
    else:
        name = spec
        fps = 0
    name = name.upper()
    if name not in RESOLUTION_MAP:
        raise ValueError(f"Unknown resolution {name}; options={sorted(RESOLUTION_MAP)}")
    return name, RESOLUTION_MAP[name], fps


def benchmark_one(
    *,
    res_name: str,
    resolution: Any,
    fps: int,
    mode_name: str,
    warmup_frames: int,
    measure_frames: int,
    depth_min_m: float,
    depth_max_m: float,
) -> dict[str, Any]:
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = resolution
    if fps > 0:
        init.camera_fps = fps
    init.depth_mode = getattr(sl.DEPTH_MODE, mode_name)
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = depth_min_m
    init.depth_maximum_distance = depth_max_m

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        return {"ok": False, "error": str(status)}

    runtime = sl.RuntimeParameters()
    rgb = sl.Mat()
    depth = sl.Mat()

    warmup_done = 0
    warmup_timeout_s = max(5.0, warmup_frames / max(1, fps or 30) * 4.0)
    warmup_start = time.perf_counter()
    while warmup_done < warmup_frames and time.perf_counter() - warmup_start < warmup_timeout_s:
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        cam.retrieve_image(rgb, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        _ = rgb.get_data().shape
        _ = depth.get_data().shape
        warmup_done += 1

    ok = 0
    dt_list: list[float] = []
    valid_samples: list[float] = []
    measure_timeout_s = max(8.0, measure_frames / max(1, fps or 30) * 4.0)
    start = time.perf_counter()
    prev = start
    last_rgb_shape = None
    last_depth_shape = None

    while ok < measure_frames and time.perf_counter() - start < measure_timeout_s:
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        cam.retrieve_image(rgb, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        rgb_np = rgb.get_data()
        depth_np = depth.get_data()
        last_rgb_shape = tuple(rgb_np.shape)
        last_depth_shape = tuple(depth_np.shape)

        # Touch the arrays, but do not do expensive full-frame stats every frame.
        if ok % max(1, measure_frames // 5) == 0:
            d = np.asarray(depth_np)
            valid_samples.append(float(np.isfinite(d).mean()))

        now = time.perf_counter()
        dt_list.append(now - prev)
        prev = now
        ok += 1

    elapsed = time.perf_counter() - start
    cam.close()

    if ok <= 1 or elapsed <= 0:
        return {"ok": False, "error": "no measured frames", "frames": ok, "elapsed_s": elapsed}

    hz = ok / elapsed
    dt = dt_list[1:] if len(dt_list) > 1 else dt_list
    dt_sorted = sorted(dt)
    p95_idx = int(0.95 * (len(dt_sorted) - 1)) if len(dt_sorted) > 1 else 0
    return {
        "ok": True,
        "mode": mode_name,
        "resolution": res_name,
        "target_fps": fps,
        "warmup_frames": warmup_done,
        "frames": ok,
        "elapsed_s": elapsed,
        "hz": hz,
        "dt_ms_p50": statistics.median(dt) * 1000.0 if dt else 0.0,
        "dt_ms_p95": dt_sorted[p95_idx] * 1000.0 if dt_sorted else 0.0,
        "valid_ratio_mean": statistics.mean(valid_samples) if valid_samples else float("nan"),
        "rgb_shape": last_rgb_shape,
        "depth_shape": last_depth_shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL_LIGHT", "NEURAL", "NEURAL_PLUS"],
    )
    parser.add_argument("--res", nargs="+", default=["HD2K@15", "HD1080@30"])
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--measure_frames", type=int, default=90)
    parser.add_argument("--depth_min_m", type=float, default=0.2)
    parser.add_argument("--depth_max_m", type=float, default=20.0)
    args = parser.parse_args()

    print(f"zed_sdk {sl.Camera().get_sdk_version()}")
    print("benchmark: grab + retrieve LEFT RGB + retrieve DEPTH + get_data, no saving")
    print("note: if target FPS is lower than compute capacity, Hz will be camera-limited.")

    for res_spec in args.res:
        res_name, resolution, fps = parse_res_spec(res_spec)
        print(f"\n=== {res_name}@{fps or 'default'} ===", flush=True)
        for mode in args.modes:
            try:
                result = benchmark_one(
                    res_name=res_name,
                    resolution=resolution,
                    fps=fps,
                    mode_name=mode,
                    warmup_frames=args.warmup_frames,
                    measure_frames=args.measure_frames,
                    depth_min_m=args.depth_min_m,
                    depth_max_m=args.depth_max_m,
                )
            except Exception as exc:  # noqa: BLE001 - benchmark should keep going.
                result = {"ok": False, "error": repr(exc)}

            if result.get("ok"):
                print(
                    "{mode:13s} hz={hz:6.2f} dt_p50={p50:6.1f}ms "
                    "dt_p95={p95:6.1f}ms valid={valid:.3f} rgb={rgb} depth={depth}".format(
                        mode=mode,
                        hz=result["hz"],
                        p50=result["dt_ms_p50"],
                        p95=result["dt_ms_p95"],
                        valid=result["valid_ratio_mean"],
                        rgb=result["rgb_shape"],
                        depth=result["depth_shape"],
                    ),
                    flush=True,
                )
            else:
                print(f"{mode:13s} FAILED {result}", flush=True)


if __name__ == "__main__":
    main()
