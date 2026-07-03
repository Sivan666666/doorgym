#!/usr/bin/env python3
"""Capture a few D435 RGB + aligned depth frames at 1280x720.

The script configures both streams at 1280x720 @ 30Hz and aligns depth to the
chosen target stream.  With ``--align_to color`` (default), the saved RGB image
and aligned depth image are both 1280x720 in the RGB camera frame.

Example:

    python3 high-level/real_deploy/capture_d435_1280_aligned_frames.py \
      --frames 3 \
      --align_to color

Outputs:
  - frame_000_rgb.png
  - frame_000_raw_aligned_depth_mm.png    # uint16, millimetres, before filters
  - frame_000_aligned_depth_mm.png        # uint16, millimetres, after optional RealSense filters
  - frame_000_depth_gray_0p2_1p5.png      # uint8 visualization
  - frame_000_depth_color_0p2_1p5.png     # color visualization
  - frame_000_contact.png                 # RGB + depth preview
  - metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - only used on minimal systems
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


def _jsonable_intrinsics(intr: Any) -> dict[str, Any]:
    fx = float(intr.fx)
    fy = float(intr.fy)
    width = int(intr.width)
    height = int(intr.height)
    fov_x = 2.0 * math.atan(width / (2.0 * fx)) * 180.0 / math.pi if fx > 0 else 0.0
    fov_y = 2.0 * math.atan(height / (2.0 * fy)) * 180.0 / math.pi if fy > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(x) for x in intr.coeffs],
        "fov_x_deg": fov_x,
        "fov_y_deg": fov_y,
    }


def _device_info(device: Any, rs: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, label in (
        (rs.camera_info.name, "name"),
        (rs.camera_info.serial_number, "serial"),
        (rs.camera_info.firmware_version, "firmware"),
        (rs.camera_info.usb_type_descriptor, "usb"),
        (rs.camera_info.physical_port, "physical_port"),
        (rs.camera_info.product_line, "product_line"),
    ):
        try:
            if device.supports(key):
                out[label] = device.get_info(key)
        except Exception:
            pass
    return out


def _list_devices(rs: Any) -> list[dict[str, str]]:
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        devices.append(_device_info(dev, rs))
    return devices


def _str_to_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _add_legacy_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help: str = "") -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help)
    parser.add_argument(f"--no_{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def _make_realsense_filters(rs: Any, args: argparse.Namespace) -> list[Any]:
    if not args.rs_filters:
        return []
    filters = []
    if args.rs_spatial:
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, int(args.rs_spatial_magnitude))
        spatial.set_option(rs.option.filter_smooth_alpha, float(args.rs_spatial_alpha))
        spatial.set_option(rs.option.filter_smooth_delta, float(args.rs_spatial_delta))
        spatial.set_option(rs.option.holes_fill, int(args.rs_spatial_holes_fill))
        filters.append(spatial)
    if args.rs_temporal:
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, float(args.rs_temporal_alpha))
        temporal.set_option(rs.option.filter_smooth_delta, float(args.rs_temporal_delta))
        filters.append(temporal)
    if args.rs_hole_filling:
        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, int(args.rs_hole_filling_mode))
        filters.append(hole_filling)
    return filters


def _depth_frame_to_mm(depth_frame: Any, depth_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_m = depth_raw.astype(np.float32) * depth_scale
    depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return depth_raw, depth_m, depth_mm


def _depth_to_u8(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    clipped = np.clip(depth_m, near_m, far_m)
    # Same visual convention as most debug panels here: invalid is black,
    # nearer is darker, farther is brighter.
    u8 = ((clipped - near_m) / max(far_m - near_m, 1e-6) * 255.0).astype(np.uint8)
    u8[~valid] = 0
    return u8


def _depth_to_color_rgb(depth_gray: np.ndarray) -> np.ndarray:
    try:
        import matplotlib

        rgba = matplotlib.colormaps["turbo"](depth_gray.astype(np.float32) / 255.0)
        color_rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        color_rgb[depth_gray == 0] = (0, 0, 0)
        return color_rgb
    except Exception:
        # Last-resort grayscale RGB.
        return np.repeat(depth_gray[..., None], 3, axis=2)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        raise RuntimeError("PIL is not available for PNG writing.")
    Image.fromarray(array).save(path)


def _label_panel_rgb(img_rgb: np.ndarray, text: str) -> np.ndarray:
    if Image is not None:
        image = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default() if ImageFont is not None else None
        draw.rectangle((0, 0, img_rgb.shape[1], 34), fill=(0, 0, 0))
        draw.text((12, 10), text, fill=(255, 255, 255), font=font)
        return np.asarray(image)
    return img_rgb


def _make_contact(rgb: np.ndarray, depth_gray: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
    gray_rgb = np.repeat(depth_gray[..., None], 3, axis=2)
    panels = [
        ("RGB 1280x720", rgb),
        ("aligned depth gray 0.2-1.5m", gray_rgb),
        ("aligned depth color 0.2-1.5m", depth_color),
    ]
    labelled = [_label_panel_rgb(img, text) for text, img in panels]
    return np.concatenate(labelled, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="", help="Optional RealSense serial. Defaults to the first connected camera.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5, help="Number of frames to skip between saved frames.")
    parser.add_argument("--align_to", choices=("color", "depth"), default="color")
    parser.add_argument("--near_m", type=float, default=0.2)
    parser.add_argument("--far_m", type=float, default=1.5)
    parser.add_argument("--rs_filters", action="store_true", help="Enable RealSense spatial/temporal/hole-filling filters.")
    _add_legacy_bool_arg(parser, "rs_spatial", default=True)
    _add_legacy_bool_arg(parser, "rs_temporal", default=True)
    _add_legacy_bool_arg(parser, "rs_hole_filling", default=True)
    parser.add_argument("--rs_spatial_magnitude", type=int, default=2)
    parser.add_argument("--rs_spatial_alpha", type=float, default=0.5)
    parser.add_argument("--rs_spatial_delta", type=float, default=20.0)
    parser.add_argument(
        "--rs_spatial_holes_fill",
        type=int,
        default=0,
        help="spatial_filter holes_fill option. 0 disables spatial hole propagation.",
    )
    parser.add_argument("--rs_temporal_alpha", type=float, default=0.4)
    parser.add_argument("--rs_temporal_delta", type=float, default=20.0)
    parser.add_argument(
        "--rs_hole_filling_mode",
        type=int,
        default=1,
        help="hole_filling_filter mode: RealSense option value, commonly 0/1/2.",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory. Default: high-level/real_deploy/local_runs/d435_1280_aligned_<timestamp>",
    )
    parser.add_argument("--list_devices", action="store_true", help="Print connected RealSense devices and exit.")
    args = parser.parse_args()

    import pyrealsense2 as rs

    devices = _list_devices(rs)
    if args.list_devices:
        print(json.dumps(devices, indent=2, ensure_ascii=False))
        return
    if not devices:
        raise RuntimeError("No RealSense device found.")

    out_dir = Path(args.out_dir) if args.out_dir else Path("high-level/real_deploy/local_runs") / (
        "d435_1280_aligned_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(str(args.serial))
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)

    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color if args.align_to == "color" else rs.stream.depth)
    rs_filters = _make_realsense_filters(rs, args)

    meta: dict[str, Any] = {
        "requested": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "frames": args.frames,
            "align_to": args.align_to,
            "near_m": args.near_m,
            "far_m": args.far_m,
            "rs_filters": bool(args.rs_filters),
            "rs_spatial": bool(args.rs_spatial),
            "rs_temporal": bool(args.rs_temporal),
            "rs_hole_filling": bool(args.rs_hole_filling),
            "rs_spatial_magnitude": args.rs_spatial_magnitude,
            "rs_spatial_alpha": args.rs_spatial_alpha,
            "rs_spatial_delta": args.rs_spatial_delta,
            "rs_spatial_holes_fill": args.rs_spatial_holes_fill,
            "rs_temporal_alpha": args.rs_temporal_alpha,
            "rs_temporal_delta": args.rs_temporal_delta,
            "rs_hole_filling_mode": args.rs_hole_filling_mode,
        },
        "connected_devices": devices,
        "output_dir": str(out_dir),
        "saved_frames": [],
    }

    try:
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        meta["device"] = _device_info(device, rs)
        meta["depth_scale_m_per_unit"] = depth_scale
        meta["stream_intrinsics_before_align"] = {
            "depth": _jsonable_intrinsics(
                profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics
            ),
            "color": _jsonable_intrinsics(
                profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
            ),
        }

        for _ in range(max(0, args.warmup_frames)):
            frames = pipeline.wait_for_frames(5000)
            aligned_warmup = align.process(frames)
            if args.rs_filters:
                warm_depth = aligned_warmup.get_depth_frame()
                for rs_filter in rs_filters:
                    warm_depth = rs_filter.process(warm_depth)

        saved = 0
        frame_idx = 0
        t0 = time.perf_counter()
        while saved < args.frames:
            frames = pipeline.wait_for_frames(5000)
            frame_idx += 1
            if args.stride > 1 and frame_idx % args.stride != 0:
                continue

            raw_depth_frame_before_align = frames.get_depth_frame()
            raw_color_frame_before_align = frames.get_color_frame()
            aligned = align.process(frames)
            raw_aligned_depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not raw_aligned_depth_frame or not color_frame:
                continue
            depth_frame = raw_aligned_depth_frame
            if args.rs_filters:
                for rs_filter in rs_filters:
                    depth_frame = rs_filter.process(depth_frame)

            color_rgb = np.asanyarray(color_frame.get_data())
            raw_aligned_depth_raw, raw_aligned_depth_m, raw_aligned_depth_mm = _depth_frame_to_mm(
                raw_aligned_depth_frame,
                depth_scale,
            )
            depth_raw, depth_m, depth_mm = _depth_frame_to_mm(depth_frame, depth_scale)
            depth_gray = _depth_to_u8(depth_m, args.near_m, args.far_m)
            depth_color = _depth_to_color_rgb(depth_gray)
            contact = _make_contact(color_rgb, depth_gray, depth_color)

            stem = f"frame_{saved:03d}"
            _save_png(out_dir / f"{stem}_rgb.png", color_rgb)
            _save_png(out_dir / f"{stem}_raw_aligned_depth_mm.png", raw_aligned_depth_mm)
            _save_png(out_dir / f"{stem}_aligned_depth_mm.png", depth_mm)
            _save_png(out_dir / f"{stem}_depth_gray_0p2_1p5.png", depth_gray)
            _save_png(out_dir / f"{stem}_depth_color_0p2_1p5.png", depth_color)
            _save_png(out_dir / f"{stem}_contact.png", contact)

            raw_depth_intr_before_align = (
                raw_depth_frame_before_align.profile.as_video_stream_profile().intrinsics
                if raw_depth_frame_before_align
                else None
            )
            raw_color_intr_before_align = (
                raw_color_frame_before_align.profile.as_video_stream_profile().intrinsics
                if raw_color_frame_before_align
                else None
            )
            depth_intr = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intr = color_frame.profile.as_video_stream_profile().intrinsics
            valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
            raw_valid = raw_aligned_depth_m[np.isfinite(raw_aligned_depth_m) & (raw_aligned_depth_m > 0)]
            frame_meta = {
                "index": saved,
                "frame_number": int(depth_frame.get_frame_number()),
                "timestamp_ms": float(depth_frame.get_timestamp()),
                "rgb_shape": list(color_rgb.shape),
                "depth_shape": list(depth_raw.shape),
                "depth_units_dtype": str(depth_raw.dtype),
                "rs_filters_enabled": bool(args.rs_filters),
                "raw_aligned_depth_m_min": float(raw_valid.min()) if raw_valid.size else None,
                "raw_aligned_depth_m_mean": float(raw_valid.mean()) if raw_valid.size else None,
                "raw_aligned_depth_m_max": float(raw_valid.max()) if raw_valid.size else None,
                "depth_m_min": float(valid.min()) if valid.size else None,
                "depth_m_mean": float(valid.mean()) if valid.size else None,
                "depth_m_max": float(valid.max()) if valid.size else None,
                "depth_intrinsics_before_align": (
                    _jsonable_intrinsics(raw_depth_intr_before_align)
                    if raw_depth_intr_before_align is not None
                    else None
                ),
                "color_intrinsics_before_align": (
                    _jsonable_intrinsics(raw_color_intr_before_align)
                    if raw_color_intr_before_align is not None
                    else None
                ),
                "depth_intrinsics_after_align": _jsonable_intrinsics(depth_intr),
                "color_intrinsics_after_align": _jsonable_intrinsics(color_intr),
                "files": {
                    "rgb": f"{stem}_rgb.png",
                    "raw_aligned_depth_mm": f"{stem}_raw_aligned_depth_mm.png",
                    "aligned_depth_mm": f"{stem}_aligned_depth_mm.png",
                    "depth_gray": f"{stem}_depth_gray_0p2_1p5.png",
                    "depth_color": f"{stem}_depth_color_0p2_1p5.png",
                    "contact": f"{stem}_contact.png",
                },
            }
            meta["saved_frames"].append(frame_meta)
            print(
                f"saved {stem}: rgb={color_rgb.shape} depth={depth_raw.shape} "
                f"mean_depth_m={frame_meta['depth_m_mean']}"
            )
            saved += 1

        meta["elapsed_capture_s"] = time.perf_counter() - t0
    finally:
        pipeline.stop()

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"done out_dir={out_dir}")


if __name__ == "__main__":
    main()
