#!/usr/bin/env python3
"""Capture aligned RGB + depth frames from a ZED 2i.

ZED depth is retrieved in the left rectified camera frame, so ``VIEW.LEFT`` and
``MEASURE.DEPTH`` have the same shape and are already aligned.

Default mode uses the highest ZED 2i image resolution:
  - HD2K, 2208x1242 per left image, 15 FPS

Outputs:
  - frame_000_rgb.png
  - frame_000_depth_mm.png                # uint16 millimetres, invalid=0
  - frame_000_depth_m.npy                 # float32 metres, keeps NaN/Inf
  - frame_000_depth_gray_0p2_1p5.png      # uint8 visualization only
  - frame_000_depth_color_0p2_1p5.png     # color visualization only
  - frame_000_contact.png
  - contact_all_3frames.png
  - metadata.json                         # intrinsics, resolution, SDK info
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
from PIL import Image, ImageDraw, ImageFont


def _enum_from_name(enum_cls: Any, name: str) -> Any:
    key = str(name).strip().upper()
    if not hasattr(enum_cls, key):
        valid = [x for x in dir(enum_cls) if x.isupper() and not x.startswith("_")]
        raise ValueError(f"{enum_cls} has no member {key!r}. Valid: {valid}")
    return getattr(enum_cls, key)


def _getattr_float(obj: Any, name: str, default: float | None = None) -> float | None:
    try:
        return float(getattr(obj, name))
    except Exception:
        return default


def _camera_parameters_to_dict(cam: Any) -> dict[str, Any]:
    out = {
        "fx": _getattr_float(cam, "fx"),
        "fy": _getattr_float(cam, "fy"),
        "cx": _getattr_float(cam, "cx"),
        "cy": _getattr_float(cam, "cy"),
        "h_fov_deg": _getattr_float(cam, "h_fov"),
        "v_fov_deg": _getattr_float(cam, "v_fov"),
        "d_fov_deg": _getattr_float(cam, "d_fov"),
    }
    try:
        out["disto"] = [float(x) for x in cam.disto]
    except Exception:
        out["disto"] = None
    try:
        out["resolution"] = {"width": int(cam.image_size.width), "height": int(cam.image_size.height)}
    except Exception:
        pass
    return out


def _resolution_to_dict(resolution: Any) -> dict[str, int]:
    return {"width": int(resolution.width), "height": int(resolution.height)}


def _depth_to_u8(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    safe = np.where(valid, depth_m, near_m)
    clipped = np.clip(safe, near_m, far_m)
    u8 = ((clipped - near_m) / max(far_m - near_m, 1e-6) * 255.0).astype(np.uint8)
    u8[~valid] = 0
    return u8


def _depth_to_color_rgb(depth_u8: np.ndarray) -> np.ndarray:
    try:
        import matplotlib

        rgba = matplotlib.colormaps["turbo"](depth_u8.astype(np.float32) / 255.0)
        color = (rgba[..., :3] * 255.0).astype(np.uint8)
        color[depth_u8 == 0] = (0, 0, 0)
        return color
    except Exception:
        return np.repeat(depth_u8[..., None], 3, axis=2)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _label_panel_rgb(img: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
    draw.text((12, 10), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def _make_contact(rgb: np.ndarray, depth_gray: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
    gray_rgb = np.repeat(depth_gray[..., None], 3, axis=2)
    panels = [
        ("ZED LEFT RGB", rgb),
        ("ZED aligned depth gray 0.2-1.5m", gray_rgb),
        ("ZED aligned depth color 0.2-1.5m", depth_color),
    ]
    return np.concatenate([_label_panel_rgb(img, label) for label, img in panels], axis=1)


def _make_contact_all(out_dir: Path, count: int) -> None:
    imgs = []
    for i in range(count):
        path = out_dir / f"frame_{i:03d}_contact.png"
        if path.exists():
            imgs.append(Image.open(path).convert("RGB"))
    if not imgs:
        return
    # Keep the overview manageable. Original per-frame contact images remain full-res.
    scale = min(1.0, 1920.0 / max(im.width for im in imgs))
    small = [im.resize((int(im.width * scale), int(im.height * scale))) for im in imgs]
    canvas = Image.new("RGB", (max(im.width for im in small), sum(im.height for im in small)))
    y = 0
    for im in small:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(out_dir / "contact_all_3frames.png")


def _depth_stats(depth_m: np.ndarray) -> dict[str, Any]:
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if not valid.size:
        return {"valid_ratio": 0.0, "min_m": None, "mean_m": None, "max_m": None}
    return {
        "valid_ratio": float(valid.size / depth_m.size),
        "min_m": float(valid.min()),
        "mean_m": float(valid.mean()),
        "max_m": float(valid.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--resolution", default="HD2K", help="ZED resolution enum, e.g. HD2K/HD1080/HD720/VGA.")
    parser.add_argument("--fps", type=int, default=15, help="HD2K max is normally 15 FPS on ZED 2i.")
    parser.add_argument("--depth_mode", default="NEURAL", help="ZED depth mode, e.g. NEURAL/ULTRA/QUALITY/PERFORMANCE.")
    parser.add_argument("--auto_exposure", action="store_true", default=True)
    parser.add_argument("--no_auto_exposure", dest="auto_exposure", action="store_false")
    parser.add_argument("--near_m", type=float, default=0.2)
    parser.add_argument("--far_m", type=float, default=1.5)
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    import pyzed.sl as sl

    out_dir = Path(args.out_dir) if args.out_dir else Path("high-level/real_deploy/local_runs") / (
        "zed2i_aligned_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.camera_resolution = _enum_from_name(sl.RESOLUTION, args.resolution)
    init.camera_fps = int(args.fps)
    init.depth_mode = _enum_from_name(sl.DEPTH_MODE, args.depth_mode)
    init.coordinate_units = sl.UNIT.METER
    init.sdk_verbose = 1

    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")
    if args.auto_exposure:
        try:
            zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
            zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
        except Exception as exc:
            print(f"warning: failed to enable ZED auto exposure/white balance: {exc}")

    image = sl.Mat()
    depth = sl.Mat()
    runtime = sl.RuntimeParameters()

    meta: dict[str, Any] = {
        "requested": vars(args),
        "sdk_version": zed.get_sdk_version(),
        "output_dir": str(out_dir),
        "saved_frames": [],
    }

    try:
        info = zed.get_camera_information()
        cam_cfg = info.camera_configuration
        meta["camera_model"] = str(info.camera_model)
        meta["serial_number"] = int(info.serial_number)
        meta["fps"] = int(cam_cfg.fps)
        meta["resolution"] = _resolution_to_dict(cam_cfg.resolution)
        meta["calibration_parameters"] = {
            "left_cam": _camera_parameters_to_dict(cam_cfg.calibration_parameters.left_cam),
            "right_cam": _camera_parameters_to_dict(cam_cfg.calibration_parameters.right_cam),
        }
        try:
            meta["calibration_parameters"]["stereo_transform_T"] = [
                float(x) for x in cam_cfg.calibration_parameters.T
            ]
        except Exception:
            pass
        try:
            meta["calibration_parameters"]["stereo_transform_R"] = [
                float(x) for row in cam_cfg.calibration_parameters.R for x in row
            ]
        except Exception:
            pass

        for _ in range(max(0, int(args.warmup_frames))):
            zed.grab(runtime)

        saved = 0
        idx = 0
        t0 = time.perf_counter()
        while saved < args.frames:
            err = zed.grab(runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                continue
            idx += 1
            if args.stride > 1 and idx % args.stride != 0:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            bgra = image.get_data()
            depth_m = depth.get_data().astype(np.float32)
            if bgra.ndim != 3 or bgra.shape[2] < 3:
                raise RuntimeError(f"Unexpected ZED image shape: {bgra.shape}")
            # ZED images are returned in BGRA order for OpenCV-style usage.
            rgb = bgra[..., :3][..., ::-1].copy()

            depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
            valid = np.isfinite(depth_m) & (depth_m > 0)
            depth_mm[valid] = np.clip(np.rint(depth_m[valid] * 1000.0), 0, np.iinfo(np.uint16).max).astype(
                np.uint16
            )
            depth_gray = _depth_to_u8(depth_m, args.near_m, args.far_m)
            depth_color = _depth_to_color_rgb(depth_gray)
            contact = _make_contact(rgb, depth_gray, depth_color)

            stem = f"frame_{saved:03d}"
            _save_png(out_dir / f"{stem}_rgb.png", rgb)
            _save_png(out_dir / f"{stem}_depth_mm.png", depth_mm)
            np.save(out_dir / f"{stem}_depth_m.npy", depth_m)
            _save_png(out_dir / f"{stem}_depth_gray_0p2_1p5.png", depth_gray)
            _save_png(out_dir / f"{stem}_depth_color_0p2_1p5.png", depth_color)
            _save_png(out_dir / f"{stem}_contact.png", contact)

            frame_meta = {
                "index": saved,
                "rgb_shape": list(rgb.shape),
                "depth_shape": list(depth_m.shape),
                "depth_dtype": str(depth_m.dtype),
                "depth_stats": _depth_stats(depth_m),
                "files": {
                    "rgb": f"{stem}_rgb.png",
                    "depth_mm": f"{stem}_depth_mm.png",
                    "depth_m": f"{stem}_depth_m.npy",
                    "depth_gray": f"{stem}_depth_gray_0p2_1p5.png",
                    "depth_color": f"{stem}_depth_color_0p2_1p5.png",
                    "contact": f"{stem}_contact.png",
                },
            }
            meta["saved_frames"].append(frame_meta)
            print(
                f"saved {stem}: rgb={rgb.shape} depth={depth_m.shape} "
                f"valid={frame_meta['depth_stats']['valid_ratio']:.3f}"
            )
            saved += 1

        meta["elapsed_capture_s"] = time.perf_counter() - t0
        _make_contact_all(out_dir, len(meta["saved_frames"]))
    finally:
        zed.close()

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"done out_dir={out_dir}")


if __name__ == "__main__":
    main()
