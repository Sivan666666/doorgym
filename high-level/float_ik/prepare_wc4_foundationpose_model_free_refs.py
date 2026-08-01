#!/usr/bin/env python3
"""Prepare SAM3-masked WC4 RGB-D reference views for FoundationPose model-free.

This utility consumes RGB-D frames previously captured by the independent
FoundationPose front-camera wrapper.  It does not use the saved simulator mask
to produce training masks: every reference mask is inferred from RGB by the
standalone SAM3 worker.  The simulator mask is read only to report IoU.

The saved ``cam_in_ob`` matrices are camera-to-object transforms required by
the official BundleSDF/Neural Object Field helper.  Here the object coordinate
frame is the simulator handle frame, and the relative transforms come from the
recorded camera/handle transforms.  No CAD mesh is read by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    SCRIPT_DIR.parents[0]
    / "logs/foundationpose/wc4_front_camera_dense_track_smoke/stream_inputs"
)
DEFAULT_OUTPUT = SCRIPT_DIR.parents[0] / "logs/foundationpose/model_free_wc4_refs_sam3_16"
DEFAULT_SAM3_ROOT = Path("/home/sivan/whole_body/sam3")
DEFAULT_SAM3_PYTHON = Path("/home/sivan/miniconda3/envs/sam3/bin/python")
DEFAULT_SAM3_CHECKPOINT = DEFAULT_SAM3_ROOT / "sam3.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_views", type=int, default=16)
    parser.add_argument(
        "--reference_end_fraction",
        type=float,
        default=0.8,
        help="Use the first fraction of the sequence for references and reserve later views for testing.",
    )
    parser.add_argument("--sam3_root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--sam3_python", type=Path, default=DEFAULT_SAM3_PYTHON)
    parser.add_argument("--sam3_checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--sam3_prompt", default="door handle")
    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.005)
    parser.add_argument("--sam3_min_mask_pixels", type=int, default=20)
    parser.add_argument("--sam3_max_mask_pixels", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Sam3Worker:
    def __init__(self, args: argparse.Namespace, output_dir: Path):
        self.args = args
        self.output_dir = output_dir
        self.input_dir = output_dir / "inputs"
        self.result_dir = output_dir / "results"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        ready_path = output_dir / "worker_ready.json"
        ready_path.unlink(missing_ok=True)
        self.log = (output_dir / "worker.log").open("w", encoding="utf-8")
        command = [
            str(args.sam3_python.resolve()),
            "-u",
            str(SCRIPT_DIR / "sam3_stream_worker.py"),
            "--sam3_root",
            str(args.sam3_root.resolve()),
            "--checkpoint",
            str(args.sam3_checkpoint.resolve()),
            "--output_dir",
            str(output_dir.resolve()),
            "--prompt",
            args.sam3_prompt,
            "--confidence_threshold",
            str(args.sam3_confidence_threshold),
            "--selection",
            "wc4_model_free_ref_roi",
            "--min_mask_pixels",
            str(args.sam3_min_mask_pixels),
            "--max_mask_pixels",
            str(args.sam3_max_mask_pixels),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        deadline = time.monotonic() + 600.0
        while not ready_path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f"SAM3 worker failed; see {self.log.name}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"SAM3 startup timed out; see {self.log.name}")
            time.sleep(0.2)

    def segment(self, frame_id: int, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        input_path = self.input_dir / f"frame_{frame_id:06d}.npz"
        result_path = self.result_dir / f"frame_{frame_id:06d}.json"
        np.savez(input_path, rgb=np.asarray(rgb, dtype=np.uint8))
        result_path.unlink(missing_ok=True)
        request = {
            "frame_id": int(frame_id),
            "input_path": str(input_path.resolve()),
            "result_path": str(result_path.resolve()),
        }
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + float(self.args.timeout)
        while not result_path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f"SAM3 worker exited; see {self.log.name}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"SAM3 frame {frame_id} timed out")
            time.sleep(0.05)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            raise RuntimeError(f"SAM3 did not return a usable mask for frame {frame_id}: {payload}")
        return np.load(payload["mask_path"]).astype(bool), payload

    def close(self) -> None:
        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=20)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self.log.close()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_files = sorted(source_dir.glob("frame_*.npz"))
    if len(source_files) < args.num_views + 1:
        raise ValueError(f"Need at least {args.num_views + 1} source frames, found {len(source_files)}")
    if not (0.1 <= args.reference_end_fraction < 1.0):
        raise ValueError("reference_end_fraction must be in [0.1, 1.0)")
    for path in (args.sam3_root, args.sam3_python, args.sam3_checkpoint):
        if not path.expanduser().exists():
            raise FileNotFoundError(path)

    reference_last = max(args.num_views - 1, int(len(source_files) * args.reference_end_fraction) - 1)
    selected_indices = np.linspace(0, reference_last, args.num_views, dtype=np.int64)
    if len(np.unique(selected_indices)) != args.num_views:
        raise RuntimeError(f"Reference selection produced duplicate indices: {selected_indices.tolist()}")
    heldout_index = len(source_files) - 1

    for name in ("rgb", "depth_enhanced", "depth_npy", "mask", "cam_in_ob"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    sam_output = output_dir / "sam3"
    worker = Sam3Worker(args, sam_output)
    reports: list[dict] = []
    K_reference: np.ndarray | None = None
    try:
        for output_index, source_index in enumerate(selected_indices.tolist()):
            source_path = source_files[source_index]
            with np.load(source_path) as frame:
                rgb = np.asarray(frame["rgb"], dtype=np.uint8)
                depth = np.asarray(frame["depth"], dtype=np.float32)
                gt_mask_eval_only = np.asarray(frame["mask"], dtype=np.uint8).astype(bool)
                K = np.asarray(frame["K"], dtype=np.float64).reshape(3, 3)
                camera_from_object = np.asarray(
                    frame["gt_T_camera_handle"], dtype=np.float64
                ).reshape(4, 4)
            mask, sam_report = worker.segment(output_index, rgb)
            valid_depth = np.isfinite(depth) & (depth >= 0.05) & (depth <= 5.0)
            mask &= valid_depth
            if int(mask.sum()) < args.sam3_min_mask_pixels:
                raise RuntimeError(
                    f"SAM3/depth intersection too small for {source_path.name}: {int(mask.sum())}"
                )
            camera_in_object = np.linalg.inv(camera_from_object)
            stem = f"{output_index:06d}"
            cv2.imwrite(str(output_dir / "rgb" / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            depth_mm = np.clip(np.rint(depth * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
            depth_mm[~valid_depth] = 0
            cv2.imwrite(str(output_dir / "depth_enhanced" / f"{stem}.png"), depth_mm)
            np.save(output_dir / "depth_npy" / f"{stem}.npy", depth.astype(np.float32))
            cv2.imwrite(str(output_dir / "mask" / f"{stem}.png"), mask.astype(np.uint8) * 255)
            np.savetxt(output_dir / "cam_in_ob" / f"{stem}.txt", camera_in_object, fmt="%.10f")

            intersection = int(np.logical_and(mask, gt_mask_eval_only).sum())
            union = int(np.logical_or(mask, gt_mask_eval_only).sum())
            report = {
                "output_index": output_index,
                "source_index": source_index,
                "source_file": str(source_path),
                "sam3_mask_pixels": int(mask.sum()),
                "gt_mask_pixels_eval_only": int(gt_mask_eval_only.sum()),
                "mask_iou_eval_only": float(intersection / max(1, union)),
                "sam3_score": float(sam_report["selected_score"]),
                "sam3_selection_reason": sam_report["selection_reason"],
                "camera_in_object_translation_m": camera_in_object[:3, 3].tolist(),
            }
            reports.append(report)
            print(
                f"[{output_index + 1:02d}/{args.num_views}] source={source_index:02d} "
                f"mask={report['sam3_mask_pixels']} IoU(eval)={report['mask_iou_eval_only']:.3f}",
                flush=True,
            )
            if K_reference is None:
                K_reference = K
            elif not np.allclose(K_reference, K, atol=1e-6):
                raise ValueError("Reference frames have inconsistent camera intrinsics")
    finally:
        worker.close()

    assert K_reference is not None
    np.savetxt(output_dir / "K.txt", K_reference, fmt="%.10f")
    (output_dir / "select_frames.yml").write_text(
        yaml.safe_dump({"selected_source_indices": selected_indices.tolist()}), encoding="utf-8"
    )

    # Export one held-out RGB-D frame.  Its SAM3 mask is generated in a second
    # short worker run so it is not one of the reconstruction references.
    heldout_dir = output_dir / "heldout"
    heldout_dir.mkdir(exist_ok=True)
    with np.load(source_files[heldout_index]) as heldout:
        heldout_rgb = np.asarray(heldout["rgb"], dtype=np.uint8)
        heldout_depth = np.asarray(heldout["depth"], dtype=np.float32)
        heldout_gt_mask = np.asarray(heldout["mask"], dtype=np.uint8).astype(bool)
        heldout_K = np.asarray(heldout["K"], dtype=np.float64)
        heldout_pose = np.asarray(heldout["gt_T_camera_handle"], dtype=np.float64)
    heldout_worker = Sam3Worker(args, heldout_dir / "sam3")
    try:
        heldout_mask, heldout_sam = heldout_worker.segment(0, heldout_rgb)
    finally:
        heldout_worker.close()
    heldout_valid = np.isfinite(heldout_depth) & (heldout_depth >= 0.05) & (heldout_depth <= 5.0)
    heldout_mask &= heldout_valid
    cv2.imwrite(str(heldout_dir / "rgb.png"), cv2.cvtColor(heldout_rgb, cv2.COLOR_RGB2BGR))
    np.save(heldout_dir / "depth.npy", heldout_depth.astype(np.float32))
    cv2.imwrite(str(heldout_dir / "mask.png"), heldout_mask.astype(np.uint8) * 255)
    np.savetxt(heldout_dir / "K.txt", heldout_K, fmt="%.10f")
    np.savetxt(heldout_dir / "gt_T_camera_handle.txt", heldout_pose, fmt="%.10f")
    heldout_intersection = int(np.logical_and(heldout_mask, heldout_gt_mask).sum())
    heldout_union = int(np.logical_or(heldout_mask, heldout_gt_mask).sum())

    summary = {
        "status": "ok",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "num_reference_views": args.num_views,
        "selected_source_indices": selected_indices.tolist(),
        "heldout_source_index": heldout_index,
        "mask_source": "SAM3 text prompt from RGB only",
        "gt_mask_usage": "evaluation-only IoU; never used as a reconstruction mask",
        "relative_pose_source": "recorded simulator camera-to-handle transform",
        "cad_mesh_used": False,
        "reference_mask_iou_mean_eval_only": float(
            np.mean([report["mask_iou_eval_only"] for report in reports])
        ),
        "reference_mask_iou_min_eval_only": float(
            np.min([report["mask_iou_eval_only"] for report in reports])
        ),
        "heldout_mask_pixels": int(heldout_mask.sum()),
        "heldout_mask_iou_eval_only": float(heldout_intersection / max(1, heldout_union)),
        "heldout_sam3_score": float(heldout_sam["selected_score"]),
        "references": reports,
    }
    atomic_json(output_dir / "reference_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
