#!/usr/bin/env python3
"""A2W scripted grasp using SAM3 mask followed by FoundationPose.

The original simulator script and the GT-mask FoundationPose wrapper remain
unchanged. SAM3 receives only front RGB. Simulator segmentation is used after
mask prediction solely to report IoU. Simulator pose supplies pose-error
metrics and the same 15 mm crash guard used by the GT-mask evaluation; rejected
estimates are counted as failures and are never corrected or rescued with GT.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import isaacgym_float_ik_a2w_basearn_push_door_parallel_foundationpose_grasp as grasp


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAM3_ROOT = Path("/home/sivan/whole_body/sam3")
DEFAULT_SAM3_PYTHON = Path("/home/sivan/miniconda3/envs/sam3/bin/python")
DEFAULT_SAM3_CHECKPOINT = DEFAULT_SAM3_ROOT / "sam3.pt"


def parse_sam3_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sam3_root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--sam3_python", type=Path, default=DEFAULT_SAM3_PYTHON)
    parser.add_argument("--sam3_checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--sam3_output_dir", type=Path, default=None)
    parser.add_argument("--sam3_prompt", default="door handle")
    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.005)
    parser.add_argument(
        "--sam3_selection",
        choices=(
            "top_score",
            "wc4_front_roi",
            "wc4_model_free_ref_roi",
            "wc4_initial_handle_roi",
        ),
        default="wc4_front_roi",
    )
    parser.add_argument("--sam3_min_mask_pixels", type=int, default=20)
    parser.add_argument("--sam3_max_mask_pixels", type=int, default=10000)
    parser.add_argument(
        "--sam3_min_selected_score",
        type=float,
        default=0.0,
        help=(
            "Reject a selected SAM3 candidate below this visual confidence and retry on a "
            "later frame; 0 preserves the previous behavior."
        ),
    )
    parser.add_argument("--sam3_startup_timeout", type=float, default=600.0)
    parser.add_argument("--sam3_inference_timeout", type=float, default=180.0)
    return parser.parse_known_args(argv)


class Sam3Client:
    def __init__(self, cfg: argparse.Namespace, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir.expanduser().resolve()
        self.input_dir = self.output_dir / "stream_inputs"
        self.result_dir = self.output_dir / "stream_results"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        ready_path = self.output_dir / "worker_ready.json"
        ready_path.unlink(missing_ok=True)
        self.log_handle = (self.output_dir / "sam3_worker.log").open("w", encoding="utf-8")
        command = [
            str(cfg.sam3_python.expanduser().resolve()),
            "-u",
            str(SCRIPT_DIR / "sam3_stream_worker.py"),
            "--sam3_root",
            str(cfg.sam3_root.expanduser().resolve()),
            "--checkpoint",
            str(cfg.sam3_checkpoint.expanduser().resolve()),
            "--output_dir",
            str(self.output_dir),
            "--prompt",
            str(cfg.sam3_prompt),
            "--confidence_threshold",
            str(cfg.sam3_confidence_threshold),
            "--selection",
            str(cfg.sam3_selection),
            "--min_mask_pixels",
            str(cfg.sam3_min_mask_pixels),
            "--max_mask_pixels",
            str(cfg.sam3_max_mask_pixels),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.frame_id = 0
        print(f"SAM3 worker started: pid={self.process.pid} log={self.log_handle.name}", flush=True)
        deadline = time.monotonic() + float(cfg.sam3_startup_timeout)
        while not ready_path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"SAM3 worker exited during startup with code {self.process.returncode}; "
                    f"see {self.log_handle.name}"
                )
            if time.monotonic() >= deadline:
                self.process.kill()
                raise TimeoutError(f"SAM3 startup exceeded {cfg.sam3_startup_timeout:g}s")
            time.sleep(0.2)
        print("SAM3 model loaded; RGB segmentation is ready.", flush=True)

    def segment(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        if self.process.poll() is not None:
            raise RuntimeError(f"SAM3 worker exited; see {self.log_handle.name}")
        frame_id = self.frame_id
        self.frame_id += 1
        input_path = self.input_dir / f"frame_{frame_id:06d}.npz"
        result_path = self.result_dir / f"frame_{frame_id:06d}.json"
        result_path.unlink(missing_ok=True)
        # RGB is the only observation sent to SAM3.
        np.savez(input_path, rgb=np.asarray(rgb, dtype=np.uint8))
        request = {"frame_id": frame_id, "input_path": str(input_path), "result_path": str(result_path)}
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + float(self.cfg.sam3_inference_timeout)
        while not result_path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f"SAM3 worker exited; see {self.log_handle.name}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"SAM3 frame {frame_id} exceeded {self.cfg.sam3_inference_timeout:g}s")
            time.sleep(0.05)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "error":
            raise RuntimeError(f"SAM3 frame {frame_id} failed: {payload.get('error')}")
        mask = np.load(payload["mask_path"]).astype(bool)
        return mask, payload

    def close(self) -> None:
        try:
            if self.process.poll() is None and self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=10)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=3)
        finally:
            self.log_handle.close()


def main() -> None:
    sam_cfg, remaining = parse_sam3_args(sys.argv[1:])
    for required in (sam_cfg.sam3_root, sam_cfg.sam3_python, sam_cfg.sam3_checkpoint):
        if not required.expanduser().exists():
            raise FileNotFoundError(required)
    sys.argv = [sys.argv[0], *remaining]

    original_camera_inputs = grasp.camera_inputs
    original_register = grasp.register_rendered_env
    original_foundationpose_close = grasp.fp.FoundationPoseClient.close
    holder: dict[str, Sam3Client] = {}

    def sam3_camera_inputs(gym, sim, st):
        rgb, depth, gt_mask, K, gt_pose = original_camera_inputs(gym, sim, st)
        if st.traj.get("sam3_foundationpose_initial_mask_done", False):
            # FoundationPose tracking after registration consumes RGB-D only.
            # A zero mask makes that data boundary explicit and avoids running
            # SAM3 repeatedly for the same environment.
            return rgb, depth, np.zeros_like(gt_mask, dtype=bool), K, gt_pose
        client = holder.get("client")
        if client is None:
            if sam_cfg.sam3_output_dir is not None:
                output_dir = sam_cfg.sam3_output_dir
            else:
                # Keep SAM3 artifacts next to this run's FoundationPose output.
                fp_cfg, _ = grasp.fp.parse_wrapper_args(sys.argv[1:])
                output_dir = fp_cfg.foundationpose_output_dir / "sam3"
            client = Sam3Client(sam_cfg, output_dir)
            holder["client"] = client
        mask, payload = client.segment(rgb)
        payload = dict(payload)
        if float(payload.get("selected_score", 0.0)) < float(sam_cfg.sam3_min_selected_score):
            mask = np.zeros_like(mask, dtype=bool)
            payload["visual_confidence_rejected"] = True
            payload["visual_confidence_threshold"] = float(sam_cfg.sam3_min_selected_score)
        else:
            payload["visual_confidence_rejected"] = False
        intersection = int(np.logical_and(mask, gt_mask).sum())
        union = int(np.logical_or(mask, gt_mask).sum())
        payload.update(
            {
                "gt_mask_pixels_eval_only": int(gt_mask.sum()),
                "mask_iou_eval_only": float(intersection / max(1, union)),
            }
        )
        st.traj["sam3_mask_report"] = payload
        if int(mask.sum()) >= int(sam_cfg.sam3_min_mask_pixels):
            st.traj["sam3_foundationpose_initial_mask_done"] = True
        print(
            f"[SAM3 env={st.index:02d}] score={payload['selected_score']:.4f} "
            f"mask={payload['mask_pixels']} IoU(eval-only)={payload['mask_iou_eval_only']:.3f} "
            f"selection={payload['selection_reason']} "
            f"confidence_rejected={payload['visual_confidence_rejected']}",
            flush=True,
        )
        return rgb, depth, mask, K, gt_pose

    def sam3_register_rendered_env(gym, sim, st, client):
        report = original_register(gym, sim, st, client)
        report["sam3"] = st.traj.get("sam3_mask_report", {})
        return report

    grasp.camera_inputs = sam3_camera_inputs
    grasp.register_rendered_env = sam3_register_rendered_env

    def close_foundationpose_and_sam3(client):
        original_foundationpose_close(client)
        sam3_client = holder.pop("client", None)
        if sam3_client is not None:
            sam3_client.close()
            print("SAM3 worker closed before grasp control.", flush=True)

    # The FoundationPose grasp wrapper closes registration as soon as all
    # environments have a pose.  Release SAM3 at exactly the same boundary so
    # the two perception models do not hold ~7.5 GB during PhysX contact.
    grasp.fp.FoundationPoseClient.close = close_foundationpose_and_sam3
    grasp.MASK_USAGE_DESCRIPTION = (
        "SAM3 text-prompt mask from front RGB; fixed WC4 front-view ROI selects the instance; "
        "simulation GT mask is evaluation-only and never enters SAM3 or FoundationPose"
    )
    try:
        grasp.main()
    finally:
        client = holder.get("client")
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
