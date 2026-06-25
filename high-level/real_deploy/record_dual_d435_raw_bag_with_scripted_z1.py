#!/usr/bin/env python3
"""Record two D435 raw RealSense bags while replaying a scripted Z1 trajectory.

This is meant for filter-debug datasets, not policy inference.  The important
thing is that the RealSense recording keeps the original depth stream as z16
plus the color stream, so the resulting .bag files can later be replayed through
different librealsense filter chains offline.

Typical use on NX, with z1_ctrl + z1_act_ee_bridge already running:

    python3 high-level/real_deploy/record_dual_d435_raw_bag_with_scripted_z1.py \
      --out_dir /tmp/d435_raw_scripted_$(date +%Y%m%d_%H%M%S) \
      --reference_action_npz /tmp/a2w_full_reference.npz \
      --send_shutdown_back_to_start

The script never publishes robot-base velocity.  When replaying a reference
action NPZ, action[0:2] is overwritten with zeros before sending to the Z1
bridge.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from door_act_shadow import (  # type: ignore
        DEFAULT_FRONT_REALSENSE_SERIAL,
        DEFAULT_WRIST_REALSENSE_SERIAL,
        DEPTH_FPS,
        DEPTH_HEIGHT,
        DEPTH_WIDTH,
    )
except Exception:
    DEFAULT_WRIST_REALSENSE_SERIAL = "261222075130"
    DEFAULT_FRONT_REALSENSE_SERIAL = "261222075566"
    DEPTH_WIDTH = 640
    DEPTH_HEIGHT = 480
    DEPTH_FPS = 30


@dataclass
class CameraRecorder:
    name: str
    serial: str
    bag_path: Path
    width: int
    height: int
    fps: int
    enable_color: bool
    pipeline: Any | None = None
    profile: Any | None = None
    depth_scale: float | None = None
    device_info: dict[str, str] | None = None
    frames: int = 0
    stop_event: threading.Event | None = None
    thread: threading.Thread | None = None
    error: str | None = None

    def start(self) -> None:
        import pyrealsense2 as rs

        self.bag_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = rs.config()
        cfg.enable_device(str(self.serial))
        cfg.enable_record_to_file(str(self.bag_path))
        cfg.enable_stream(rs.stream.depth, int(self.width), int(self.height), rs.format.z16, int(self.fps))
        if self.enable_color:
            cfg.enable_stream(rs.stream.color, int(self.width), int(self.height), rs.format.rgb8, int(self.fps))
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(cfg)
        device = self.profile.get_device()
        self.depth_scale = float(device.first_depth_sensor().get_depth_scale())
        info: dict[str, str] = {}
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
                    info[label] = device.get_info(key)
            except Exception:
                pass
        self.device_info = info
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name=f"record-{self.name}", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        assert self.pipeline is not None
        assert self.stop_event is not None
        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(1000)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.error = f"{type(exc).__name__}: {exc}"
                continue
            if frames and frames.get_depth_frame():
                self.frames += 1

    def stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        if self.pipeline is not None:
            self.pipeline.stop()

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "serial": self.serial,
            "bag_path": str(self.bag_path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "enable_color": self.enable_color,
            "frames": self.frames,
            "depth_scale": self.depth_scale,
            "device_info": self.device_info or {},
            "error": self.error,
        }


class Z1ReferenceReplayer:
    def __init__(
        self,
        reference_npz: Path,
        action_host: str,
        action_port: int,
        hz: float,
        start_step: int,
        max_steps: int | None,
        send_home_joint_targets_from_step: int,
        dry_run: bool = False,
    ) -> None:
        self.reference_npz = Path(reference_npz)
        self.action_host = action_host
        self.action_port = int(action_port)
        self.hz = float(hz)
        self.start_step = max(0, int(start_step))
        self.max_steps = None if max_steps is None or int(max_steps) <= 0 else int(max_steps)
        self.send_home_joint_targets_from_step = int(send_home_joint_targets_from_step)
        self.dry_run = bool(dry_run)
        self.summary: dict[str, Any] = {}

    def run(self) -> None:
        with np.load(self.reference_npz, allow_pickle=True) as data:
            if "action" not in data:
                raise ValueError(f"{self.reference_npz} does not contain key 'action'")
            actions = np.asarray(data["action"], dtype=np.float32).copy()
            sim_q = np.asarray(data["sim_q"], dtype=np.float32) if "sim_q" in data else None
        if actions.ndim != 2 or actions.shape[1] != 10:
            raise ValueError(f"expected action [T,10], got {actions.shape}")
        if sim_q is not None and sim_q.shape != (len(actions), 6):
            raise ValueError(f"expected sim_q [{len(actions)},6], got {sim_q.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("reference action contains NaN/Inf")
        if sim_q is not None and not np.isfinite(sim_q).all():
            raise ValueError("reference sim_q contains NaN/Inf")

        end_step = len(actions) if self.max_steps is None else min(len(actions), self.start_step + self.max_steps)
        actions = actions[self.start_step:end_step].copy()
        sim_q_slice = None if sim_q is None else sim_q[self.start_step:end_step].copy()
        actions[:, 0:2] = 0.0  # never move the base from this recorder

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = (self.action_host, self.action_port)
        period = 1.0 / max(self.hz, 1.0e-6)
        sent = 0
        start_wall = time.time()
        start_mono = time.monotonic()
        next_t = time.monotonic()
        try:
            for local_i, action in enumerate(actions):
                global_step = self.start_step + local_i
                payload: dict[str, Any] = {
                    "action": [float(x) for x in action],
                    "source": "record_dual_d435_raw_bag_with_scripted_z1",
                    "reference_step": int(global_step),
                }
                if (
                    sim_q_slice is not None
                    and self.send_home_joint_targets_from_step >= 0
                    and global_step >= self.send_home_joint_targets_from_step
                ):
                    payload["control_mode"] = "home"
                    payload["joint_target"] = [float(x) for x in sim_q_slice[local_i]]
                if not self.dry_run:
                    sock.sendto(json.dumps(payload).encode("utf-8"), addr)
                sent += 1
                next_t += period
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.monotonic()
        finally:
            sock.close()
        self.summary = {
            "reference_npz": str(self.reference_npz),
            "action_host": self.action_host,
            "action_port": self.action_port,
            "hz": self.hz,
            "start_step": self.start_step,
            "end_step_exclusive": int(end_step),
            "sent": sent,
            "dry_run": self.dry_run,
            "duration_s": time.monotonic() - start_mono,
            "start_wall_time": start_wall,
            "end_wall_time": time.time(),
            "dog_commands_published": False,
            "send_home_joint_targets_from_step": self.send_home_joint_targets_from_step,
        }


def send_bridge_shutdown_back_to_start(host: str, port: int, count: int = 3) -> None:
    payload = {
        "bridge_command": "shutdown_back_to_start",
        "source": "record_dual_d435_raw_bag_with_scripted_z1",
    }
    data = json.dumps(payload).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _ in range(max(1, int(count))):
            sock.sendto(data, (host, int(port)))
            time.sleep(0.05)
    finally:
        sock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record dual D435 z16/color .bag files while triggering a scripted Z1 trajectory."
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--wrist_serial", default=DEFAULT_WRIST_REALSENSE_SERIAL)
    parser.add_argument("--front_serial", default=DEFAULT_FRONT_REALSENSE_SERIAL)
    parser.add_argument("--width", type=int, default=DEPTH_WIDTH)
    parser.add_argument("--height", type=int, default=DEPTH_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEPTH_FPS)
    parser.add_argument("--no_color", action="store_true", help="Record only depth z16. Offline align(color) requires color.")
    parser.add_argument("--pre_record_s", type=float, default=2.0)
    parser.add_argument("--post_record_s", type=float, default=2.0)
    parser.add_argument("--duration_s", type=float, default=0.0, help="If no trajectory is given, record for this many seconds.")

    parser.add_argument("--reference_action_npz", type=Path)
    parser.add_argument("--trajectory_hz", type=float, default=25.0)
    parser.add_argument("--trajectory_start_step", type=int, default=0)
    parser.add_argument("--trajectory_max_steps", type=int, default=0)
    parser.add_argument(
        "--send_home_joint_targets_from_step",
        type=int,
        default=364,
        help="When reference contains sim_q, send joint_target/home from this global step. Use -1 to disable.",
    )
    parser.add_argument("--action_host", default="127.0.0.1")
    parser.add_argument("--action_port", type=int, default=15011)
    parser.add_argument("--dry_run_trajectory", action="store_true")

    parser.add_argument(
        "--trajectory_command",
        type=str,
        default="",
        help="Optional shell command to run while recording, instead of/in addition to --reference_action_npz.",
    )
    parser.add_argument("--send_shutdown_back_to_start", action="store_true")
    parser.add_argument("--metadata_name", default="record_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wrist_bag = args.out_dir / "wrist_raw.bag"
    front_bag = args.out_dir / "front_raw.bag"
    manifest_path = args.out_dir / args.metadata_name

    recorders = [
        CameraRecorder("wrist", args.wrist_serial, wrist_bag, args.width, args.height, args.fps, not args.no_color),
        CameraRecorder("front", args.front_serial, front_bag, args.width, args.height, args.fps, not args.no_color),
    ]
    print(
        "record_starting "
        + json.dumps(
            {
                "out_dir": str(args.out_dir),
                "wrist_bag": str(wrist_bag),
                "front_bag": str(front_bag),
                "wrist_serial": args.wrist_serial,
                "front_serial": args.front_serial,
                "fps": args.fps,
                "resolution": [args.width, args.height],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    started_wall = time.time()
    trajectory_summary: dict[str, Any] | None = None
    trajectory_command_summary: dict[str, Any] | None = None
    try:
        for recorder in recorders:
            recorder.start()
        time.sleep(max(0.0, float(args.pre_record_s)))

        command_proc: subprocess.Popen | None = None
        if args.trajectory_command.strip():
            command_proc = subprocess.Popen(args.trajectory_command, shell=True)
            trajectory_command_summary = {
                "command": args.trajectory_command,
                "pid": command_proc.pid,
                "start_wall_time": time.time(),
            }

        if args.reference_action_npz is not None:
            replayer = Z1ReferenceReplayer(
                args.reference_action_npz,
                action_host=args.action_host,
                action_port=args.action_port,
                hz=args.trajectory_hz,
                start_step=args.trajectory_start_step,
                max_steps=args.trajectory_max_steps,
                send_home_joint_targets_from_step=args.send_home_joint_targets_from_step,
                dry_run=args.dry_run_trajectory,
            )
            replayer.run()
            trajectory_summary = replayer.summary
        elif command_proc is None:
            time.sleep(max(0.0, float(args.duration_s)))

        if command_proc is not None:
            ret = command_proc.wait()
            assert trajectory_command_summary is not None
            trajectory_command_summary.update({"returncode": ret, "end_wall_time": time.time()})
            if ret != 0:
                raise RuntimeError(f"trajectory_command exited with {ret}: {args.trajectory_command}")

        time.sleep(max(0.0, float(args.post_record_s)))
    finally:
        for recorder in recorders:
            try:
                recorder.stop()
            except Exception as exc:
                recorder.error = f"stop:{type(exc).__name__}: {exc}"
        if args.send_shutdown_back_to_start:
            send_bridge_shutdown_back_to_start(args.action_host, args.action_port)

    manifest = {
        "script": Path(__file__).name,
        "started_wall_time": started_wall,
        "finished_wall_time": time.time(),
        "out_dir": str(args.out_dir),
        "bags": {recorder.name: str(recorder.bag_path) for recorder in recorders},
        "cameras": {recorder.name: recorder.metadata() for recorder in recorders},
        "trajectory": trajectory_summary,
        "trajectory_command": trajectory_command_summary,
        "pre_record_s": float(args.pre_record_s),
        "post_record_s": float(args.post_record_s),
        "notes": [
            "Bags contain raw RealSense streams; use offline_realsense_filter_sweep.py for A-E filter comparisons.",
            "Robot-base velocity is never published by this script.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("record_done " + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
