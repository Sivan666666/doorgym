#!/usr/bin/env python3
"""Replay only the scripted A2W arm trajectory before push and compare q.

The input reference is a compact NPZ containing:

    action: [T, 10] Door-ACT actions
    sim_q:  [T, 6] IsaacGym joint1..joint6 feedback

No ROS publisher is created.  action[0:2] is always overwritten with zero, so
this tool controls only the Z1 bridge over its local UDP action interface.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np


class LatestStateReceiver:
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.settimeout(0.1)
        self.lock = threading.Lock()
        self.latest: dict | None = None
        self.latest_stamp = 0.0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            try:
                state = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            with self.lock:
                self.latest = state
                self.latest_stamp = time.monotonic()

    def snapshot(self) -> tuple[dict | None, float]:
        with self.lock:
            return self.latest, self.latest_stamp

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.sock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--action_host", default="127.0.0.1")
    parser.add_argument("--action_port", type=int, default=15011)
    parser.add_argument("--state_bind_host", default="127.0.0.1")
    parser.add_argument("--state_port", type=int, default=15013)
    parser.add_argument("--hz", type=float, default=25.0)
    parser.add_argument("--hold_final_s", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.reference) as data:
        actions = np.asarray(data["action"], dtype=np.float32).copy()
        sim_q = np.asarray(data["sim_q"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 10:
        raise ValueError(f"expected action [T,10], got {actions.shape}")
    if sim_q.shape != (len(actions), 6):
        raise ValueError(f"expected sim_q [{len(actions)},6], got {sim_q.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(sim_q).all():
        raise ValueError("reference contains NaN or Inf")

    # This executable never commands the robot base.
    actions[:, :2] = 0.0
    receiver = LatestStateReceiver(args.state_bind_host, args.state_port)
    receiver.start()
    action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    action_addr = (args.action_host, int(args.action_port))
    period = 1.0 / max(float(args.hz), 1.0e-6)
    records: list[dict] = []

    try:
        deadline = time.monotonic() + 5.0
        while receiver.snapshot()[0] is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if receiver.snapshot()[0] is None:
            raise RuntimeError("no Z1 bridge state received")

        next_t = time.monotonic()
        return_home_start = 364
        for frame, (action, q_ref) in enumerate(zip(actions, sim_q)):
            payload = {"action": [float(x) for x in action]}
            if frame >= return_home_start:
                payload.update(
                    {
                        "control_mode": "home",
                        "joint_target": [float(x) for x in q_ref],
                    }
                )
            action_sock.sendto(
                json.dumps(payload).encode("utf-8"),
                action_addr,
            )
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()
            state, state_stamp = receiver.snapshot()
            q_real = np.asarray((state or {}).get("q", [np.nan] * 6), dtype=np.float64)
            gripper_real = float((state or {}).get("gripper", np.nan))
            records.append(
                {
                    "frame": frame,
                    "state_age_s": round(time.monotonic() - state_stamp, 6),
                    "command_ee": action[2:5].astype(float).tolist(),
                    "command_gripper": float(action[9]),
                    "sim_q": q_ref.astype(float).tolist(),
                    "real_q": q_real.astype(float).tolist(),
                    "q_error_real_minus_sim": (q_real - q_ref).astype(float).tolist(),
                    "real_ee": (state or {}).get("ee_pos"),
                    "real_gripper": gripper_real,
                    "gripper_error_real_minus_command": gripper_real - float(action[9]),
                    "ik_ok": (state or {}).get("ik_ok"),
                    "ik_source": (state or {}).get("ik_source"),
                    "last_error": (state or {}).get("last_error"),
                }
            )

        hold_end = time.monotonic() + max(float(args.hold_final_s), 0.0)
        final_payload = {
            "action": [float(x) for x in actions[-1]],
            "control_mode": "home",
            "joint_target": [float(x) for x in sim_q[-1]],
        }
        while time.monotonic() < hold_end:
            action_sock.sendto(
                json.dumps(final_payload).encode("utf-8"),
                action_addr,
            )
            time.sleep(period)

        final_state, _ = receiver.snapshot()
        final_real_q = np.asarray((final_state or {}).get("q", [np.nan] * 6), dtype=np.float64)
        final_real_gripper = float((final_state or {}).get("gripper", np.nan))
        final_error = final_real_q - sim_q[-1]
        valid_errors = np.asarray(
            [row["q_error_real_minus_sim"] for row in records], dtype=np.float64
        )
        summary = {
            "frames": len(actions),
            "hz": float(args.hz),
            "dog_commands_published": False,
            "sim_final_q": sim_q[-1].astype(float).tolist(),
            "real_final_q": final_real_q.astype(float).tolist(),
            "final_q_error_real_minus_sim": final_error.astype(float).tolist(),
            "final_max_abs_q_error": float(np.nanmax(np.abs(final_error))),
            "trajectory_joint_rmse": np.sqrt(np.nanmean(valid_errors**2, axis=0)).tolist(),
            "trajectory_max_abs_q_error": float(np.nanmax(np.abs(valid_errors))),
            "real_final_ee": (final_state or {}).get("ee_pos"),
            "command_final_ee": actions[-1, 2:5].astype(float).tolist(),
            "command_final_gripper": float(actions[-1, 9]),
            "real_final_gripper": final_real_gripper,
            "final_gripper_error_real_minus_command": final_real_gripper - float(actions[-1, 9]),
            "final_ik_ok": (final_state or {}).get("ik_ok"),
            "final_ik_source": (final_state or {}).get("ik_source"),
            "final_last_error": (final_state or {}).get("last_error"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        action_sock.close()
        receiver.close()


if __name__ == "__main__":
    main()
