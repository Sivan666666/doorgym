#!/usr/bin/env python3
"""Replay gripper targets from a prior ACT jsonl through z1_act_ee_bridge.

WARNING: this still sends a full EE action to z1_act_ee_bridge.  Even if the EE
pose is copied from the current bridge state, the bridge will solve IK and send
6-DOF arm LOWCMD, so arm joints can move.  This script is therefore NOT a pure
gripper-only diagnostic.  Use it only with --allow_ee_hold and only when small
arm motion is acceptable.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import statistics
import time
from pathlib import Path
from typing import Any


ACT_DIM = 10


def _isfinite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_gripper_sequence(path: Path, *, max_steps: int | None = None) -> list[tuple[int, float]]:
    seq: list[tuple[int, float]] = []
    with path.open(errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "step" not in obj:
                continue
            action = obj.get("action")
            if not isinstance(action, list) or len(action) <= 9:
                continue
            value = action[9]
            if not _isfinite(value):
                continue
            seq.append((int(obj["step"]), float(value)))
            if max_steps is not None and len(seq) >= max_steps:
                break
    if not seq:
        raise RuntimeError(f"no action[9] gripper sequence found in {path}")
    return seq


class BridgeStateReceiver:
    def __init__(self, bind_host: str, port: int, timeout_s: float) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind_host, port))
        self.sock.settimeout(timeout_s)

    def close(self) -> None:
        self.sock.close()

    def read_latest(self, *, duration_s: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + max(0.0, float(duration_s))
        latest = None
        while time.monotonic() < deadline:
            try:
                data, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            try:
                latest = json.loads(data.decode("utf-8"))
            except Exception:
                continue
        return latest

    def wait_for_state(self, *, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        latest = None
        while time.monotonic() < deadline:
            latest = self.read_latest(duration_s=0.2)
            if latest and isinstance(latest.get("state"), list) and len(latest["state"]) >= ACT_DIM:
                return latest
        raise TimeoutError("timed out waiting for z1 bridge state UDP")


def send_action(sock: socket.socket, host: str, port: int, action: list[float]) -> None:
    sock.sendto(json.dumps({"action": action}, separators=(",", ":")).encode("utf-8"), (host, port))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    alpha = pos - lo
    return values[lo] * (1.0 - alpha) + values[hi] * alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_log", type=Path, required=True)
    parser.add_argument("--out_jsonl", type=Path, required=True)
    parser.add_argument("--target_host", type=str, default="127.0.0.1")
    parser.add_argument("--target_port", type=int, default=15011)
    parser.add_argument("--state_bind_host", type=str, default="127.0.0.1")
    parser.add_argument("--state_port", type=int, default=15013)
    parser.add_argument("--hz", type=float, default=25.0)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means all steps from source log.")
    parser.add_argument("--settle_s", type=float, default=1.0)
    parser.add_argument("--state_wait_s", type=float, default=5.0)
    parser.add_argument("--state_read_s", type=float, default=0.010)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--allow_ee_hold",
        action="store_true",
        help="Required because this sends full EE actions and may move arm joints.",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.allow_ee_hold:
        raise SystemExit(
            "Refusing to run: this script sends full EE actions through z1_act_ee_bridge "
            "and may move arm joints. Use --allow_ee_hold only if that is acceptable. "
            "For pure gripper testing, use a direct LOWCMD script that holds current q."
        )

    seq = load_gripper_sequence(args.source_log, max_steps=(args.max_steps or None))
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    state_rx = BridgeStateReceiver(args.state_bind_host, args.state_port, timeout_s=0.2)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    records: list[dict[str, Any]] = []

    try:
        initial = state_rx.wait_for_state(timeout_s=args.state_wait_s)
        initial_state = [float(x) for x in initial["state"][:ACT_DIM]]
        hold_ee = initial_state[2:9]
        initial_gripper = float(initial_state[9])
        print(
            "initial_state",
            json.dumps(
                {
                    "ee": [round(x, 6) for x in hold_ee],
                    "gripper": round(initial_gripper, 6),
                    "startup_zero_done": initial.get("startup_zero_done"),
                    "last_error": initial.get("last_error", ""),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        period = 1.0 / max(float(args.hz), 1.0e-6)
        next_t = time.monotonic()
        with args.out_jsonl.open("w") as f:
            for replay_i, (src_step, grip_target) in enumerate(seq):
                action = [0.0, 0.0] + hold_ee + [float(grip_target)]
                if not args.dry_run:
                    send_action(tx, args.target_host, args.target_port, action)
                feedback = state_rx.read_latest(duration_s=args.state_read_s)
                actual = None
                if feedback and isinstance(feedback.get("state"), list) and len(feedback["state"]) >= ACT_DIM:
                    actual = float(feedback["state"][9])
                rec = {
                    "phase": "replay",
                    "replay_i": replay_i,
                    "src_step": src_step,
                    "target_gripper": float(grip_target),
                    "actual_gripper": actual,
                    "error_rad": None if actual is None else float(grip_target) - actual,
                    "abs_error_rad": None if actual is None else abs(float(grip_target) - actual),
                    "bridge_last_error": "" if not feedback else feedback.get("last_error", ""),
                    "ik_ok": None if not feedback else feedback.get("ik_ok"),
                    "ik_source": "" if not feedback else feedback.get("ik_source", ""),
                    "action": action,
                    "wall_time": time.time(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                records.append(rec)

                next_t += period
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.monotonic()

            # Keep the final gripper target briefly so the last open/close command can settle.
            settle_steps = max(0, int(round(float(args.settle_s) * float(args.hz))))
            if settle_steps and seq:
                final_target = float(seq[-1][1])
                for settle_i in range(settle_steps):
                    action = [0.0, 0.0] + hold_ee + [final_target]
                    if not args.dry_run:
                        send_action(tx, args.target_host, args.target_port, action)
                    feedback = state_rx.read_latest(duration_s=args.state_read_s)
                    actual = None
                    if feedback and isinstance(feedback.get("state"), list) and len(feedback["state"]) >= ACT_DIM:
                        actual = float(feedback["state"][9])
                    rec = {
                        "phase": "settle",
                        "settle_i": settle_i,
                        "target_gripper": final_target,
                        "actual_gripper": actual,
                        "error_rad": None if actual is None else final_target - actual,
                        "abs_error_rad": None if actual is None else abs(final_target - actual),
                        "bridge_last_error": "" if not feedback else feedback.get("last_error", ""),
                        "ik_ok": None if not feedback else feedback.get("ik_ok"),
                        "ik_source": "" if not feedback else feedback.get("ik_source", ""),
                        "wall_time": time.time(),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    records.append(rec)
                    next_t += period
                    sleep_s = next_t - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        next_t = time.monotonic()
    finally:
        state_rx.close()
        tx.close()

    errors = [
        float(r["abs_error_rad"])
        for r in records
        if r.get("phase") == "replay" and _isfinite(r.get("abs_error_rad"))
    ]
    summary = {
        "source_log": str(args.source_log),
        "out_jsonl": str(args.out_jsonl),
        "steps": len(seq),
        "valid_feedback": len(errors),
        "target_min": min(v for _, v in seq),
        "target_max": max(v for _, v in seq),
        "abs_error_rad": {
            "mean": statistics.mean(errors) if errors else None,
            "p50": percentile(errors, 50.0),
            "p95": percentile(errors, 95.0),
            "max": max(errors) if errors else None,
        },
    }
    print("gripper_replay_summary", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
