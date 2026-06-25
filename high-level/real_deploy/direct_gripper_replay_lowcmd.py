#!/usr/bin/env python3
"""Replay ACT gripper targets with direct Z1 LOWCMD while holding arm joints fixed.

This script is intentionally independent of z1_act_ee_bridge.  It reads a prior
ACT jsonl, extracts action[9], captures the current 6-DOF arm joint position as
q_hold, and then sends direct LOWCMD at 500 Hz:

    setArmCmd(q_hold, zeros, inverseDynamics(q_hold, zeros, zeros, zeros))
    setGripperCmd(smoothed_gripper_cmd, gripper_qd_cmd, 0)

The goal is to test gripper tracking without EE IK moving arm joints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _setup_z1_sdk_path() -> None:
    candidates = []
    env = os.environ.get("Z1_SDK_LIB")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/home/anx/door_act_deploy/z1_sdk/lib"),
            Path("/home/sivan/whole_body/z1/z1_sdk/lib"),
            Path(__file__).resolve().parents[3] / "z1" / "z1_sdk" / "lib",
        ]
    )
    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path))


_setup_z1_sdk_path()

try:
    import unitree_arm_interface  # type: ignore
except Exception as exc:  # pragma: no cover - depends on NX SDK install
    raise RuntimeError(
        "Failed to import unitree_arm_interface. Set Z1_SDK_LIB to the Z1 SDK lib directory."
    ) from exc


ACT_DIM = 10
GRIPPER_INDEX = 9


def isfinite_number(x: Any) -> bool:
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
            if not isinstance(action, list) or len(action) <= GRIPPER_INDEX:
                continue
            value = action[GRIPPER_INDEX]
            if not isfinite_number(value):
                continue
            seq.append((int(obj["step"]), float(value)))
            if max_steps is not None and len(seq) >= max_steps:
                break
    if not seq:
        raise RuntimeError(f"no action[9] gripper sequence found in {path}")
    return seq


def advance_gripper_command(
    current_position: float,
    current_velocity: float,
    target_position: float,
    dt: float,
    max_speed: float,
    max_acceleration: float,
    position_min: float,
    position_max: float,
) -> tuple[float, float]:
    """Same online velocity/acceleration-limited gripper tracker used by bridge."""

    period = max(float(dt), 1.0e-6)
    speed_limit = max(float(max_speed), 0.0)
    acceleration_limit = max(float(max_acceleration), 1.0e-6)
    q_min = float(min(position_min, position_max))
    q_max = float(max(position_min, position_max))
    position = float(np.clip(current_position, q_min, q_max))
    target = float(np.clip(target_position, q_min, q_max))
    velocity = float(np.clip(current_velocity, -speed_limit, speed_limit))
    error = target - position

    if abs(error) <= 1.0e-9 and abs(velocity) <= acceleration_limit * period:
        return target, 0.0

    direction = math.copysign(1.0, error) if error != 0.0 else 0.0
    velocity_toward_target = velocity * direction
    stopping_distance = (
        max(velocity_toward_target, 0.0) ** 2 / (2.0 * acceleration_limit)
        + 2.0 * max(velocity_toward_target, 0.0) * period
    )
    should_brake = direction == 0.0 or (
        velocity_toward_target >= 0.0 and abs(error) <= stopping_distance
    )
    desired_velocity = 0.0 if should_brake else direction * speed_limit
    velocity += float(
        np.clip(
            desired_velocity - velocity,
            -acceleration_limit * period,
            acceleration_limit * period,
        )
    )
    velocity = float(np.clip(velocity, -speed_limit, speed_limit))
    next_position = position + velocity * period

    if (error > 0.0 and next_position >= target) or (error < 0.0 and next_position <= target):
        return target, 0.0
    return float(np.clip(next_position, q_min, q_max)), velocity


def apply_gripper_close_latch(
    mapped_goal: float,
    *,
    enabled: bool,
    threshold: float,
    confirm_steps: int,
    close_target: float,
    gripper_min: float,
    gripper_max: float,
    previous_count: int,
    previous_forced: bool,
    close_target_max: float | None = None,
) -> tuple[float, int, bool]:
    """Match z1_act_ee_bridge gripper close latch semantics.

    Count only once per fresh ACT action packet.  One command below threshold
    exits the latch immediately; there is intentionally no release hysteresis.
    """

    goal = float(np.clip(float(mapped_goal), float(gripper_min), float(gripper_max)))
    if not bool(enabled):
        return goal, 0, False

    confirm_steps = max(1, int(confirm_steps))
    if goal >= float(threshold):
        count = int(previous_count) + 1
        forced = bool(previous_forced) or count >= confirm_steps
    else:
        count = 0
        forced = False

    if forced:
        forced_max = float(gripper_max) if close_target_max is None else float(close_target_max)
        forced_max = max(forced_max, float(gripper_max))
        goal = float(np.clip(float(close_target), float(gripper_min), forced_max))
    return goal, count, forced


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


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_log", type=Path, required=True)
    parser.add_argument("--out_jsonl", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=25.0, help="ACT target replay frequency.")
    parser.add_argument("--control_hz", type=float, default=500.0, help="Direct LOWCMD frequency.")
    parser.add_argument("--log_hz", type=float, default=25.0)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means all source steps.")
    parser.add_argument("--settle_s", type=float, default=1.0)
    parser.add_argument("--max_gripper_speed", type=float, default=3.14)
    parser.add_argument("--max_gripper_acceleration", type=float, default=120.0)
    parser.add_argument("--gripper_min", type=float, default=-math.pi / 2.0)
    parser.add_argument("--gripper_max", type=float, default=0.0)
    parser.add_argument("--gripper_close_latch", dest="gripper_close_latch", action="store_true")
    parser.add_argument("--no_gripper_close_latch", dest="gripper_close_latch", action="store_false")
    parser.set_defaults(gripper_close_latch=True)
    parser.add_argument("--gripper_close_latch_threshold", type=float, default=-0.5)
    parser.add_argument("--gripper_close_latch_confirm_steps", type=int, default=3)
    parser.add_argument("--gripper_close_latch_target", type=float, default=0.1)
    parser.add_argument("--q_drift_abort_rad", type=float, default=0.25)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if int(args.gripper_close_latch_confirm_steps) <= 0:
        raise ValueError("--gripper_close_latch_confirm_steps must be positive.")
    if float(args.gripper_min) > float(args.gripper_max):
        raise ValueError("--gripper_min must be <= --gripper_max.")
    if float(args.gripper_close_latch_target) < float(args.gripper_min):
        raise ValueError("--gripper_close_latch_target must be >= --gripper_min.")

    seq = load_gripper_sequence(args.source_log, max_steps=(args.max_steps or None))
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    arm_model = arm._ctrlComp.armModel
    dt = 1.0 / max(float(args.control_hz), 1.0)
    if hasattr(arm, "_ctrlComp") and getattr(arm._ctrlComp, "dt", None):
        dt = float(arm._ctrlComp.dt)
    replay_period = 1.0 / max(float(args.hz), 1.0e-6)
    log_period = 1.0 / max(float(args.log_hz), 1.0e-6)

    if not args.dry_run:
        arm.setFsmLowcmd()
        time.sleep(0.2)

    q_hold = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
    qd_zero = np.zeros(6, dtype=np.float64)
    qdd_zero = np.zeros(6, dtype=np.float64)
    tau_hold = np.asarray(arm_model.inverseDynamics(q_hold, qd_zero, qdd_zero, qd_zero), dtype=np.float64).reshape(6)
    gripper_cmd = float(np.clip(float(arm.lowstate.getGripperQ()), args.gripper_min, args.gripper_max))
    gripper_qd_cmd = 0.0

    print(
        "direct_gripper_replay_start",
        json.dumps(
            {
                "source_log": str(args.source_log),
                "out_jsonl": str(args.out_jsonl),
                "steps": len(seq),
                "control_dt": dt,
                "q_hold": np.round(q_hold, 6).tolist(),
                "initial_gripper": round(gripper_cmd, 6),
                "max_gripper_speed": args.max_gripper_speed,
                "max_gripper_acceleration": args.max_gripper_acceleration,
                "gripper_close_latch": bool(args.gripper_close_latch),
                "gripper_close_latch_threshold": float(args.gripper_close_latch_threshold),
                "gripper_close_latch_confirm_steps": int(args.gripper_close_latch_confirm_steps),
                "gripper_close_latch_target": float(args.gripper_close_latch_target),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    target_errors: list[float] = []
    raw_target_errors: list[float] = []
    cmd_errors: list[float] = []
    q_drift_norms: list[float] = []
    q_drift_max_abs: list[float] = []
    records_written = 0
    control_count = 0

    total_replay_s = len(seq) * replay_period
    total_s = total_replay_s + max(0.0, float(args.settle_s))
    start_mono = time.monotonic()
    next_control = start_mono
    next_log = start_mono
    last_action_index = -1
    last_latch_action_index = -1
    gripper_close_latch_count = 0
    gripper_force_closed = False
    current_raw_target = float(seq[0][1])
    current_target = current_raw_target
    current_src_step = int(seq[0][0])

    with args.out_jsonl.open("w") as f:
        while True:
            now = time.monotonic()
            elapsed = now - start_mono
            if elapsed > total_s:
                break

            if elapsed < total_replay_s:
                action_index = min(len(seq) - 1, int(elapsed / replay_period))
            else:
                action_index = len(seq) - 1
            current_src_step, current_raw_target = seq[action_index]
            phase = "replay" if elapsed < total_replay_s else "settle"
            if action_index != last_latch_action_index:
                current_target, gripper_close_latch_count, gripper_force_closed = apply_gripper_close_latch(
                    current_raw_target,
                    enabled=bool(args.gripper_close_latch),
                    threshold=float(args.gripper_close_latch_threshold),
                    confirm_steps=int(args.gripper_close_latch_confirm_steps),
                    close_target=float(args.gripper_close_latch_target),
                    gripper_min=float(args.gripper_min),
                    gripper_max=float(args.gripper_max),
                    previous_count=gripper_close_latch_count,
                    previous_forced=gripper_force_closed,
                    close_target_max=max(float(args.gripper_max), float(args.gripper_close_latch_target)),
                )
                last_latch_action_index = action_index
            gripper_command_max = (
                max(float(args.gripper_max), float(args.gripper_close_latch_target))
                if bool(gripper_force_closed)
                else float(args.gripper_max)
            )

            gripper_cmd, gripper_qd_cmd = advance_gripper_command(
                gripper_cmd,
                gripper_qd_cmd,
                current_target,
                dt,
                args.max_gripper_speed,
                args.max_gripper_acceleration,
                args.gripper_min,
                gripper_command_max,
            )

            if not args.dry_run:
                arm.setArmCmd(q_hold, qd_zero, tau_hold)
                arm.setGripperCmd(gripper_cmd, gripper_qd_cmd, 0.0)
                arm.sendRecv()

            q_now = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
            qd_now = np.asarray(arm.lowstate.getQd(), dtype=np.float64).reshape(6)
            gripper_actual = float(arm.lowstate.getGripperQ())
            q_drift = q_now - q_hold
            drift_norm = float(np.linalg.norm(q_drift))
            drift_max = float(np.max(np.abs(q_drift)))
            q_drift_norms.append(drift_norm)
            q_drift_max_abs.append(drift_max)

            if drift_max > float(args.q_drift_abort_rad):
                raise RuntimeError(
                    f"joint drift exceeded abort threshold: max_abs={drift_max:.4f} rad, "
                    f"q_now={q_now.tolist()}, q_hold={q_hold.tolist()}"
                )

            target_abs_error = abs(float(current_target) - gripper_actual)
            raw_target_abs_error = abs(float(current_raw_target) - gripper_actual)
            cmd_abs_error = abs(float(gripper_cmd) - gripper_actual)
            if phase == "replay":
                target_errors.append(target_abs_error)
                raw_target_errors.append(raw_target_abs_error)
                cmd_errors.append(cmd_abs_error)

            if now >= next_log or action_index != last_action_index:
                rec = {
                    "phase": phase,
                    "control_count": control_count,
                    "elapsed_s": elapsed,
                    "action_index": action_index,
                    "src_step": int(current_src_step),
                    "raw_target_gripper": float(current_raw_target),
                    "target_gripper": float(current_target),
                    "gripper_close_latch_count": int(gripper_close_latch_count),
                    "gripper_force_closed": bool(gripper_force_closed),
                    "cmd_gripper": float(gripper_cmd),
                    "cmd_gripper_qd": float(gripper_qd_cmd),
                    "actual_gripper": gripper_actual,
                    "raw_target_error_rad": float(current_raw_target) - gripper_actual,
                    "raw_target_abs_error_rad": raw_target_abs_error,
                    "target_error_rad": float(current_target) - gripper_actual,
                    "target_abs_error_rad": target_abs_error,
                    "cmd_error_rad": float(gripper_cmd) - gripper_actual,
                    "cmd_abs_error_rad": cmd_abs_error,
                    "q_hold": np.round(q_hold, 6).tolist(),
                    "q_now": np.round(q_now, 6).tolist(),
                    "qd_now": np.round(qd_now, 6).tolist(),
                    "q_drift_norm_rad": drift_norm,
                    "q_drift_max_abs_rad": drift_max,
                    "wall_time": time.time(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records_written += 1
                next_log = now + log_period
                last_action_index = action_index

            control_count += 1
            next_control += dt
            sleep_s = next_control - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_control = time.monotonic()

    summary = {
        "source_log": str(args.source_log),
        "out_jsonl": str(args.out_jsonl),
        "steps": len(seq),
        "records_written": records_written,
        "control_count": control_count,
        "target_min": min(v for _, v in seq),
        "target_max": max(v for _, v in seq),
        "gripper_close_latch": bool(args.gripper_close_latch),
        "gripper_close_latch_threshold": float(args.gripper_close_latch_threshold),
        "gripper_close_latch_confirm_steps": int(args.gripper_close_latch_confirm_steps),
        "gripper_close_latch_target": float(args.gripper_close_latch_target),
        "initial_q_hold": np.round(q_hold, 6).tolist(),
        "final_q": np.round(np.asarray(arm.lowstate.getQ(), dtype=np.float64), 6).tolist(),
        "raw_target_abs_error_rad": stats(raw_target_errors),
        "target_abs_error_rad": stats(target_errors),
        "cmd_abs_error_rad": stats(cmd_errors),
        "q_drift_norm_rad": stats(q_drift_norms),
        "q_drift_max_abs_rad": stats(q_drift_max_abs),
        "max_gripper_speed": float(args.max_gripper_speed),
        "max_gripper_acceleration": float(args.max_gripper_acceleration),
    }
    print("direct_gripper_replay_summary", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
