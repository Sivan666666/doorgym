#!/usr/bin/env python3
"""Replay a recorded Door-ACT joint9 episode through the real Z1 bridge.

Only the six arm joints and gripper are consumed by the joint9 bridge.  The
recorded vx/vyaw fields are always overwritten with zero, and this utility
does not create a ROS publisher, so it cannot command the robot base.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--action_host", default="127.0.0.1")
    parser.add_argument("--action_port", type=int, default=15011)
    parser.add_argument("--state_bind_host", default="0.0.0.0")
    parser.add_argument("--state_port", type=int, default=15013)
    parser.add_argument("--hz", type=float, default=0.0, help="0 uses episode fps")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="0 replays to episode end")
    parser.add_argument("--startup_timeout_s", type=float, default=30.0)
    parser.add_argument("--settle_s", type=float, default=0.5)
    parser.add_argument("--shutdown_timeout_s", type=float, default=20.0)
    parser.add_argument("--log_path", type=Path, required=True)
    parser.add_argument("--no_shutdown_back_to_start", action="store_true")
    return parser.parse_args()


def recv_latest(sock: socket.socket, timeout_s: float) -> dict | None:
    deadline = time.monotonic() + max(0.0, timeout_s)
    latest = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return latest
        sock.settimeout(remaining)
        try:
            payload, _ = sock.recvfrom(65535)
        except socket.timeout:
            return latest
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except Exception:
            continue
        if isinstance(decoded, dict) and decoded.get("type") == "act_state":
            latest = decoded
        sock.setblocking(False)
        try:
            while True:
                payload, _ = sock.recvfrom(65535)
                decoded = json.loads(payload.decode("utf-8"))
                if isinstance(decoded, dict) and decoded.get("type") == "act_state":
                    latest = decoded
        except (BlockingIOError, json.JSONDecodeError, UnicodeDecodeError):
            pass
        finally:
            sock.setblocking(True)
        return latest


def finite_vector(record: dict, key: str, length: int) -> np.ndarray | None:
    try:
        value = np.asarray(record[key], dtype=np.float64).reshape(length)
    except Exception:
        return None
    return value if np.isfinite(value).all() else None


def percentile_summary(values: np.ndarray) -> dict[str, float | list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "per_joint_p95": np.percentile(values, 95, axis=0).tolist(),
        "per_joint_max": np.max(values, axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    with np.load(args.episode, allow_pickle=True) as data:
        actions = np.asarray(data["action"], dtype=np.float64).copy()
        episode_hz = float(np.asarray(data["fps"]).item())
    if actions.ndim != 2 or actions.shape[1] != 9:
        raise ValueError(f"joint9 episode action must have shape [T,9], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("episode action contains NaN or Inf")
    start = max(0, int(args.start))
    end = len(actions) if args.steps <= 0 else min(len(actions), start + int(args.steps))
    if start >= end:
        raise ValueError(f"empty replay range [{start}, {end})")
    actions = actions[start:end]
    actions[:, :2] = 0.0
    hz = float(args.hz) if args.hz > 0.0 else episode_hz
    period = 1.0 / hz

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    state_sock.bind((args.state_bind_host, int(args.state_port)))
    action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    action_addr = (args.action_host, int(args.action_port))

    startup_deadline = time.monotonic() + float(args.startup_timeout_s)
    startup = None
    while time.monotonic() < startup_deadline:
        record = recv_latest(state_sock, 0.5)
        if record is None:
            continue
        q = finite_vector(record, "q", 6)
        qd = finite_vector(record, "qd", 6)
        if (
            record.get("state_action_mode") == "joint9"
            and bool(record.get("arm_enabled"))
            and not bool(record.get("dry_run"))
            and bool(record.get("startup_zero_done"))
            and q is not None
            and qd is not None
        ):
            startup = record
            break
    if startup is None:
        raise TimeoutError("joint9 Z1 bridge did not report a ready real-arm startup state")

    rows: list[dict] = []
    start_mono = time.monotonic()
    next_t = start_mono
    with args.log_path.open("w", encoding="utf-8") as log_file:
        for local_step, action in enumerate(actions):
            episode_step = start + local_step
            send_wall = time.time()
            packet = {
                "action": action.tolist(),
                "source": "replay_joint9_episode_z1",
                "episode_step": episode_step,
                "dog_commands_published": False,
            }
            action_sock.sendto(json.dumps(packet).encode("utf-8"), action_addr)
            state = recv_latest(state_sock, min(0.01, period * 0.5))
            row = {
                "episode_step": episode_step,
                "send_wall_time": send_wall,
                "target": action.tolist(),
                "state": state,
            }
            rows.append(row)
            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if local_step % max(1, int(round(hz))) == 0:
                q = None if state is None else finite_vector(state, "q", 6)
                err = float("nan") if q is None else float(np.max(np.abs(q - action[2:8])))
                print(
                    f"replay step={episode_step:04d}/{end - 1:04d} "
                    f"target_q={np.round(action[2:8], 3).tolist()} "
                    f"max_target_err={err:.3f} grip_target={action[8]:+.3f}",
                    flush=True,
                )
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()

        settle_deadline = time.monotonic() + max(0.0, float(args.settle_s))
        while time.monotonic() < settle_deadline:
            action_sock.sendto(
                json.dumps({"action": actions[-1].tolist(), "source": "replay_joint9_episode_z1_settle"}).encode(
                    "utf-8"
                ),
                action_addr,
            )
            recv_latest(state_sock, min(0.01, period * 0.5))
            time.sleep(period)

        shutdown_done = False
        if not args.no_shutdown_back_to_start:
            shutdown_packet = json.dumps(
                {"bridge_command": "shutdown_back_to_start", "source": "replay_joint9_episode_z1"}
            ).encode("utf-8")
            for _ in range(3):
                action_sock.sendto(shutdown_packet, action_addr)
                time.sleep(0.05)
            deadline = time.monotonic() + float(args.shutdown_timeout_s)
            while time.monotonic() < deadline:
                state = recv_latest(state_sock, 0.5)
                if state is not None and bool(state.get("shutdown_done")):
                    shutdown_done = True
                    break

    samples = []
    gripper_samples = []
    qd_samples = []
    target_samples = []
    for row in rows:
        state = row["state"]
        if state is None:
            continue
        q = finite_vector(state, "q", 6)
        qd = finite_vector(state, "qd", 6)
        if q is None or qd is None:
            continue
        target = np.asarray(row["target"], dtype=np.float64)
        samples.append(q)
        qd_samples.append(qd)
        target_samples.append(target[2:8])
        try:
            gripper_samples.append(abs(float(state["gripper"]) - float(target[8])))
        except Exception:
            pass
    if not samples:
        raise RuntimeError("replay completed but no finite Z1 feedback samples were received")
    actual_q = np.asarray(samples)
    target_q = np.asarray(target_samples)
    q_error = np.abs(actual_q - target_q)
    qd_abs = np.abs(np.asarray(qd_samples))

    # Estimate the causal target/feedback lag that minimizes median L1 error.
    lag_scores: list[tuple[int, float]] = []
    for lag in range(0, min(31, len(actual_q) - 1)):
        lhs = actual_q[lag:]
        rhs = target_q[: len(target_q) - lag]
        lag_scores.append((lag, float(np.median(np.abs(lhs - rhs)))))
    best_lag, best_lag_error = min(lag_scores, key=lambda item: item[1])
    summary = {
        "episode": str(args.episode),
        "range": [start, end],
        "hz": hz,
        "sent": int(len(actions)),
        "elapsed_s": time.monotonic() - start_mono,
        "dog_commands_published": False,
        "startup_home_q": startup.get("startup_home_q"),
        "feedback_samples": int(len(actual_q)),
        "raw_target_vs_actual_abs_error_rad": percentile_summary(q_error),
        "measured_abs_joint_speed_rad_s": percentile_summary(qd_abs),
        "gripper_target_vs_actual_abs_error_rad": (
            None
            if not gripper_samples
            else {
                "p50": float(np.percentile(gripper_samples, 50)),
                "p95": float(np.percentile(gripper_samples, 95)),
                "max": float(np.max(gripper_samples)),
            }
        ),
        "best_causal_lag": {
            "frames": int(best_lag),
            "ms": 1000.0 * best_lag / hz,
            "median_abs_error_rad": best_lag_error,
        },
        "shutdown_back_to_start_requested": not args.no_shutdown_back_to_start,
        "shutdown_back_to_start_done": shutdown_done,
        "log_path": str(args.log_path),
    }
    summary_path = args.log_path.with_suffix(args.log_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("joint9_replay_summary " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
