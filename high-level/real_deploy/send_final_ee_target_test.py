#!/usr/bin/env python3
"""Send one final Door-ACT EE target repeatedly over UDP.

This is a small real-robot test sender for z1_act_ee_bridge.py.  It reads the
latest ACT state from the bridge JSONL log, builds one final target

    target_ee = current_ee + [dx, dy, dz]

and then repeatedly sends that same 10D action.  It does not interpolate EE
poses; if the bridge runs in --joint_command_mode direct_hold, the bridge should
solve IK once for this target and then keep sending the same q_goal.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any


def load_latest_state(log_path: Path) -> dict[str, Any]:
    for _ in range(100):
        if log_path.is_file():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines[-500:]):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                state = record.get("state") or []
                ee_pos = record.get("ee_pos") or []
                if len(state) >= 10 and len(ee_pos) == 3 and any(abs(float(x)) > 1.0e-6 for x in ee_pos):
                    return record
        time.sleep(0.1)
    raise RuntimeError(f"no valid ACT state found in {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one final EE target to z1_act_ee_bridge.py.")
    parser.add_argument("--log_path", type=Path, required=True, help="Bridge JSONL log used to read the current ACT state.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15011)
    parser.add_argument("--dx", type=float, default=0.10)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.10)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--hz", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = load_latest_state(args.log_path)
    state = record["state"]
    start = [float(state[2]), float(state[3]), float(state[4])]
    quat = [float(state[5]), float(state[6]), float(state[7]), float(state[8])]
    gripper = float(state[9])
    target = [start[0] + args.dx, start[1] + args.dy, start[2] + args.dz]
    action = [0.0, 0.0] + target + quat + [gripper]

    print(
        json.dumps(
            {
                "mode": "single_final_target_only",
                "start": start,
                "target": target,
                "quat": quat,
                "gripper": gripper,
                "duration": args.duration,
                "hz": args.hz,
            },
            indent=2,
        ),
        flush=True,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, int(args.port))
    period = 1.0 / max(float(args.hz), 1.0e-6)
    end_t = time.monotonic() + max(float(args.duration), 0.0)
    sent = 0
    next_report = time.monotonic()
    while time.monotonic() < end_t:
        sock.sendto(json.dumps({"action": action}).encode("utf-8"), addr)
        sent += 1
        now = time.monotonic()
        if now >= next_report:
            try:
                latest = load_latest_state(args.log_path)
            except Exception:
                latest = {}
            print(
                json.dumps(
                    {
                        "t_left": round(end_t - now, 2),
                        "sent": sent,
                        "meas_ee": latest.get("ee_pos"),
                        "ik_ok": latest.get("ik_ok"),
                        "ik_source": latest.get("ik_source"),
                        "err": latest.get("last_error"),
                        "action_age": latest.get("action_age_s"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            next_report = now + 0.5
        time.sleep(period)

    latest = load_latest_state(args.log_path)
    print(
        "FINAL "
        + json.dumps(
            {
                "sent": sent,
                "meas_ee": latest.get("ee_pos"),
                "state": latest.get("state"),
                "ik_ok": latest.get("ik_ok"),
                "ik_source": latest.get("ik_source"),
                "err": latest.get("last_error"),
                "q": latest.get("q"),
                "qd": latest.get("qd"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
