#!/usr/bin/env python3
"""Keyboard EE teleop, recording, and replay for the Z1 ACT bridge.

This tool intentionally does *not* talk to the Unitree SDK directly.  It sends
the same 10D Door-ACT EE action packet consumed by z1_act_ee_bridge.py:

    [vx, yaw_rate, ee_x, ee_y, ee_z, qx, qy, qz, qw, gripper]

The bridge remains the only process that performs IK, joint speed limiting,
online quintic smoothing, gripper smoothing, and LOWCMD output.  That keeps this
keyboard tool small and makes it suitable for quick experiments without
duplicating any real-robot safety logic.

Typical usage on NX, after z1_ctrl and z1_act_ee_bridge.py are already running:

    python3 high-level/real_deploy/keyboard_z1_ee_teleop.py \\
      --record_path /tmp/z1_keyboard_$(date +%Y%m%d_%H%M%S).jsonl

Replay:

    python3 high-level/real_deploy/keyboard_z1_ee_teleop.py \\
      --replay_path /tmp/z1_keyboard_YYYYmmdd_HHMMSS.jsonl
"""

from __future__ import annotations

import argparse
import curses
import json
import math
import select
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ACT_DIM = 10
EE_POS_SLICE = slice(2, 5)
EE_QUAT_SLICE = slice(5, 9)
GRIPPER_INDEX = 9


def normalize_quat_xyzw(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1.0e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / n
    # Keep signs stable in logs.
    if q[3] < 0.0:
        q = -q
    return q


def quat_multiply_xyzw(a: Any, b: Any) -> np.ndarray:
    ax, ay, az, aw = normalize_quat_xyzw(a)
    bx, by, bz, bw = normalize_quat_xyzw(b)
    return normalize_quat_xyzw(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def axis_angle_quat_xyzw(axis: str, angle_rad: float) -> np.ndarray:
    half = 0.5 * float(angle_rad)
    s = math.sin(half)
    c = math.cos(half)
    if axis == "x":
        return np.asarray([s, 0.0, 0.0, c], dtype=np.float64)
    if axis == "y":
        return np.asarray([0.0, s, 0.0, c], dtype=np.float64)
    if axis == "z":
        return np.asarray([0.0, 0.0, s, c], dtype=np.float64)
    raise ValueError(f"unknown axis: {axis}")


def quat_to_rpy_xyz_intrinsic(q: Any) -> tuple[float, float, float]:
    """Return a readable roll/pitch/yaw estimate for display only."""
    x, y, z, w = normalize_quat_xyzw(q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class LatestStateReceiver:
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.setblocking(False)
        self.lock = threading.Lock()
        self.latest: dict[str, Any] | None = None
        self.latest_stamp_mono = 0.0
        self.count = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="z1-state-receiver")

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            readable, _, _ = select.select([self.sock], [], [], 0.1)
            if not readable:
                continue
            try:
                data, _ = self.sock.recvfrom(65536)
                record = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            with self.lock:
                self.latest = record
                self.latest_stamp_mono = time.monotonic()
                self.count += 1

    def snapshot(self) -> tuple[dict[str, Any] | None, float, int]:
        with self.lock:
            return self.latest, self.latest_stamp_mono, self.count

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.sock.close()


def extract_ee_from_state(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    state = record.get("state") or []
    ee_pos = record.get("ee_pos")
    if ee_pos is None and len(state) >= ACT_DIM:
        ee_pos = state[EE_POS_SLICE]
    ee_quat = (
        record.get("ee_quat_xyzw")
        or record.get("ee_quat")
        or record.get("quat_xyzw")
        or (state[EE_QUAT_SLICE] if len(state) >= ACT_DIM else None)
    )
    gripper = record.get("gripper")
    if gripper is None and len(state) >= ACT_DIM:
        gripper = state[GRIPPER_INDEX]
    if ee_pos is None or ee_quat is None or gripper is None:
        raise ValueError(f"state record does not contain EE pose/gripper: keys={sorted(record.keys())}")
    pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
    quat = normalize_quat_xyzw(ee_quat)
    grip = float(gripper)
    if not (np.isfinite(pos).all() and np.isfinite(quat).all() and np.isfinite(grip)):
        raise ValueError("state record contains NaN or Inf")
    return pos, quat, grip


def make_action(pos: np.ndarray, quat: np.ndarray, gripper: float) -> list[float]:
    action = np.zeros(ACT_DIM, dtype=np.float64)
    action[EE_POS_SLICE] = np.asarray(pos, dtype=np.float64).reshape(3)
    action[EE_QUAT_SLICE] = normalize_quat_xyzw(quat)
    action[GRIPPER_INDEX] = float(gripper)
    return [float(x) for x in action]


def send_action(sock: socket.socket, addr: tuple[str, int], action: list[float], dry_run: bool) -> None:
    if dry_run:
        return
    payload = {"control_mode": "ee", "action": action}
    sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), addr)


class JsonlRecorder:
    def __init__(self, path: Path | None, metadata: dict[str, Any]) -> None:
        self.path = path
        self.file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", encoding="utf-8")
            self.write({"kind": "metadata", **metadata})

    def write(self, record: dict[str, Any]) -> None:
        if self.file is None:
            return
        self.file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


@dataclass
class TargetState:
    pos: np.ndarray
    quat: np.ndarray
    gripper: float

    def clamp(self, args: argparse.Namespace) -> None:
        if args.min_xyz is not None:
            self.pos = np.maximum(self.pos, np.asarray(args.min_xyz, dtype=np.float64))
        if args.max_xyz is not None:
            self.pos = np.minimum(self.pos, np.asarray(args.max_xyz, dtype=np.float64))
        self.gripper = float(np.clip(self.gripper, float(args.gripper_min), float(args.gripper_max)))
        self.quat = normalize_quat_xyzw(self.quat)


def wait_for_initial_state(receiver: LatestStateReceiver, timeout_s: float) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    last_record: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        record, _, _ = receiver.snapshot()
        if record is not None:
            last_record = record
            try:
                pos, quat, gripper = extract_ee_from_state(record)
                return pos, quat, gripper, record
            except Exception:
                pass
        time.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for Z1 bridge state on UDP; last_record={last_record}")


def apply_key(target: TargetState, key: int, args: argparse.Namespace) -> str | None:
    step = float(args.position_step)
    rot_step = math.radians(float(args.rotation_step_deg))
    grip_step = float(args.gripper_step)
    name: str | None = None

    if key in (ord("w"), ord("W")):
        target.pos[0] += step
        name = "+x"
    elif key in (ord("s"), ord("S")):
        target.pos[0] -= step
        name = "-x"
    elif key in (ord("a"), ord("A")):
        target.pos[1] += step
        name = "+y"
    elif key in (ord("d"), ord("D")):
        target.pos[1] -= step
        name = "-y"
    elif key in (ord("r"), ord("R")):
        target.pos[2] += step
        name = "+z"
    elif key in (ord("f"), ord("F")):
        target.pos[2] -= step
        name = "-z"
    elif key in (ord("u"), ord("U")):
        target.quat = quat_multiply_xyzw(target.quat, axis_angle_quat_xyzw("x", +rot_step))
        name = "+roll"
    elif key in (ord("o"), ord("O")):
        target.quat = quat_multiply_xyzw(target.quat, axis_angle_quat_xyzw("x", -rot_step))
        name = "-roll"
    elif key in (ord("i"), ord("I")):
        target.quat = quat_multiply_xyzw(target.quat, axis_angle_quat_xyzw("y", +rot_step))
        name = "+pitch"
    elif key in (ord("k"), ord("K")):
        target.quat = quat_multiply_xyzw(target.quat, axis_angle_quat_xyzw("y", -rot_step))
        name = "-pitch"
    elif key in (ord("j"), ord("J")):
        target.quat = quat_multiply_xyzw(target.quat, axis_angle_quat_xyzw("z", +rot_step))
        name = "+yaw"
    elif key in (ord("l"), ord("L")):
        target.quat = quat_multiply_xyzw(target.quat, axis_angle_quat_xyzw("z", -rot_step))
        name = "-yaw"
    elif key == ord("["):
        target.gripper -= grip_step
        name = "open_gripper"
    elif key == ord("]"):
        target.gripper += grip_step
        name = "close_gripper"
    elif key in (ord("g"), ord("G")):
        target.gripper = float(args.gripper_min)
        name = "gripper_full_open"
    elif key in (ord("c"), ord("C")):
        target.gripper = float(args.gripper_max)
        name = "gripper_closed"
    elif key in (ord("h"), ord("H")):
        name = "hold_feedback"
    elif key in (ord("p"), ord("P")):
        name = "toggle_publish"
    elif key in (ord("q"), ord("Q"), 27):
        name = "quit"
    elif key == ord(" "):
        name = "mark"

    target.clamp(args)
    return name


def draw_live_screen(
    stdscr: Any,
    args: argparse.Namespace,
    target: TargetState,
    feedback: dict[str, Any] | None,
    enabled: bool,
    sent: int,
    elapsed_s: float,
    last_key_name: str,
    record_path: Path | None,
) -> None:
    stdscr.erase()
    rpy = tuple(math.degrees(x) for x in quat_to_rpy_xyz_intrinsic(target.quat))
    fb_age = float("inf")
    fb_pos = fb_quat = None
    fb_grip = None
    if feedback is not None:
        try:
            fb_pos, fb_quat, fb_grip = extract_ee_from_state(feedback)
        except Exception:
            pass
    if fb_pos is not None:
        fb_age = max(0.0, time.time() - float(feedback.get("wall_time", time.time())))
    stdscr.addstr(0, 0, "Z1 EE keyboard teleop via z1_act_ee_bridge UDP")
    stdscr.addstr(1, 0, f"publish={'ON' if enabled else 'OFF'} sent={sent} elapsed={elapsed_s:.1f}s dry_run={args.dry_run}")
    stdscr.addstr(2, 0, f"target xyz={np.round(target.pos, 4).tolist()}  rpy_deg={[round(x, 1) for x in rpy]}  grip={target.gripper:+.3f}")
    if fb_pos is not None and fb_grip is not None:
        err = float(np.linalg.norm(target.pos - fb_pos))
        stdscr.addstr(
            3,
            0,
            f"feedback xyz={np.round(fb_pos, 4).tolist()} grip={fb_grip:+.3f} "
            f"pos_err={1000.0 * err:.1f}mm ik={feedback.get('ik_ok')} src={feedback.get('ik_source')} "
            f"last_error={feedback.get('last_error')}",
        )
    else:
        stdscr.addstr(3, 0, f"feedback waiting... age={fb_age:.2f}s")
    stdscr.addstr(5, 0, "Keys:")
    stdscr.addstr(6, 2, "w/s: x +/-    a/d: y +/-    r/f: z +/-")
    stdscr.addstr(7, 2, "u/o: roll +/- i/k: pitch +/- j/l: yaw +/-")
    stdscr.addstr(8, 2, "[: gripper open step   ]: close step   g: full open   c: closed")
    stdscr.addstr(9, 2, "h: reset target to current feedback   p: pause/resume publish   SPACE: mark   q/ESC: quit")
    stdscr.addstr(11, 0, f"step={args.position_step:.4f}m rot={args.rotation_step_deg:.2f}deg grip_step={args.gripper_step:.3f}rad hz={args.hz:.1f}")
    stdscr.addstr(12, 0, f"last_key={last_key_name or '-'}")
    if record_path is not None:
        stdscr.addstr(13, 0, f"recording: {record_path}")
    stdscr.refresh()


def run_live(args: argparse.Namespace) -> None:
    receiver = LatestStateReceiver(args.state_bind_host, args.state_port)
    receiver.start()
    action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    action_addr = (args.action_host, int(args.action_port))
    recorder = JsonlRecorder(
        args.record_path,
        {
            "tool": "keyboard_z1_ee_teleop",
            "mode": "live",
            "created_wall_time": time.time(),
            "action_addr": f"{args.action_host}:{args.action_port}",
            "state_bind": f"{args.state_bind_host}:{args.state_port}",
            "hz": args.hz,
            "position_step": args.position_step,
            "rotation_step_deg": args.rotation_step_deg,
            "gripper_step": args.gripper_step,
            "dry_run": args.dry_run,
        },
    )
    try:
        pos, quat, gripper, initial_record = wait_for_initial_state(receiver, args.state_timeout_s)
        target = TargetState(pos=pos.copy(), quat=quat.copy(), gripper=gripper)
        target.clamp(args)
        recorder.write(
            {
                "kind": "initial_state",
                "t": 0.0,
                "wall_time": time.time(),
                "target": {
                    "ee_pos": target.pos.tolist(),
                    "ee_quat_xyzw": target.quat.tolist(),
                    "gripper": target.gripper,
                },
                "feedback": initial_record,
            }
        )

        def curses_main(stdscr: Any) -> None:
            curses.curs_set(0)
            stdscr.nodelay(True)
            stdscr.keypad(True)
            publish_enabled = not bool(args.start_paused)
            sent = 0
            last_key_name = ""
            t0 = time.monotonic()
            next_send = t0
            next_draw = t0
            period = 1.0 / max(float(args.hz), 1.0e-6)
            draw_period = 1.0 / max(float(args.display_hz), 1.0e-6)
            while True:
                now = time.monotonic()
                while True:
                    key = stdscr.getch()
                    if key < 0:
                        break
                    key_name = apply_key(target, key, args)
                    if key_name is None:
                        continue
                    if key_name == "quit":
                        recorder.write({"kind": "event", "event": "quit", "t": now - t0, "wall_time": time.time()})
                        return
                    if key_name == "toggle_publish":
                        publish_enabled = not publish_enabled
                    elif key_name == "hold_feedback":
                        record, _, _ = receiver.snapshot()
                        if record is not None:
                            try:
                                p, q, g = extract_ee_from_state(record)
                                target.pos = p.copy()
                                target.quat = q.copy()
                                target.gripper = float(g)
                            except Exception:
                                pass
                    elif key_name == "mark":
                        recorder.write({"kind": "event", "event": "mark", "t": now - t0, "wall_time": time.time()})
                    last_key_name = key_name
                    recorder.write(
                        {
                            "kind": "key",
                            "key": key_name,
                            "t": now - t0,
                            "wall_time": time.time(),
                            "target": {
                                "ee_pos": target.pos.tolist(),
                                "ee_quat_xyzw": target.quat.tolist(),
                                "gripper": target.gripper,
                            },
                        }
                    )

                if now >= next_send:
                    record, stamp, count = receiver.snapshot()
                    action = make_action(target.pos, target.quat, target.gripper)
                    if publish_enabled:
                        send_action(action_sock, action_addr, action, args.dry_run)
                        sent += 1
                    recorder.write(
                        {
                            "kind": "sample",
                            "t": now - t0,
                            "wall_time": time.time(),
                            "sent": bool(publish_enabled and not args.dry_run),
                            "sent_count": sent,
                            "action": action,
                            "target": {
                                "ee_pos": target.pos.tolist(),
                                "ee_quat_xyzw": target.quat.tolist(),
                                "gripper": target.gripper,
                            },
                            "feedback_age_s": None if stamp <= 0.0 else time.monotonic() - stamp,
                            "feedback_count": count,
                            "feedback": record,
                        }
                    )
                    next_send += period
                    if next_send < now - period:
                        next_send = now + period

                if now >= next_draw:
                    record, _, _ = receiver.snapshot()
                    draw_live_screen(
                        stdscr,
                        args,
                        target,
                        record,
                        publish_enabled,
                        sent,
                        now - t0,
                        last_key_name,
                        args.record_path,
                    )
                    next_draw += draw_period
                    if next_draw < now - draw_period:
                        next_draw = now + draw_period
                time.sleep(0.002)

        curses.wrapper(curses_main)
    finally:
        recorder.close()
        action_sock.close()
        receiver.close()


def load_replay_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") != "sample":
            continue
        action = record.get("action")
        if isinstance(action, list) and len(action) == ACT_DIM:
            samples.append(record)
    if not samples:
        raise ValueError(f"no replay samples found in {path}")
    samples.sort(key=lambda x: float(x.get("t", 0.0)))
    return samples


def run_replay(args: argparse.Namespace) -> None:
    if args.replay_path is None:
        raise ValueError("--replay_path is required for replay mode")
    samples = load_replay_samples(args.replay_path)
    receiver = LatestStateReceiver(args.state_bind_host, args.state_port)
    receiver.start()
    action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    action_addr = (args.action_host, int(args.action_port))
    recorder = JsonlRecorder(
        args.record_path,
        {
            "tool": "keyboard_z1_ee_teleop",
            "mode": "replay",
            "source": str(args.replay_path),
            "created_wall_time": time.time(),
            "speed": args.replay_speed,
            "dry_run": args.dry_run,
        },
    )
    try:
        wait_for_initial_state(receiver, args.state_timeout_s)
        source_t0 = float(samples[0].get("t", 0.0))
        wall_t0 = time.monotonic()
        sent = 0
        for i, sample in enumerate(samples):
            target_time = (float(sample.get("t", 0.0)) - source_t0) / max(float(args.replay_speed), 1.0e-6)
            while time.monotonic() - wall_t0 < target_time:
                time.sleep(0.001)
            action = [float(x) for x in sample["action"]]
            send_action(action_sock, action_addr, action, args.dry_run)
            sent += 1
            state, stamp, count = receiver.snapshot()
            row = {
                "kind": "replay_sample",
                "index": i,
                "t": time.monotonic() - wall_t0,
                "wall_time": time.time(),
                "sent": not args.dry_run,
                "sent_count": sent,
                "action": action,
                "feedback_age_s": None if stamp <= 0.0 else time.monotonic() - stamp,
                "feedback_count": count,
                "feedback": state,
            }
            recorder.write(row)
            if i % max(1, int(args.replay_print_every)) == 0:
                ee = action[EE_POS_SLICE]
                print(
                    f"replay {i+1}/{len(samples)} t={row['t']:.2f}s "
                    f"ee=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f}) grip={action[GRIPPER_INDEX]:+.3f} "
                    f"ik={(state or {}).get('ik_ok')} err={(state or {}).get('last_error')}",
                    flush=True,
                )

        if args.replay_hold_final_s > 0.0:
            final_action = [float(x) for x in samples[-1]["action"]]
            period = 1.0 / max(float(args.hz), 1.0e-6)
            end_t = time.monotonic() + float(args.replay_hold_final_s)
            while time.monotonic() < end_t:
                send_action(action_sock, action_addr, final_action, args.dry_run)
                sent += 1
                time.sleep(period)
        print(f"replay_done samples={len(samples)} sent={sent} dry_run={args.dry_run}", flush=True)
    finally:
        recorder.close()
        action_sock.close()
        receiver.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard Z1 EE teleop through z1_act_ee_bridge UDP.")
    parser.add_argument("--action_host", default="127.0.0.1", help="z1_act_ee_bridge action UDP host.")
    parser.add_argument("--action_port", type=int, default=15011, help="z1_act_ee_bridge action UDP port.")
    parser.add_argument("--state_bind_host", default="127.0.0.1", help="Local host to bind for bridge state UDP.")
    parser.add_argument("--state_port", type=int, default=15013, help="Bridge state UDP port.")
    parser.add_argument("--state_timeout_s", type=float, default=5.0)
    parser.add_argument("--hz", type=float, default=25.0, help="Command send frequency.")
    parser.add_argument("--display_hz", type=float, default=10.0)
    parser.add_argument("--position_step", type=float, default=0.005, help="Meters per key press.")
    parser.add_argument("--rotation_step_deg", type=float, default=2.0, help="Degrees per key press.")
    parser.add_argument("--gripper_step", type=float, default=0.05, help="Radians per key press.")
    parser.add_argument("--gripper_min", type=float, default=-math.pi / 2.0, help="Fully open Z1 gripper command.")
    parser.add_argument("--gripper_max", type=float, default=0.0, help="Closed Z1 gripper command.")
    parser.add_argument("--min_xyz", type=float, nargs=3, default=None, help="Optional ACT-frame lower XYZ clamp.")
    parser.add_argument("--max_xyz", type=float, nargs=3, default=None, help="Optional ACT-frame upper XYZ clamp.")
    parser.add_argument("--record_path", type=Path, default=None, help="JSONL file to write live or replay logs.")
    parser.add_argument("--replay_path", type=Path, default=None, help="Replay a JSONL generated by this tool.")
    parser.add_argument("--replay_speed", type=float, default=1.0)
    parser.add_argument("--replay_hold_final_s", type=float, default=1.0)
    parser.add_argument("--replay_print_every", type=int, default=25)
    parser.add_argument("--dry_run", action="store_true", help="Do not send UDP commands.")
    parser.add_argument("--start_paused", action="store_true", help="Live mode starts without publishing commands.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replay_path is not None:
        run_replay(args)
    else:
        if not sys.stdin.isatty():
            raise RuntimeError("live keyboard mode requires a TTY; use --replay_path for non-interactive replay")
        run_live(args)


if __name__ == "__main__":
    main()
