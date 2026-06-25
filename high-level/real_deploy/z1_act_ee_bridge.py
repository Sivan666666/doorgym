#!/usr/bin/env python3
"""Bridge Door-ACT 10D EE actions to Unitree Z1 low-level commands.

This script is intentionally small and explicit.  It runs two worker threads:

1. arm thread
   - consumes the latest ACT action
   - converts [ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw] to a Z1 arm-base
     end-effector transform
   - solves Z1 IK and sends LOWCMD q/qd/tau + gripper commands

2. IO thread
   - receives ACT action packets over UDP
   - receives the robot dog velocity state sent by the i7 over UDP
   - optionally publishes the assembled ACT observation.state:
       [vx, yaw_rate, ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper]

The ACT action/state convention follows the checkpoint metadata:

    [vx, yaw_rate, ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper]

EE pose is in the ACT/base frame.  By default the ACT/A2W base frame is offset
from the Z1 SDK arm base by the mount translation used in the simulation URDF:
ACT_from_arm.xyz = [0.174, 0.0, 0.142].  Therefore arm_from_act.xyz defaults to
[-0.174, 0.0, -0.142], so incoming ACT EE targets have the A2W mount offset
subtracted before Z1 IK.  The Unitree SDK FK/IK frame also stops short of the
simulated ee_gripper_link, so a default SDK_EE -> ACT_EE tool offset of
[0.086, 0.0, 0.0] is applied: state FK is reported at the simulated EE point,
and ACT targets are converted back to the SDK EE frame before inverse IK.

By default this bridge follows the real deployment path: it instantiates
ArmInterface, switches to LOWCMD, and sends commands.  Pass --no_enable_arm for
a dry-run state/action plumbing test.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import select
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ACT_DIM = 10
EE_POS_SLICE = slice(2, 5)
EE_QUAT_SLICE = slice(5, 9)
GRIPPER_INDEX = 9
A2W_Z1_MOUNT_XYZ_ACT_FROM_ARM = np.asarray([0.174, 0.0, 0.142], dtype=np.float64)
Z1_SDK_EE_TO_ACT_EE_XYZ = np.asarray([0.086, 0.0, 0.0], dtype=np.float64)


def _as_float_array(values: Any, length: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if length is not None and arr.shape[0] < length:
        raise ValueError(f"expected at least {length} floats, got {arr.shape[0]}")
    return arr


def normalize_quat_xyzw(quat: Any) -> np.ndarray:
    q = _as_float_array(quat, 4)[:4].astype(np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1.0e-9:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / norm
    # Keep quaternion signs stable for logs / policy state.
    if q[3] < 0.0:
        q = -q
    return q


def quat_xyzw_to_rot(quat: Any) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rot_to_quat_xyzw(rot: Any) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        diag = np.diag(r)
        if diag[0] > diag[1] and diag[0] > diag[2]:
            s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1.0e-12)) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif diag[1] > diag[2]:
            s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1.0e-12)) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1.0e-12)) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    return normalize_quat_xyzw([x, y, z, w])


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def make_transform(xyz: Any, rpy: Any) -> np.ndarray:
    t = _as_float_array(xyz, 3)[:3].astype(np.float64)
    rr = _as_float_array(rpy, 3)[:3].astype(np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rpy_to_rot(float(rr[0]), float(rr[1]), float(rr[2]))
    out[:3, 3] = t
    return out


def pose_to_transform(pos: Any, quat_xyzw: Any) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_xyzw_to_rot(quat_xyzw)
    out[:3, 3] = _as_float_array(pos, 3)[:3].astype(np.float64)
    return out


def transform_to_pose(transform: Any) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    pos = t[:3, 3].astype(np.float32)
    quat = rot_to_quat_xyzw(t[:3, :3]).astype(np.float32)
    return pos, quat


def quat_angle_distance_rad(q0: Any, q1: Any) -> float:
    qa = normalize_quat_xyzw(q0)
    qb = normalize_quat_xyzw(q1)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(2.0 * math.acos(dot))


def rotation_error_vector_world(target_rot: Any, current_rot: Any) -> np.ndarray:
    """Shortest world-frame rotation vector from current to target."""

    target = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    current = np.asarray(current_rot, dtype=np.float64).reshape(3, 3)
    q_error = rot_to_quat_xyzw(target @ current.T)
    vector = q_error[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, max(float(q_error[3]), 0.0))
    return vector * (angle / vector_norm)


def add_z1_sdk_lib_path(path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.exists():
        return
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    old = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in old.split(":") if p]
    if text not in parts:
        os.environ["LD_LIBRARY_PATH"] = text + (":" + old if old else "")
    for lib_name in ("libZ1_SDK_aarch64.so", "libZ1_SDK_x86_64.so"):
        lib_path = path / lib_name
        if lib_path.is_file():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            break


def _try_parse_json_packet(data: bytes) -> Any | None:
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _try_parse_float32_packet(data: bytes) -> np.ndarray | None:
    if len(data) < 4 or len(data) % 4 != 0:
        return None
    count = len(data) // 4
    if count > 64:
        return None
    try:
        return np.asarray(struct.unpack("<" + "f" * count, data), dtype=np.float32)
    except struct.error:
        return None


def parse_action_packet(data: bytes) -> np.ndarray:
    payload = _try_parse_json_packet(data)
    if payload is None:
        arr = _try_parse_float32_packet(data)
        if arr is None:
            raise ValueError("not JSON and not little-endian float32 data")
        return _as_float_array(arr, ACT_DIM)[:ACT_DIM].astype(np.float32)

    if isinstance(payload, list):
        return _as_float_array(payload, ACT_DIM)[:ACT_DIM].astype(np.float32)

    if not isinstance(payload, dict):
        raise ValueError("action JSON must be a list or object")

    for key in ("action", "act_action", "door_action", "target"):
        if key in payload:
            return _as_float_array(payload[key], ACT_DIM)[:ACT_DIM].astype(np.float32)

    ee_obj = payload.get("ee", {}) if isinstance(payload.get("ee"), dict) else {}
    base_obj = payload.get("base", {}) if isinstance(payload.get("base"), dict) else {}

    vx = payload.get("vx", payload.get("base_vx", base_obj.get("vx", 0.0)))
    yaw_rate = payload.get(
        "yaw_rate",
        payload.get("yaw", payload.get("wz", payload.get("vyaw", base_obj.get("yaw_rate", base_obj.get("wz", 0.0))))),
    )
    ee_pos = payload.get("ee_pos", payload.get("position", ee_obj.get("pos", ee_obj.get("position"))))
    ee_quat = payload.get(
        "ee_quat_xyzw",
        payload.get("ee_quat", payload.get("quat_xyzw", payload.get("quat", ee_obj.get("quat_xyzw", ee_obj.get("quat"))))),
    )
    gripper = payload.get("gripper", payload.get("gripper_q", ee_obj.get("gripper", 0.0)))

    if ee_pos is None or ee_quat is None:
        raise ValueError("action object needs either action[10] or ee_pos + ee_quat_xyzw")

    action = np.zeros(ACT_DIM, dtype=np.float32)
    action[0] = float(vx)
    action[1] = float(yaw_rate)
    action[EE_POS_SLICE] = _as_float_array(ee_pos, 3)[:3]
    action[EE_QUAT_SLICE] = normalize_quat_xyzw(ee_quat).astype(np.float32)
    action[GRIPPER_INDEX] = float(gripper)
    return action


def parse_action_command(data: bytes) -> tuple[np.ndarray, str, np.ndarray | None]:
    """Parse an EE action plus an optional explicit joint-space target."""

    action = parse_action_packet(data)
    payload = _try_parse_json_packet(data)
    if not isinstance(payload, dict):
        return action, "ee", None

    mode = str(payload.get("control_mode", payload.get("mode", "ee"))).strip().lower()
    joint_value = payload.get("joint_target", payload.get("q_target"))
    if joint_value is None:
        return action, "ee", None
    joint_target = _as_float_array(joint_value, 6)[:6].astype(np.float32)
    if not np.isfinite(joint_target).all():
        raise ValueError("joint_target contains NaN or Inf")
    if mode not in ("joint", "joint_target", "home"):
        mode = "joint"
    return action, mode, joint_target


def parse_vel_state_packet(data: bytes) -> np.ndarray:
    payload = _try_parse_json_packet(data)
    if payload is None:
        arr = _try_parse_float32_packet(data)
        if arr is None:
            raise ValueError("not JSON and not little-endian float32 data")
        return _as_float_array(arr, 2)[:2].astype(np.float32)

    if isinstance(payload, list):
        return _as_float_array(payload, 2)[:2].astype(np.float32)

    if not isinstance(payload, dict):
        raise ValueError("vel_state JSON must be a list or object")

    for key in ("vel_state", "velocity_state", "base_vel_state"):
        if key in payload:
            return _as_float_array(payload[key], 2)[:2].astype(np.float32)

    linear = payload.get("linear", {}) if isinstance(payload.get("linear"), dict) else {}
    angular = payload.get("angular", {}) if isinstance(payload.get("angular"), dict) else {}
    vx = payload.get("vx", payload.get("linear_x", linear.get("x", 0.0)))
    yaw_rate = payload.get(
        "yaw_rate",
        payload.get("yaw", payload.get("wz", payload.get("omega_z", payload.get("vyaw", angular.get("z", 0.0))))),
    )
    return np.asarray([float(vx), float(yaw_rate)], dtype=np.float32)


def map_act_gripper_to_z1(value: float, args: argparse.Namespace) -> float:
    mapped = float(value) * float(args.gripper_scale) + float(args.gripper_offset)
    return float(np.clip(mapped, float(args.gripper_min), float(args.gripper_max)))


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
    """Snap sustained near-closed gripper ACT targets to a hard closed target.

    The latch has intentionally no release hysteresis: one command below the
    threshold immediately exits forced-closed mode and follows the ACT target.
    """

    goal = float(np.clip(float(mapped_goal), float(gripper_min), float(gripper_max)))
    if not bool(enabled):
        return goal, 0, False

    threshold = float(threshold)
    confirm_steps = max(1, int(confirm_steps))
    if goal >= threshold:
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


def make_act_state(vel_state: np.ndarray, ee_pos_act: np.ndarray, ee_quat_act: np.ndarray, gripper: float) -> np.ndarray:
    out = np.zeros(ACT_DIM, dtype=np.float32)
    out[0:2] = np.asarray(vel_state, dtype=np.float32).reshape(-1)[:2]
    out[EE_POS_SLICE] = np.asarray(ee_pos_act, dtype=np.float32).reshape(3)
    out[EE_QUAT_SLICE] = normalize_quat_xyzw(ee_quat_act).astype(np.float32)
    out[GRIPPER_INDEX] = float(gripper)
    return out


@dataclass
class PoseQCacheEntry:
    pos: np.ndarray
    quat: np.ndarray
    q: np.ndarray
    stamp: float
    count: int = 1


class PoseQCache:
    """Small nearest-neighbor cache from observed ACT-frame EE poses to q.

    Unitree's analytic IK can be brittle near folded/start postures.  The cache
    gives the controller a safe way to return to poses that were actually
    observed through FK, without pretending to solve a new IK problem.
    """

    def __init__(self, max_size: int, min_pos_delta: float, min_rot_delta_rad: float) -> None:
        self.max_size = max(1, int(max_size))
        self.min_pos_delta = max(0.0, float(min_pos_delta))
        self.min_rot_delta_rad = max(0.0, float(min_rot_delta_rad))
        self.entries: list[PoseQCacheEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add_transform(self, transform_act: Any, q: Any) -> None:
        try:
            pos, quat = transform_to_pose(transform_act)
            q_arr = np.asarray(q, dtype=np.float64).reshape(6)
        except Exception:
            return
        if not (np.isfinite(pos).all() and np.isfinite(quat).all() and np.isfinite(q_arr).all()):
            return

        now = time.time()
        # Dedupe nearby FK samples so the cache stays useful instead of becoming
        # a 500Hz log of almost-identical poses.
        for entry in reversed(self.entries):
            pos_err = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - entry.pos))
            rot_err = quat_angle_distance_rad(quat, entry.quat)
            if pos_err <= self.min_pos_delta and rot_err <= self.min_rot_delta_rad:
                # Keep the cached pose as a fixed anchor.  If we move the pose
                # to every latest FK sample, a continuous trajectory collapses
                # into one sliding cache entry and cannot be used for returning.
                entry.q = q_arr.copy()
                entry.stamp = now
                entry.count += 1
                return

        self.entries.append(
            PoseQCacheEntry(
                pos=np.asarray(pos, dtype=np.float64).reshape(3),
                quat=normalize_quat_xyzw(quat),
                q=q_arr.copy(),
                stamp=now,
            )
        )
        if len(self.entries) > self.max_size:
            del self.entries[: len(self.entries) - self.max_size]

    def lookup(
        self,
        target_transform_act: Any,
        q_seed: np.ndarray,
        pos_tolerance: float,
        rot_tolerance_rad: float,
        max_joint_delta: float,
    ) -> tuple[np.ndarray | None, dict[str, float]]:
        if not self.entries:
            return None, {"cache_size": 0.0}
        try:
            target_pos, target_quat = transform_to_pose(target_transform_act)
        except Exception:
            return None, {"cache_size": float(len(self.entries))}

        q_seed = np.asarray(q_seed, dtype=np.float64).reshape(6)
        pos_tol = max(float(pos_tolerance), 1.0e-9)
        rot_tol = max(float(rot_tolerance_rad), 1.0e-9)
        max_joint_delta = max(float(max_joint_delta), 1.0e-9)
        best_entry: PoseQCacheEntry | None = None
        best_score = float("inf")
        best_stats: dict[str, float] = {"cache_size": float(len(self.entries))}

        for entry in self.entries:
            pos_err = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float64) - entry.pos))
            rot_err = quat_angle_distance_rad(target_quat, entry.quat)
            joint_delta = float(np.max(np.abs(entry.q - q_seed)))
            if pos_err > pos_tol or rot_err > rot_tol or joint_delta > max_joint_delta:
                continue
            score = pos_err / pos_tol + rot_err / rot_tol + 0.05 * joint_delta / max_joint_delta
            if score < best_score:
                best_score = score
                best_entry = entry
                best_stats = {
                    "cache_size": float(len(self.entries)),
                    "pos_err": pos_err,
                    "rot_err_deg": math.degrees(rot_err),
                    "joint_delta": joint_delta,
                    "score": score,
                }

        if best_entry is None:
            return None, best_stats
        return best_entry.q.copy(), best_stats


@dataclass
class SharedBridgeState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    action: np.ndarray = field(default_factory=lambda: np.zeros(ACT_DIM, dtype=np.float32))
    action_stamp: float = 0.0
    action_source: str = ""
    has_action: bool = False
    action_mode: str = "ee"
    joint_target: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    has_joint_target: bool = False
    vel_state: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    vel_state_stamp: float = 0.0
    vel_state_source: str = ""
    has_vel_state: bool = False
    ee_pos_act: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    ee_quat_act: np.ndarray = field(default_factory=lambda: np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    gripper: float = 0.0
    q: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    qd: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    ik_ok: bool = False
    ik_source: str = ""
    ik_fail_count: int = 0
    control_count: int = 0
    arm_enabled: bool = False
    dry_run: bool = True
    startup_zero_requested: bool = False
    startup_zero_active: bool = False
    startup_zero_done: bool = False
    startup_zero_max_err: float = float("nan")
    startup_home_q: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    startup_home_max_speed: float = float("nan")
    startup_zero_error: str = ""
    shutdown_requested: bool = False
    shutdown_active: bool = False
    shutdown_done: bool = False
    shutdown_error: str = ""
    last_error: str = ""
    gripper_action_raw: float = float("nan")
    gripper_goal_before_latch: float = float("nan")
    gripper_goal_after_latch: float = float("nan")
    gripper_close_latch_count: int = 0
    gripper_force_closed: bool = False

    def snapshot(self, args: argparse.Namespace) -> dict[str, Any]:
        now = time.time()
        with self.lock:
            vel = self.vel_state.copy()
            if (not self.has_vel_state) or (now - self.vel_state_stamp > args.vel_timeout_s):
                vel[:] = 0.0
            act_state = make_act_state(vel, self.ee_pos_act, self.ee_quat_act, self.gripper)
            return {
                "type": "act_state",
                "stamp": now,
                "state": np.round(act_state, 6).tolist(),
                "vel_state": np.round(vel, 6).tolist(),
                "vel_state_age_s": None if not self.has_vel_state else round(now - self.vel_state_stamp, 4),
                "action_age_s": None if not self.has_action else round(now - self.action_stamp, 4),
                "ee_pos": np.round(self.ee_pos_act, 6).tolist(),
                "ee_quat_xyzw": np.round(self.ee_quat_act, 6).tolist(),
                "gripper": round(float(self.gripper), 6),
                "q": np.round(self.q, 6).tolist(),
                "qd": np.round(self.qd, 6).tolist(),
                "ik_ok": bool(self.ik_ok),
                "ik_source": self.ik_source,
                "ik_fail_count": int(self.ik_fail_count),
                "control_count": int(self.control_count),
                "arm_enabled": bool(self.arm_enabled),
                "dry_run": bool(self.dry_run),
                "startup_zero_requested": bool(self.startup_zero_requested),
                "startup_zero_active": bool(self.startup_zero_active),
                "startup_zero_done": bool(self.startup_zero_done),
                "startup_zero_max_err": (
                    None
                    if not np.isfinite(float(self.startup_zero_max_err))
                    else round(float(self.startup_zero_max_err), 6)
                ),
                "startup_home_q": np.round(self.startup_home_q, 6).tolist(),
                "startup_home_max_speed": (
                    None
                    if not np.isfinite(float(self.startup_home_max_speed))
                    else round(float(self.startup_home_max_speed), 6)
                ),
                "startup_zero_error": self.startup_zero_error,
                "shutdown_requested": bool(self.shutdown_requested),
                "shutdown_active": bool(self.shutdown_active),
                "shutdown_done": bool(self.shutdown_done),
                "shutdown_error": self.shutdown_error,
                "last_error": self.last_error,
                "gripper_action_raw": (
                    None if not np.isfinite(float(self.gripper_action_raw)) else round(float(self.gripper_action_raw), 6)
                ),
                "gripper_goal_before_latch": (
                    None
                    if not np.isfinite(float(self.gripper_goal_before_latch))
                    else round(float(self.gripper_goal_before_latch), 6)
                ),
                "gripper_goal_after_latch": (
                    None
                    if not np.isfinite(float(self.gripper_goal_after_latch))
                    else round(float(self.gripper_goal_after_latch), 6)
                ),
                "gripper_close_latch_count": int(self.gripper_close_latch_count),
                "gripper_force_closed": bool(self.gripper_force_closed),
            }


def update_shared_arm_state(
    shared: SharedBridgeState,
    q: np.ndarray,
    qd: np.ndarray,
    gripper: float,
    ee_pos_act: np.ndarray,
    ee_quat_act: np.ndarray,
    ik_ok: bool,
    ik_fail_count: int,
    control_count: int,
    last_error: str = "",
    ik_source: str = "",
) -> None:
    with shared.lock:
        shared.q = np.asarray(q, dtype=np.float32).reshape(6)
        shared.qd = np.asarray(qd, dtype=np.float32).reshape(6)
        shared.gripper = float(gripper)
        shared.ee_pos_act = np.asarray(ee_pos_act, dtype=np.float32).reshape(3)
        shared.ee_quat_act = normalize_quat_xyzw(ee_quat_act).astype(np.float32)
        shared.ik_ok = bool(ik_ok)
        shared.ik_source = str(ik_source)
        shared.ik_fail_count = int(ik_fail_count)
        shared.control_count = int(control_count)
        shared.last_error = str(last_error)


def update_shared_arm_feedback(
    shared: SharedBridgeState,
    q: np.ndarray,
    qd: np.ndarray,
    gripper: float,
    ee_pos_act: np.ndarray,
    ee_quat_act: np.ndarray,
) -> None:
    with shared.lock:
        shared.q = np.asarray(q, dtype=np.float32).reshape(6)
        shared.qd = np.asarray(qd, dtype=np.float32).reshape(6)
        shared.gripper = float(gripper)
        shared.ee_pos_act = np.asarray(ee_pos_act, dtype=np.float32).reshape(3)
        shared.ee_quat_act = normalize_quat_xyzw(ee_quat_act).astype(np.float32)


def update_shared_control_status(
    shared: SharedBridgeState,
    ik_ok: bool,
    ik_fail_count: int,
    control_count: int,
    last_error: str = "",
    ik_source: str = "",
) -> None:
    with shared.lock:
        shared.ik_ok = bool(ik_ok)
        shared.ik_source = str(ik_source)
        shared.ik_fail_count = int(ik_fail_count)
        shared.control_count = int(control_count)
        shared.last_error = str(last_error)


def update_shared_startup_zero_status(
    shared: SharedBridgeState,
    requested: bool,
    active: bool,
    done: bool,
    max_err: float = float("nan"),
    home_q: np.ndarray | None = None,
    max_speed: float = float("nan"),
    error: str = "",
) -> None:
    with shared.lock:
        shared.startup_zero_requested = bool(requested)
        shared.startup_zero_active = bool(active)
        shared.startup_zero_done = bool(done)
        shared.startup_zero_max_err = float(max_err)
        if home_q is not None:
            shared.startup_home_q = np.asarray(home_q, dtype=np.float32).reshape(6)
        shared.startup_home_max_speed = float(max_speed)
        shared.startup_zero_error = str(error)


def update_shared_shutdown_status(
    shared: SharedBridgeState,
    *,
    requested: bool,
    active: bool,
    done: bool,
    error: str = "",
) -> None:
    with shared.lock:
        shared.shutdown_requested = bool(requested)
        shared.shutdown_active = bool(active)
        shared.shutdown_done = bool(done)
        shared.shutdown_error = str(error)


def update_shared_gripper_latch_status(
    shared: SharedBridgeState,
    *,
    action_raw: float = float("nan"),
    goal_before_latch: float = float("nan"),
    goal_after_latch: float = float("nan"),
    latch_count: int = 0,
    force_closed: bool = False,
) -> None:
    with shared.lock:
        shared.gripper_action_raw = float(action_raw)
        shared.gripper_goal_before_latch = float(goal_before_latch)
        shared.gripper_goal_after_latch = float(goal_after_latch)
        shared.gripper_close_latch_count = int(latch_count)
        shared.gripper_force_closed = bool(force_closed)


def get_action_snapshot(
    shared: SharedBridgeState,
) -> tuple[np.ndarray, bool, float, float, str, np.ndarray | None]:
    now = time.time()
    with shared.lock:
        age = float("inf") if not shared.has_action else now - shared.action_stamp
        joint_target = shared.joint_target.copy() if shared.has_joint_target else None
        return (
            shared.action.copy(),
            bool(shared.has_action),
            age,
            float(shared.action_stamp),
            str(shared.action_mode),
            joint_target,
        )


def clamp_joint_target(
    q_current: np.ndarray,
    q_target: np.ndarray,
    dt: float,
    max_speed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta_limit = np.maximum(max_speed * max(dt, 1.0e-6), 1.0e-6)
    delta = np.clip(q_target - q_current, -delta_limit, delta_limit)
    q_next = q_current + delta
    qd = delta / max(dt, 1.0e-6)
    return q_next.astype(np.float64), qd.astype(np.float64)


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
    """Advance a smooth gripper command without differentiating noisy feedback."""

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


@dataclass
class SoftIkResult:
    success: bool
    q: np.ndarray
    position_error: float
    orientation_error: float
    initial_position_error: float
    iterations: int


def _tool_pose_and_jacobian(
    arm_model,
    sdk_lock: threading.Lock,
    q: np.ndarray,
    act_ee_from_sdk_ee: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arm-frame ACT tool transform plus world spatial Jacobians."""

    with sdk_lock:
        sdk_transform = np.asarray(
            arm_model.forwardKinematics(np.asarray(q, dtype=np.float64), 6),
            dtype=np.float64,
        ).reshape(4, 4)
        sdk_jacobian = np.asarray(
            arm_model.CalcJacobian(np.asarray(q, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(6, 6)

    tool_transform = sdk_transform @ np.asarray(act_ee_from_sdk_ee, dtype=np.float64).reshape(4, 4)
    angular_jacobian = sdk_jacobian[:3, :]
    sdk_linear_jacobian = sdk_jacobian[3:, :]
    tool_offset_world = sdk_transform[:3, :3] @ np.asarray(
        act_ee_from_sdk_ee[:3, 3],
        dtype=np.float64,
    )
    tool_linear_jacobian = sdk_linear_jacobian + np.cross(
        angular_jacobian.T,
        tool_offset_world,
    ).T
    return tool_transform, tool_linear_jacobian, angular_jacobian


def solve_soft_pose_ik(
    arm_model,
    sdk_lock: threading.Lock,
    target_tool_transform_arm: np.ndarray,
    q_seed: np.ndarray,
    joint_min: np.ndarray,
    joint_max: np.ndarray,
    act_ee_from_sdk_ee: np.ndarray,
    *,
    max_iterations: int,
    damping: float,
    orientation_weight: float,
    seed_weight: float,
    max_joint_step: float,
    position_tolerance: float,
    min_position_improvement: float,
) -> SoftIkResult:
    """Find the closest local tool pose with strict position-first priority."""

    target = np.asarray(target_tool_transform_arm, dtype=np.float64).reshape(4, 4)
    joint_min = np.asarray(joint_min, dtype=np.float64).reshape(6)
    joint_max = np.asarray(joint_max, dtype=np.float64).reshape(6)
    seed = np.clip(np.asarray(q_seed, dtype=np.float64).reshape(6), joint_min, joint_max)
    q = seed.copy()
    damping = max(float(damping), 1.0e-6)
    orientation_weight = max(float(orientation_weight), 0.0)
    seed_weight = max(float(seed_weight), 0.0)
    max_joint_step = max(float(max_joint_step), 1.0e-5)
    position_tolerance = max(float(position_tolerance), 0.0)
    min_position_improvement = max(float(min_position_improvement), 0.0)
    identity = np.eye(6, dtype=np.float64)

    def evaluate(
        q_value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        transform, position_jacobian, angular_jacobian = _tool_pose_and_jacobian(
            arm_model,
            sdk_lock,
            q_value,
            act_ee_from_sdk_ee,
        )
        position_error_vector = target[:3, 3] - transform[:3, 3]
        orientation_error_vector = rotation_error_vector_world(target[:3, :3], transform[:3, :3])
        return (
            position_error_vector,
            orientation_error_vector,
            position_jacobian,
            angular_jacobian,
            float(np.linalg.norm(position_error_vector)),
            float(np.linalg.norm(orientation_error_vector)),
        )

    (
        position_error_vector,
        orientation_error_vector,
        position_jacobian,
        angular_jacobian,
        initial_position_error,
        initial_orientation_error,
    ) = evaluate(q)
    best_q = q.copy()
    best_position_error = initial_position_error
    best_orientation_error = initial_orientation_error
    iterations_used = 0
    total_iterations = max(1, int(max_iterations))
    position_iterations = max(1, int(math.ceil(total_iterations * 2.0 / 3.0)))

    # Stage 1: position-only DLS. Keep moving through local singular regions,
    # while retaining the best bounded waypoint seen along the path.
    for _ in range(position_iterations):
        iterations_used += 1
        try:
            delta = position_jacobian.T @ np.linalg.solve(
                position_jacobian @ position_jacobian.T + damping * damping * np.eye(3),
                position_error_vector,
            )
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(position_jacobian, position_error_vector, rcond=None)[0]
        if not np.isfinite(delta).all():
            break
        delta = np.clip(delta, -max_joint_step, max_joint_step)
        if float(np.linalg.norm(delta)) < 1.0e-8:
            break
        q = np.clip(q + delta, joint_min, joint_max)
        (
            position_error_vector,
            orientation_error_vector,
            position_jacobian,
            angular_jacobian,
            position_error,
            orientation_error,
        ) = evaluate(q)
        if position_error < best_position_error:
            best_q = q.copy()
            best_position_error = position_error
            best_orientation_error = orientation_error
        if position_error <= position_tolerance:
            break

    # Stage 2: improve orientation only when the position remains at the best
    # reachable point (within a small numerical guard).
    q = best_q.copy()
    (
        position_error_vector,
        orientation_error_vector,
        position_jacobian,
        angular_jacobian,
        position_error,
        orientation_error,
    ) = evaluate(q)
    position_guard = max(position_tolerance, best_position_error + 5.0e-4)
    best_soft_score = (
        position_error * position_error
        + (orientation_weight * orientation_error) ** 2
        + seed_weight * float(np.dot(q - seed, q - seed))
    )
    for _ in range(max(0, total_iterations - iterations_used)):
        iterations_used += 1
        task_jacobian = np.vstack((position_jacobian, orientation_weight * angular_jacobian))
        task_error = np.concatenate(
            (position_error_vector, orientation_weight * orientation_error_vector)
        )
        normal = task_jacobian.T @ task_jacobian + (damping * damping + seed_weight) * identity
        rhs = task_jacobian.T @ task_error + seed_weight * (seed - q)
        try:
            delta = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        if not np.isfinite(delta).all():
            break
        delta = np.clip(delta, -max_joint_step, max_joint_step)
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1):
            proposal = np.clip(q + delta * scale, joint_min, joint_max)
            evaluated = evaluate(proposal)
            proposal_position_error = evaluated[-2]
            proposal_orientation_error = evaluated[-1]
            proposal_score = (
                proposal_position_error * proposal_position_error
                + (orientation_weight * proposal_orientation_error) ** 2
                + seed_weight * float(np.dot(proposal - seed, proposal - seed))
            )
            if (
                proposal_position_error <= position_guard
                and proposal_score + 1.0e-12 < best_soft_score
            ):
                q = proposal
                (
                    position_error_vector,
                    orientation_error_vector,
                    position_jacobian,
                    angular_jacobian,
                    position_error,
                    orientation_error,
                ) = evaluated
                best_q = q.copy()
                best_position_error = position_error
                best_orientation_error = orientation_error
                best_soft_score = proposal_score
                accepted = True
                break
        if not accepted:
            break

    improved = initial_position_error - best_position_error
    success = bool(
        np.isfinite(best_q).all()
        and (
            best_position_error <= position_tolerance
            or improved >= min_position_improvement
        )
    )
    return SoftIkResult(
        success=success,
        q=best_q,
        position_error=best_position_error,
        orientation_error=best_orientation_error,
        initial_position_error=initial_position_error,
        iterations=iterations_used,
    )


@dataclass
class OnlineQuinticJointTrajectory:
    """Causal joint-space waypoint interpolation for the 500 Hz LOWCMD loop.

    Every successful IK result becomes one new waypoint.  Replanning starts
    from the exact q/qd/qdd state of the active segment, so command position,
    velocity, and acceleration remain continuous without requiring future
    waypoints or an offline trajectory.
    """

    q_end: np.ndarray
    qd_end: np.ndarray
    qdd_end: np.ndarray
    start_time: float
    duration: float
    coefficients: np.ndarray

    @classmethod
    def hold(cls, q: np.ndarray, now: float) -> "OnlineQuinticJointTrajectory":
        q = np.asarray(q, dtype=np.float64).reshape(6)
        coefficients = np.zeros((6, 6), dtype=np.float64)
        coefficients[:, 0] = q
        return cls(
            q_end=q.copy(),
            qd_end=np.zeros(6, dtype=np.float64),
            qdd_end=np.zeros(6, dtype=np.float64),
            start_time=float(now),
            duration=0.0,
            coefficients=coefficients,
        )

    @staticmethod
    def _coefficients(
        q0: np.ndarray,
        qd0: np.ndarray,
        qdd0: np.ndarray,
        q1: np.ndarray,
        qd1: np.ndarray,
        qdd1: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        """Return a quintic satisfying arbitrary endpoint q/qd/qdd."""

        q0 = np.asarray(q0, dtype=np.float64).reshape(6)
        qd0 = np.asarray(qd0, dtype=np.float64).reshape(6)
        qdd0 = np.asarray(qdd0, dtype=np.float64).reshape(6)
        q1 = np.asarray(q1, dtype=np.float64).reshape(6)
        qd1 = np.asarray(qd1, dtype=np.float64).reshape(6)
        qdd1 = np.asarray(qdd1, dtype=np.float64).reshape(6)
        t = max(float(duration), 1.0e-6)

        coefficients = np.zeros((6, 6), dtype=np.float64)
        coefficients[:, 0] = q0
        coefficients[:, 1] = qd0
        coefficients[:, 2] = 0.5 * qdd0

        matrix = np.asarray(
            [
                [t**3, t**4, t**5],
                [3.0 * t**2, 4.0 * t**3, 5.0 * t**4],
                [6.0 * t, 12.0 * t**2, 20.0 * t**3],
            ],
            dtype=np.float64,
        )
        rhs = np.stack(
            [
                q1 - (q0 + qd0 * t + 0.5 * qdd0 * t**2),
                qd1 - (qd0 + qdd0 * t),
                qdd1 - qdd0,
            ],
            axis=0,
        )
        coefficients[:, 3:6] = np.linalg.solve(matrix, rhs).T
        return coefficients

    @staticmethod
    def _evaluate_coefficients(
        coefficients: np.ndarray,
        elapsed: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = np.asarray(coefficients, dtype=np.float64).reshape(6, 6)
        t = max(float(elapsed), 0.0)
        q = (
            c[:, 0]
            + c[:, 1] * t
            + c[:, 2] * t**2
            + c[:, 3] * t**3
            + c[:, 4] * t**4
            + c[:, 5] * t**5
        )
        qd = (
            c[:, 1]
            + 2.0 * c[:, 2] * t
            + 3.0 * c[:, 3] * t**2
            + 4.0 * c[:, 4] * t**3
            + 5.0 * c[:, 5] * t**4
        )
        qdd = (
            2.0 * c[:, 2]
            + 6.0 * c[:, 3] * t
            + 12.0 * c[:, 4] * t**2
            + 20.0 * c[:, 5] * t**3
        )
        return q, qd, qdd

    def sample(self, now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.duration <= 0.0:
            return self.q_end.copy(), self.qd_end.copy(), self.qdd_end.copy()
        elapsed = float(now) - self.start_time
        if elapsed >= self.duration:
            # Briefly continue at terminal velocity until the next 25 Hz packet
            # arrives. A missing/failed packet explicitly installs a brake
            # segment, so this extrapolation does not continue indefinitely.
            overrun = elapsed - self.duration
            q = self.q_end + self.qd_end * overrun + 0.5 * self.qdd_end * overrun**2
            qd = self.qd_end + self.qdd_end * overrun
            return q, qd, self.qdd_end.copy()
        return self._evaluate_coefficients(self.coefficients, max(elapsed, 0.0))

    def retarget(
        self,
        q_target: np.ndarray,
        qd_target: np.ndarray,
        now: float,
        nominal_duration: float,
        max_speed: np.ndarray,
        max_acceleration: np.ndarray,
    ) -> float:
        """Start a new online segment and return its constraint-safe duration."""

        q0, qd0, qdd0 = self.sample(now)
        q1 = np.asarray(q_target, dtype=np.float64).reshape(6)
        qd1 = np.asarray(qd_target, dtype=np.float64).reshape(6)
        qdd1 = np.zeros(6, dtype=np.float64)
        speed_limit = np.maximum(np.asarray(max_speed, dtype=np.float64).reshape(6), 1.0e-6)
        qd1 = np.clip(qd1, -speed_limit, speed_limit)
        acceleration_limit = np.maximum(
            np.asarray(max_acceleration, dtype=np.float64).reshape(6),
            1.0e-6,
        )
        duration = max(float(nominal_duration), 1.0e-3)

        # Forty milliseconds is the nominal 25 Hz waypoint interval.  For a
        # far target, extend only this active segment until the sampled quintic
        # respects the per-joint speed and acceleration safety limits.
        for _ in range(20):
            coefficients = self._coefficients(q0, qd0, qdd0, q1, qd1, qdd1, duration)
            sample_t = np.linspace(0.0, duration, 65)
            peak_speed = np.zeros(6, dtype=np.float64)
            peak_acceleration = np.zeros(6, dtype=np.float64)
            for t in sample_t:
                _, qd, qdd = self._evaluate_coefficients(coefficients, float(t))
                peak_speed = np.maximum(peak_speed, np.abs(qd))
                peak_acceleration = np.maximum(peak_acceleration, np.abs(qdd))
            speed_ratio = float(np.max(peak_speed / speed_limit))
            acceleration_ratio = float(np.max(peak_acceleration / acceleration_limit))
            ratio = max(speed_ratio, math.sqrt(max(acceleration_ratio, 0.0)))
            if ratio <= 1.0 + 1.0e-6:
                break
            duration *= max(1.05, ratio * 1.02)

        self.q_end = q1.copy()
        self.qd_end = qd1.copy()
        self.qdd_end = qdd1.copy()
        self.start_time = float(now)
        self.duration = float(duration)
        self.coefficients = coefficients
        return self.duration

    def brake(
        self,
        now: float,
        nominal_duration: float,
        max_speed: np.ndarray,
        max_acceleration: np.ndarray,
    ) -> float:
        """Install one smooth deceleration segment from the current state."""

        q0, qd0, _ = self.sample(now)
        acceleration_limit = np.maximum(
            np.asarray(max_acceleration, dtype=np.float64).reshape(6),
            1.0e-6,
        )
        duration = max(
            float(nominal_duration),
            float(np.max(2.0 * np.abs(qd0) / acceleration_limit)),
        )
        q_stop = q0 + 0.5 * qd0 * duration
        return self.retarget(
            q_stop,
            np.zeros(6, dtype=np.float64),
            now,
            duration,
            max_speed,
            max_acceleration,
        )


def send_lowcmd_joint_target(
    arm,
    arm_model,
    sdk_lock: threading.Lock,
    q_target: np.ndarray,
    qd_target: np.ndarray,
    qdd_target: np.ndarray,
    gripper_target: float | None,
    gripper_qd: float,
    joint_min: np.ndarray,
    joint_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_next = np.asarray(q_target, dtype=np.float64).reshape(6)
    qd_cmd = np.asarray(qd_target, dtype=np.float64).reshape(6)
    qdd_cmd = np.asarray(qdd_target, dtype=np.float64).reshape(6)
    try:
        with sdk_lock:
            q_next, qd_cmd = arm_model.jointProtect(q_next, qd_cmd)
        q_next = np.asarray(q_next, dtype=np.float64).reshape(6)
        qd_cmd = np.asarray(qd_cmd, dtype=np.float64).reshape(6)
    except Exception:
        # Some SDK builds expose jointProtect but do not accept writable Eigen refs cleanly.
        q_next = np.clip(q_next, joint_min, joint_max)

    with sdk_lock:
        tau_cmd = arm_model.inverseDynamics(q_next, qd_cmd, qdd_cmd, np.zeros(6))
        arm.setArmCmd(q_next, qd_cmd, tau_cmd)
        if gripper_target is not None:
            arm.setGripperCmd(float(gripper_target), float(gripper_qd), 0.0)
        arm.sendRecv()
    return q_next, qd_cmd


def startup_home_sample_is_stable(
    q_measured: np.ndarray,
    qd_measured: np.ndarray,
    home_q: np.ndarray,
    position_tolerance: float,
    max_joint_speed: float,
) -> tuple[bool, float, float]:
    q = np.asarray(q_measured, dtype=np.float64).reshape(6)
    qd = np.asarray(qd_measured, dtype=np.float64).reshape(6)
    home = np.asarray(home_q, dtype=np.float64).reshape(6)
    if not (np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(home).all()):
        return False, float("inf"), float("inf")
    max_drift = float(np.max(np.abs(q - home)))
    measured_max_speed = float(np.max(np.abs(qd)))
    stable = (
        max_drift <= max(float(position_tolerance), 0.0)
        and measured_max_speed <= max(float(max_joint_speed), 0.0)
    )
    return stable, max_drift, measured_max_speed


def confirm_z1_startup_home(
    args: argparse.Namespace,
    shared: SharedBridgeState,
    stop_event: threading.Event,
    arm,
    arm_model,
    sdk_lock: threading.Lock,
    joint_min: np.ndarray,
    joint_max: np.ndarray,
    joint_speed_limit: np.ndarray,
    period: float,
    q_cmd: np.ndarray,
    gripper_cmd: float,
    control_count: int,
    home_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    requested = bool(args.zero_joints_on_start)
    if not requested:
        update_shared_startup_zero_status(
            shared,
            requested=False,
            active=False,
            done=True,
            home_q=q_cmd,
            max_err=0.0,
            max_speed=0.0,
        )
        return q_cmd, np.zeros(6, dtype=np.float64), gripper_cmd, control_count

    home_q = np.asarray(home_q, dtype=np.float64).reshape(6)
    if not np.isfinite(home_q).all():
        msg = "startup_home_invalid:non_finite_home_q"
        update_shared_startup_zero_status(
            shared,
            requested=True,
            active=False,
            done=False,
            home_q=home_q,
            error=msg,
        )
        raise RuntimeError(msg)
    home_q = np.clip(home_q, joint_min, joint_max)
    gripper_target = gripper_cmd
    if args.zero_joints_gripper is not None:
        gripper_target = float(np.clip(float(args.zero_joints_gripper), args.gripper_min, args.gripper_max))

    timeout_s = max(float(args.zero_joints_timeout_s), 0.0)
    tolerance = max(float(args.zero_joints_tolerance), 0.0)
    max_joint_speed = max(float(args.startup_home_max_joint_speed), 0.0)
    hold_s = max(float(args.zero_joints_hold_s), 0.0)
    deadline = time.monotonic() + timeout_s if timeout_s > 0.0 else float("inf")
    hold_start: float | None = None
    last_drift = float("inf")
    last_speed = float("inf")
    qd_cmd = np.zeros(6, dtype=np.float64)
    next_t = time.monotonic()
    timed_out = False
    gripper_open_sent = False
    gripper_neutral_sent = False

    print(
        "z1_bridge startup_home begin "
        f"home_q={np.round(home_q, 5).tolist()} drift_tolerance={tolerance:.4f} "
        f"max_joint_speed={max_joint_speed:.4f} "
        f"timeout_s={timeout_s:.2f} hold_s={hold_s:.2f} "
        "command_mode=controller_home",
        flush=True,
    )
    update_shared_startup_zero_status(
        shared,
        requested=True,
        active=True,
        done=False,
        max_err=last_drift,
        home_q=home_q,
        max_speed=last_speed,
    )

    while not stop_event.is_set():
        with sdk_lock:
            q_meas = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
            qd_meas = np.asarray(arm.lowstate.getQd(), dtype=np.float64).reshape(6)
        q_current = q_meas if np.isfinite(q_meas).all() else q_cmd
        stable, last_drift, last_speed = startup_home_sample_is_stable(
            q_meas,
            qd_meas,
            home_q,
            tolerance,
            max_joint_speed,
        )
        if stable:
            if hold_start is None:
                hold_start = time.monotonic()
            if time.monotonic() - hold_start >= hold_s:
                break
        else:
            hold_start = None

        if time.monotonic() >= deadline:
            msg = (
                "startup_home_timeout:"
                f"max_drift={last_drift:.4f},max_speed={last_speed:.4f}"
            )
            update_shared_startup_zero_status(
                shared,
                requested=True,
                active=False,
                done=False,
                max_err=last_drift,
                home_q=home_q,
                max_speed=last_speed,
                error=msg,
            )
            if bool(args.zero_joints_strict):
                raise RuntimeError(msg)
            print(f"warning: {msg}; proceeding because --no_zero_joints_strict was set.", file=sys.stderr, flush=True)
            timed_out = True
            break

        # backToStart() performed the only startup return motion. In LOWCMD we
        # merely hold the captured controller home pose while sendRecv keeps
        # feedback fresh; no mathematical all-zero target is ever generated.
        q_next = home_q.copy()
        qd_cmd = np.zeros(6, dtype=np.float64)
        gripper_to_send: float | None = gripper_target
        if bool(args.startup_gripper_open_once):
            if not gripper_open_sent:
                gripper_to_send = gripper_target
                gripper_open_sent = True
            elif not gripper_neutral_sent:
                with sdk_lock:
                    gripper_now = float(arm.lowstate.getGripperQ())
                gripper_target = float(np.clip(gripper_now, args.gripper_min, args.gripper_max))
                gripper_to_send = gripper_target
                gripper_neutral_sent = True
            else:
                gripper_to_send = None
        q_next, qd_cmd = send_lowcmd_joint_target(
            arm,
            arm_model,
            sdk_lock,
            q_next,
            qd_cmd,
            np.zeros(6, dtype=np.float64),
            gripper_to_send,
            0.0,
            joint_min,
            joint_max,
        )
        q_cmd = q_next
        update_shared_control_status(
            shared,
            ik_ok=True,
            ik_fail_count=0,
            control_count=control_count,
            last_error="startup_home_confirm",
            ik_source="startup_home",
        )
        update_shared_startup_zero_status(
            shared,
            requested=True,
            active=True,
            done=False,
            max_err=last_drift,
            home_q=home_q,
            max_speed=last_speed,
        )
        control_count += 1

        next_t += period
        sleep_s = next_t - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_t = time.monotonic()

    if stop_event.is_set() or timed_out:
        return q_cmd, np.zeros(6, dtype=np.float64), gripper_target, control_count

    with sdk_lock:
        q_meas = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
        qd_meas = np.asarray(arm.lowstate.getQd(), dtype=np.float64).reshape(6)
    if np.isfinite(q_meas).all():
        q_cmd = q_meas
    _, last_drift, last_speed = startup_home_sample_is_stable(
        q_meas,
        qd_meas,
        home_q,
        tolerance,
        max_joint_speed,
    )
    qd_cmd = np.zeros(6, dtype=np.float64)
    update_shared_startup_zero_status(
        shared,
        requested=True,
        active=False,
        done=True,
        max_err=last_drift,
        home_q=home_q,
        max_speed=last_speed,
    )
    print(
        "z1_bridge startup_home done "
        f"home_q={np.round(home_q, 5).tolist()} "
        f"max_drift={last_drift:.4f} max_speed={last_speed:.4f}",
        flush=True,
    )
    return q_cmd.copy(), qd_cmd, gripper_target, control_count


def arm_state_read_loop(
    args: argparse.Namespace,
    shared: SharedBridgeState,
    stop_event: threading.Event,
    arm_stop_event: threading.Event,
    arm,
    arm_model,
    sdk_lock: threading.Lock,
    act_from_arm: np.ndarray,
    act_ee_from_sdk_ee: np.ndarray,
) -> None:
    period = 1.0 / max(float(args.arm_state_hz), 1.0)
    next_t = time.monotonic()
    while not stop_event.is_set() and not arm_stop_event.is_set():
        try:
            with sdk_lock:
                q_actual = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
                qd_actual = np.asarray(arm.lowstate.getQd(), dtype=np.float64).reshape(6)
                gripper_actual = float(arm.lowstate.getGripperQ())
                ee_transform_sdk_arm = np.asarray(arm_model.forwardKinematics(q_actual, 6), dtype=np.float64).reshape(4, 4)
                ee_transform_arm = ee_transform_sdk_arm @ act_ee_from_sdk_ee
            ee_transform_act = act_from_arm @ ee_transform_arm
            ee_pos_act, ee_quat_act = transform_to_pose(ee_transform_act)
            update_shared_arm_feedback(
                shared,
                q_actual,
                qd_actual,
                gripper_actual,
                ee_pos_act,
                ee_quat_act,
            )
        except Exception as exc:
            with shared.lock:
                shared.last_error = f"state_read_exception:{exc}"

        next_t += period
        sleep_s = next_t - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_t = time.monotonic()


def should_resolve_ik(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    last_target_pos: np.ndarray | None,
    last_target_quat: np.ndarray | None,
    pos_delta: float,
    rot_delta_rad: float,
) -> bool:
    if last_target_pos is None or last_target_quat is None:
        return True
    pos_err = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float64).reshape(3) - last_target_pos))
    rot_err = quat_angle_distance_rad(target_quat, last_target_quat)
    return pos_err > float(pos_delta) or rot_err > float(rot_delta_rad)


def arm_command_loop(
    args: argparse.Namespace,
    shared: SharedBridgeState,
    stop_event: threading.Event,
) -> None:
    arm_from_act = make_transform(args.arm_from_act_xyz, args.arm_from_act_rpy)
    act_from_arm = np.linalg.inv(arm_from_act)
    act_ee_from_sdk_ee = make_transform(args.act_ee_from_sdk_ee_xyz, args.act_ee_from_sdk_ee_rpy)
    sdk_ee_from_act_ee = np.linalg.inv(act_ee_from_sdk_ee)
    period = 1.0 / max(float(args.dry_run_hz), 1.0)
    q_cmd = np.zeros(6, dtype=np.float64)
    qd_cmd = np.zeros(6, dtype=np.float64)
    gripper_cmd = float(np.clip(args.initial_gripper, args.gripper_min, args.gripper_max))
    joint_speed_limit = np.full(6, float(args.max_joint_speed), dtype=np.float64)
    joint_acceleration_limit = np.full(6, float(args.max_joint_acceleration), dtype=np.float64)
    ik_fail_count = 0
    control_count = 0
    gripper_close_latch_count = 0
    gripper_force_closed = False
    last_processed_gripper_action_stamp = 0.0

    if not args.enable_arm:
        with shared.lock:
            shared.arm_enabled = False
            shared.dry_run = True
            shared.startup_zero_requested = False
            shared.startup_zero_active = False
            shared.startup_zero_done = True
        next_t = time.monotonic()
        ee_pos = np.asarray(args.dry_run_initial_ee_xyz, dtype=np.float32)
        ee_quat = normalize_quat_xyzw(args.dry_run_initial_ee_quat).astype(np.float32)
        print("z1_bridge dry_run=1; --no_enable_arm was set, so no LOWCMD is sent.", flush=True)
        while not stop_event.is_set():
            action, has_action, action_age, action_stamp, _, _ = get_action_snapshot(shared)
            if has_action and action_age <= args.command_timeout_s:
                ee_pos = action[EE_POS_SLICE].astype(np.float32)
                ee_quat = normalize_quat_xyzw(action[EE_QUAT_SLICE]).astype(np.float32)
                if action_stamp > last_processed_gripper_action_stamp:
                    last_processed_gripper_action_stamp = action_stamp
                    raw_gripper_action = float(action[GRIPPER_INDEX])
                    mapped_gripper_goal = map_act_gripper_to_z1(raw_gripper_action, args)
                    gripper_cmd, gripper_close_latch_count, gripper_force_closed = apply_gripper_close_latch(
                        mapped_gripper_goal,
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
                    update_shared_gripper_latch_status(
                        shared,
                        action_raw=raw_gripper_action,
                        goal_before_latch=mapped_gripper_goal,
                        goal_after_latch=gripper_cmd,
                        latch_count=gripper_close_latch_count,
                        force_closed=gripper_force_closed,
                    )
            else:
                gripper_close_latch_count = 0
                gripper_force_closed = False
                last_processed_gripper_action_stamp = 0.0
                update_shared_gripper_latch_status(shared)
            update_shared_arm_state(
                shared,
                q_cmd,
                qd_cmd,
                gripper_cmd,
                ee_pos,
                ee_quat,
                ik_ok=bool(has_action),
                ik_fail_count=ik_fail_count,
                control_count=control_count,
            )
            control_count += 1
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()
        return

    add_z1_sdk_lib_path(args.z1_sdk_lib)
    import unitree_arm_interface

    with shared.lock:
        shared.arm_enabled = True
        shared.dry_run = False

    arm = unitree_arm_interface.ArmInterface(hasGripper=bool(args.has_gripper))
    arm_model = arm._ctrlComp.armModel
    sdk_lock = threading.Lock()
    dt = float(arm._ctrlComp.dt)
    period = dt if dt > 0.0 else 0.002
    with sdk_lock:
        joint_min = np.asarray(arm_model.getJointQMin(), dtype=np.float64).reshape(6)
        joint_max = np.asarray(arm_model.getJointQMax(), dtype=np.float64).reshape(6)
        sdk_speed_max = np.asarray(arm_model.getJointSpeedMax(), dtype=np.float64).reshape(6)
    joint_speed_limit = np.minimum(joint_speed_limit, sdk_speed_max)

    print(
        f"z1_bridge enabling LOWCMD dt={period:.6f}s "
        "joint_command_mode=online_quintic "
        f"max_joint_speed={np.round(joint_speed_limit, 3)} "
        f"max_joint_acceleration={np.round(joint_acceleration_limit, 3)} "
        f"joint_trajectory_duration_s={float(args.joint_trajectory_duration_s):.4f} "
        f"arm_from_act_xyz={np.round(np.asarray(args.arm_from_act_xyz, dtype=np.float64), 4).tolist()} "
        f"arm_from_act_rpy={np.round(np.asarray(args.arm_from_act_rpy, dtype=np.float64), 4).tolist()} "
        f"act_ee_from_sdk_ee_xyz={np.round(np.asarray(args.act_ee_from_sdk_ee_xyz, dtype=np.float64), 4).tolist()} "
        f"act_ee_from_sdk_ee_rpy={np.round(np.asarray(args.act_ee_from_sdk_ee_rpy, dtype=np.float64), 4).tolist()}",
        flush=True,
    )
    startup_home_q: np.ndarray | None = None
    if bool(args.zero_joints_on_start):
        print(
            "z1_bridge controller_home begin: calling Z1 backToStart() "
            "before entering LOWCMD",
            flush=True,
        )
        with sdk_lock:
            arm.loopOn()
            try:
                arm.backToStart()
                startup_home_q = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6).copy()
            finally:
                arm.loopOff()
        print(
            "z1_bridge controller_home command completed "
            f"home_q={np.round(startup_home_q, 5).tolist()}",
            flush=True,
        )

    with sdk_lock:
        arm.setFsmLowcmd()
        q_cmd = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
        qd_cmd = np.zeros(6, dtype=np.float64)
        gripper_cmd = float(np.clip(float(arm.lowstate.getGripperQ()), args.gripper_min, args.gripper_max))
    if startup_home_q is None:
        startup_home_q = q_cmd.copy()
    q_cmd, qd_cmd, gripper_cmd, control_count = confirm_z1_startup_home(
        args,
        shared,
        stop_event,
        arm,
        arm_model,
        sdk_lock,
        joint_min,
        joint_max,
        joint_speed_limit,
        period,
        q_cmd,
        gripper_cmd,
        control_count,
        startup_home_q,
    )
    with sdk_lock:
        initial_transform_sdk_arm = np.asarray(arm_model.forwardKinematics(q_cmd, 6), dtype=np.float64).reshape(4, 4)
        initial_transform_arm = initial_transform_sdk_arm @ act_ee_from_sdk_ee
    initial_transform_act = act_from_arm @ initial_transform_arm
    q_goal = q_cmd.copy()
    joint_trajectory = OnlineQuinticJointTrajectory.hold(q_cmd, time.monotonic())
    gripper_goal = gripper_cmd
    gripper_qd_cmd = 0.0
    gripper_idle_neutral_sent = bool(args.startup_gripper_open_once)
    last_target_pos_act: np.ndarray | None = None
    last_target_quat_act: np.ndarray | None = None
    last_ik_source = ""
    last_processed_action_stamp = 0.0
    last_successful_waypoint = q_cmd.copy()
    last_successful_waypoint_stamp = 0.0
    last_waypoint_velocity = np.zeros(6, dtype=np.float64)
    last_brake_action_stamp = -1.0
    pose_q_cache = PoseQCache(
        max_size=args.ik_cache_size,
        min_pos_delta=args.ik_cache_min_pos_delta,
        min_rot_delta_rad=math.radians(args.ik_cache_min_rot_delta_deg),
    )
    pose_q_cache.add_transform(initial_transform_act, q_cmd)
    next_t = time.monotonic()
    arm_stop_event = threading.Event()
    state_thread = threading.Thread(
        target=arm_state_read_loop,
        args=(args, shared, stop_event, arm_stop_event, arm, arm_model, sdk_lock, act_from_arm, act_ee_from_sdk_ee),
        daemon=True,
        name="z1-state",
    )
    state_thread.start()

    try:
        while not stop_event.is_set():
            with shared.lock:
                shutdown_requested = bool(shared.shutdown_requested)
            if shutdown_requested:
                update_shared_control_status(
                    shared,
                    ik_ok=False,
                    ik_fail_count=ik_fail_count,
                    control_count=control_count,
                    last_error="shutdown_back_to_start",
                    ik_source="shutdown",
                )
                break

            loop_now = time.monotonic()
            last_error = ""
            (
                action,
                has_action,
                action_age,
                action_stamp,
                action_mode,
                explicit_joint_target,
            ) = get_action_snapshot(shared)
            q_plan, _, _ = joint_trajectory.sample(loop_now)
            with sdk_lock:
                q_meas = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
                ee_transform_sdk_arm_meas = np.asarray(arm_model.forwardKinematics(q_meas, 6), dtype=np.float64).reshape(4, 4)
                ee_transform_arm_meas = ee_transform_sdk_arm_meas @ act_ee_from_sdk_ee
            ee_transform_act_meas = act_from_arm @ ee_transform_arm_meas
            q_feedback = q_meas if np.isfinite(q_meas).all() else q_cmd
            pose_q_cache.add_transform(ee_transform_act_meas, q_feedback)
            # Solve around the joint state that is being commanded now. This
            # prevents an unexecuted, far-ahead q_goal from dragging local IK
            # onto another solution branch.
            q_seed = q_plan.copy() if np.isfinite(q_plan).all() else q_feedback

            ik_ok = False
            ik_source = ""
            new_command = bool(
                has_action
                and action_age <= args.command_timeout_s
                and action_stamp > last_processed_action_stamp
            )
            if has_action and action_age <= args.command_timeout_s:
                if action_stamp > last_processed_gripper_action_stamp:
                    last_processed_gripper_action_stamp = action_stamp
                    raw_gripper_action = float(action[GRIPPER_INDEX])
                    mapped_gripper_goal = map_act_gripper_to_z1(raw_gripper_action, args)
                    gripper_goal, gripper_close_latch_count, gripper_force_closed = apply_gripper_close_latch(
                        mapped_gripper_goal,
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
                    update_shared_gripper_latch_status(
                        shared,
                        action_raw=raw_gripper_action,
                        goal_before_latch=mapped_gripper_goal,
                        goal_after_latch=gripper_goal,
                        latch_count=gripper_close_latch_count,
                        force_closed=gripper_force_closed,
                    )
                gripper_idle_neutral_sent = False
                if new_command:
                    last_processed_action_stamp = action_stamp
                    target_pos_act = action[EE_POS_SLICE].astype(np.float64)
                    target_quat_act = normalize_quat_xyzw(action[EE_QUAT_SLICE])
                    candidate: np.ndarray | None = None
                    attempt_errors: list[str] = []
                    candidate_note = ""

                    if action_mode in ("joint", "joint_target", "home") and explicit_joint_target is not None:
                        candidate = np.clip(
                            np.asarray(explicit_joint_target, dtype=np.float64).reshape(6),
                            joint_min,
                            joint_max,
                        )
                        candidate_delta = float(np.max(np.abs(candidate - q_seed)))
                        if candidate_delta > float(args.joint_target_max_delta):
                            attempt_errors.append(f"joint_target_jump:{candidate_delta:.3f}")
                            candidate = None
                        else:
                            ik_source = "joint_target"
                    else:
                        resolve_ik = should_resolve_ik(
                            target_pos_act,
                            target_quat_act,
                            last_target_pos_act,
                            last_target_quat_act,
                            args.ik_recompute_pos_delta,
                            math.radians(args.ik_recompute_rot_delta_deg),
                        )
                        if not resolve_ik:
                            ik_ok = True
                            ik_source = f"hold:{last_ik_source}" if last_ik_source else "hold"

                    if candidate is None and not ik_ok and not attempt_errors:
                        target_transform_act = pose_to_transform(target_pos_act, target_quat_act)
                        target_tool_transform_arm = arm_from_act @ target_transform_act
                        target_sdk_transform_arm = target_tool_transform_arm @ sdk_ee_from_act_ee

                        for check_in_workspace, source_name in ((False, "local"), (True, "global")):
                            try:
                                with sdk_lock:
                                    has_ik, ik_result = arm_model.inverseKinematics(
                                        target_sdk_transform_arm,
                                        q_seed.astype(np.float64),
                                        bool(check_in_workspace),
                                    )
                                if bool(has_ik):
                                    proposed = np.asarray(ik_result, dtype=np.float64).reshape(6)
                                    proposed = np.clip(proposed, joint_min, joint_max)
                                    candidate_delta = float(np.max(np.abs(proposed - q_seed)))
                                    if candidate_delta > float(args.ik_max_joint_delta):
                                        attempt_errors.append(f"{source_name}_jump:{candidate_delta:.3f}")
                                        continue
                                    candidate = proposed
                                    ik_source = source_name
                                    break
                                attempt_errors.append(f"{source_name}_failed")
                            except Exception as exc:
                                attempt_errors.append(f"{source_name}_exception:{exc}")

                        if candidate is None and bool(args.soft_ik_fallback):
                            try:
                                soft_result = solve_soft_pose_ik(
                                    arm_model,
                                    sdk_lock,
                                    target_tool_transform_arm,
                                    q_seed,
                                    joint_min,
                                    joint_max,
                                    act_ee_from_sdk_ee,
                                    max_iterations=args.soft_ik_max_iterations,
                                    damping=args.soft_ik_damping,
                                    orientation_weight=args.soft_ik_orientation_weight,
                                    seed_weight=args.soft_ik_seed_weight,
                                    max_joint_step=args.soft_ik_max_joint_step,
                                    position_tolerance=args.soft_ik_position_tolerance,
                                    min_position_improvement=args.soft_ik_min_position_improvement,
                                )
                                soft_delta = float(np.max(np.abs(soft_result.q - q_seed)))
                                if not soft_result.success:
                                    attempt_errors.append(
                                        "soft_failed:"
                                        f"p={soft_result.position_error:.4f},"
                                        f"r={math.degrees(soft_result.orientation_error):.1f}"
                                    )
                                elif soft_delta > float(args.ik_max_joint_delta):
                                    attempt_errors.append(f"soft_jump:{soft_delta:.3f}")
                                else:
                                    candidate = np.clip(soft_result.q, joint_min, joint_max)
                                    ik_source = "soft"
                                    candidate_note = (
                                        "soft_ik:"
                                        f"p={soft_result.position_error:.4f},"
                                        f"r={math.degrees(soft_result.orientation_error):.1f},"
                                        f"n={soft_result.iterations}"
                                    )
                            except Exception as exc:
                                attempt_errors.append(f"soft_exception:{exc}")

                        if (
                            candidate is None
                            and bool(args.ik_cache)
                            and not bool(args.soft_ik_fallback)
                        ):
                            cached_q, cache_stats = pose_q_cache.lookup(
                                target_transform_act,
                                q_seed,
                                pos_tolerance=args.ik_cache_pos_tolerance,
                                rot_tolerance_rad=math.radians(args.ik_cache_rot_tolerance_deg),
                                max_joint_delta=min(
                                    float(args.ik_cache_max_joint_delta),
                                    float(args.ik_max_joint_delta),
                                ),
                            )
                            if cached_q is not None:
                                candidate = np.clip(cached_q, joint_min, joint_max)
                                ik_source = "cache"
                            else:
                                attempt_errors.append(
                                    "cache_miss:"
                                    f"n={int(cache_stats.get('cache_size', 0.0))}"
                                )

                    if candidate is not None:
                        q_goal = candidate
                        packet_dt = (
                            float(action_stamp - last_successful_waypoint_stamp)
                            if last_successful_waypoint_stamp > 0.0
                            else float(args.joint_trajectory_duration_s)
                        )
                        packet_dt = max(packet_dt, 1.0e-3)
                        raw_waypoint_velocity = (q_goal - last_successful_waypoint) / packet_dt
                        raw_waypoint_velocity = np.clip(
                            raw_waypoint_velocity,
                            -joint_speed_limit,
                            joint_speed_limit,
                        )
                        velocity_alpha = float(np.clip(args.joint_waypoint_velocity_alpha, 0.0, 1.0))
                        waypoint_velocity = (
                            velocity_alpha * raw_waypoint_velocity
                            + (1.0 - velocity_alpha) * last_waypoint_velocity
                        )
                        segment_duration = joint_trajectory.retarget(
                            q_goal,
                            waypoint_velocity,
                            loop_now,
                            args.joint_trajectory_duration_s,
                            joint_speed_limit,
                            joint_acceleration_limit,
                        )
                        last_successful_waypoint = q_goal.copy()
                        last_successful_waypoint_stamp = action_stamp
                        last_waypoint_velocity = waypoint_velocity.copy()
                        last_brake_action_stamp = -1.0
                        last_target_pos_act = target_pos_act.copy()
                        last_target_quat_act = target_quat_act.copy()
                        last_ik_source = ik_source
                        ik_ok = True
                        last_error = candidate_note
                        if segment_duration > float(args.joint_trajectory_duration_s) * 1.05:
                            extension_note = f"trajectory_extended:{segment_duration:.4f}s"
                            last_error = f"{last_error};{extension_note}" if last_error else extension_note
                    elif not ik_ok:
                        ik_fail_count += 1
                        last_error = "ik_failed:" + ",".join(attempt_errors)
                        if last_brake_action_stamp != action_stamp:
                            joint_trajectory.brake(
                                loop_now,
                                args.joint_brake_duration_s,
                                joint_speed_limit,
                                joint_acceleration_limit,
                            )
                            last_brake_action_stamp = action_stamp
                            last_waypoint_velocity[:] = 0.0
                else:
                    ik_ok = True
                    ik_source = f"hold:{last_ik_source}" if last_ik_source else "hold"

                gripper_target = gripper_goal
            else:
                gripper_close_latch_count = 0
                gripper_force_closed = False
                last_processed_gripper_action_stamp = 0.0
                update_shared_gripper_latch_status(shared)
                gripper_target = None
                if not gripper_idle_neutral_sent:
                    with sdk_lock:
                        gripper_now = float(arm.lowstate.getGripperQ())
                    gripper_cmd = float(np.clip(gripper_now, args.gripper_min, args.gripper_max))
                    gripper_qd_cmd = 0.0
                    gripper_goal = gripper_cmd
                    gripper_target = gripper_cmd
                    gripper_idle_neutral_sent = True
                if last_brake_action_stamp != action_stamp:
                    joint_trajectory.brake(
                        loop_now,
                        args.joint_brake_duration_s,
                        joint_speed_limit,
                        joint_acceleration_limit,
                    )
                    last_brake_action_stamp = action_stamp
                    last_waypoint_velocity[:] = 0.0
                if has_action:
                    last_error = f"stale_action:{action_age:.3f}s"
                else:
                    last_error = "no_action"

            # Each 25 Hz IK waypoint is spread over a causal quintic segment.
            # The 500 Hz loop therefore sends continuous q/qd/qdd rather than
            # reaching a nearby waypoint early and waiting for the next packet.
            q_next, qd_cmd, qdd_cmd = joint_trajectory.sample(loop_now)
            if gripper_target is not None:
                gripper_command_max = (
                    max(float(args.gripper_max), float(args.gripper_close_latch_target))
                    if bool(gripper_force_closed)
                    else float(args.gripper_max)
                )
                gripper_cmd, gripper_qd_cmd = advance_gripper_command(
                    gripper_cmd,
                    gripper_qd_cmd,
                    gripper_target,
                    period,
                    args.max_gripper_speed,
                    args.max_gripper_acceleration,
                    args.gripper_min,
                    gripper_command_max,
                )
                gripper_to_send: float | None = gripper_cmd
            else:
                gripper_to_send = None
                gripper_qd_cmd = 0.0

            q_next, qd_cmd = send_lowcmd_joint_target(
                arm,
                arm_model,
                sdk_lock,
                q_next,
                qd_cmd,
                qdd_cmd,
                gripper_to_send,
                gripper_qd_cmd,
                joint_min,
                joint_max,
            )

            q_cmd = q_next
            update_shared_control_status(
                shared,
                ik_ok=ik_ok,
                ik_fail_count=ik_fail_count,
                control_count=control_count,
                last_error=last_error,
                ik_source=ik_source,
            )

            control_count += 1
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()
    finally:
        arm_stop_event.set()
        state_thread.join(timeout=2.0)
        with shared.lock:
            shutdown_requested = bool(shared.shutdown_requested)
        if shutdown_requested:
            update_shared_shutdown_status(
                shared,
                requested=True,
                active=True,
                done=False,
            )
        if args.passive_on_exit or args.back_to_start_on_exit:
            try:
                with sdk_lock:
                    arm.loopOn()
                    if args.back_to_start_on_exit:
                        arm.backToStart()
                    if args.passive_on_exit:
                        arm.setFsm(unitree_arm_interface.ArmFSMState.PASSIVE)
                    arm.loopOff()
            except Exception as exc:
                print(f"warning: failed to safely stop Z1 on exit: {exc}", file=sys.stderr, flush=True)
                if shutdown_requested:
                    update_shared_shutdown_status(
                        shared,
                        requested=True,
                        active=False,
                        done=False,
                        error=str(exc),
                    )
            else:
                if shutdown_requested:
                    update_shared_shutdown_status(
                        shared,
                        requested=True,
                        active=False,
                        done=True,
                    )
                    print("z1_bridge shutdown_back_to_start done", flush=True)
        elif shutdown_requested:
            update_shared_shutdown_status(
                shared,
                requested=True,
                active=False,
                done=False,
                error="back_to_start_on_exit_disabled",
            )
        if shutdown_requested:
            # Keep the IO thread alive briefly so the final completion state is
            # transmitted to the ACT process before the bridge exits.
            time.sleep(0.15)
            stop_event.set()


def make_udp_socket(bind_host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_host, int(port)))
    sock.setblocking(False)
    return sock


def add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str = "") -> None:
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", help=help_text)
    parser.add_argument("--no_" + dest, dest=dest, action="store_false")
    parser.set_defaults(**{dest: bool(default)})


def io_loop(
    args: argparse.Namespace,
    shared: SharedBridgeState,
    stop_event: threading.Event,
) -> None:
    action_sock = make_udp_socket(args.udp_bind_host, args.action_udp_port)
    vel_sock = make_udp_socket(args.udp_bind_host, args.vel_state_udp_port)
    tx_sock: socket.socket | None = None
    tx_addr: tuple[str, int] | None = None
    if args.state_tx_host and args.state_tx_port > 0:
        tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx_addr = (args.state_tx_host, int(args.state_tx_port))

    log_file = None
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log_path.open("a", encoding="utf-8")

    print(
        f"z1_bridge IO listening action_udp={args.udp_bind_host}:{args.action_udp_port} "
        f"vel_state_udp={args.udp_bind_host}:{args.vel_state_udp_port}",
        flush=True,
    )
    state_period = 1.0 / max(float(args.state_hz), 1.0e-6)
    print_period = 1.0 / max(float(args.print_hz), 1.0e-6) if args.print_hz > 0 else float("inf")
    next_state_t = time.monotonic()
    next_print_t = time.monotonic()

    try:
        while not stop_event.is_set():
            timeout = max(0.0, min(next_state_t, next_print_t) - time.monotonic())
            readable, _, _ = select.select([action_sock, vel_sock], [], [], min(timeout, 0.05))
            now = time.time()
            for sock in readable:
                try:
                    data, addr = sock.recvfrom(4096)
                except BlockingIOError:
                    continue
                if sock is action_sock:
                    try:
                        payload = _try_parse_json_packet(data)
                        bridge_command = (
                            str(payload.get("bridge_command", "")).strip().lower()
                            if isinstance(payload, dict)
                            else ""
                        )
                        if bridge_command == "shutdown_back_to_start":
                            with shared.lock:
                                first_request = not bool(shared.shutdown_requested)
                                if first_request:
                                    shared.shutdown_requested = True
                                    shared.shutdown_active = False
                                    shared.shutdown_done = False
                                    shared.shutdown_error = ""
                                    shared.last_error = "shutdown_back_to_start_requested"
                            if first_request:
                                print(
                                    f"z1_bridge shutdown_back_to_start requested by {addr[0]}:{addr[1]}",
                                    flush=True,
                                )
                            continue
                        if bridge_command:
                            raise ValueError(f"unknown bridge_command: {bridge_command}")
                        action, action_mode, joint_target = parse_action_command(data)
                        action[EE_QUAT_SLICE] = normalize_quat_xyzw(action[EE_QUAT_SLICE]).astype(np.float32)
                        with shared.lock:
                            shared.action = action.astype(np.float32)
                            shared.action_stamp = now
                            shared.action_source = f"{addr[0]}:{addr[1]}"
                            shared.has_action = True
                            shared.action_mode = action_mode
                            shared.has_joint_target = joint_target is not None
                            if joint_target is not None:
                                shared.joint_target = joint_target.astype(np.float32)
                            shared.last_error = ""
                    except Exception as exc:
                        with shared.lock:
                            shared.last_error = f"bad_action_packet:{exc}"
                        print(f"bad action packet from {addr}: {exc}", file=sys.stderr, flush=True)
                else:
                    try:
                        vel_state = parse_vel_state_packet(data)
                        with shared.lock:
                            shared.vel_state = vel_state.astype(np.float32)
                            shared.vel_state_stamp = now
                            shared.vel_state_source = f"{addr[0]}:{addr[1]}"
                            shared.has_vel_state = True
                            shared.last_error = ""
                    except Exception as exc:
                        with shared.lock:
                            shared.last_error = f"bad_vel_state_packet:{exc}"
                        print(f"bad vel_state packet from {addr}: {exc}", file=sys.stderr, flush=True)

            mono = time.monotonic()
            if mono >= next_state_t:
                record = shared.snapshot(args)
                if tx_sock is not None and tx_addr is not None:
                    tx_sock.sendto(json.dumps(record, ensure_ascii=False).encode("utf-8"), tx_addr)
                if log_file is not None:
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                next_state_t += state_period
                if next_state_t < mono - state_period:
                    next_state_t = mono + state_period

            if args.print_hz > 0 and mono >= next_print_t:
                record = shared.snapshot(args)
                state = record["state"]
                print(
                    "act_state "
                    f"vx={state[0]:+.3f} yaw_rate={state[1]:+.3f} "
                    f"ee=({state[2]:+.3f},{state[3]:+.3f},{state[4]:+.3f}) "
                    f"grip={state[9]:+.3f} ik={record['ik_ok']} src={record.get('ik_source', '')} "
                    f"age_action={record['action_age_s']} age_vel={record['vel_state_age_s']} "
                    f"err={record['last_error']}",
                    flush=True,
                )
                next_print_t += print_period
                if next_print_t < mono - print_period:
                    next_print_t = mono + print_period
    finally:
        action_sock.close()
        vel_sock.close()
        if tx_sock is not None:
            tx_sock.close()
        if log_file is not None:
            log_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z1 LOWCMD bridge for Door ACT EE actions.")
    add_bool_argument(
        parser,
        "--enable_arm",
        default=True,
        help_text="Instantiate Z1 and send LOWCMD by default. Use --no_enable_arm for a dry-run plumbing test.",
    )
    parser.add_argument("--z1_sdk_lib", type=Path, default=Path(os.environ.get("Z1_SDK_LIB", "/home/anx/door_act_deploy/z1_sdk/lib")))
    add_bool_argument(parser, "--has_gripper", default=True)
    parser.add_argument("--udp_bind_host", type=str, default="0.0.0.0")
    parser.add_argument("--action_udp_port", type=int, default=15011)
    parser.add_argument("--vel_state_udp_port", type=int, default=15012)
    parser.add_argument(
        "--state_tx_host",
        type=str,
        default="127.0.0.1",
        help="Host to send assembled ACT state JSON over UDP. Empty string disables state transmission.",
    )
    parser.add_argument("--state_tx_port", type=int, default=15013)
    parser.add_argument("--state_hz", type=float, default=25.0)
    parser.add_argument("--print_hz", type=float, default=5.0)
    parser.add_argument("--log_path", type=Path, default=None)
    parser.add_argument("--command_timeout_s", type=float, default=0.25)
    parser.add_argument("--vel_timeout_s", type=float, default=0.5)
    parser.add_argument("--dry_run_hz", type=float, default=100.0)
    parser.add_argument("--arm_state_hz", type=float, default=50.0, help="Asynchronous Z1 lowstate/FK read frequency.")
    add_bool_argument(
        parser,
        "--zero_joints_on_start",
        default=True,
        help_text=(
            "Before accepting ACT EE commands, call the Z1 controller's "
            "backToStart() and confirm its calibrated home feedback."
        ),
    )
    add_bool_argument(
        parser,
        "--zero_joints_strict",
        default=True,
        help_text="Abort startup if calibrated home feedback is not stable before --zero_joints_timeout_s.",
    )
    parser.add_argument(
        "--zero_joints_target",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        help=(
            "Deprecated compatibility option. Startup home is captured from "
            "backToStart() feedback; this value is never commanded."
        ),
    )
    parser.add_argument(
        "--zero_joints_tolerance",
        type=float,
        default=0.03,
        help="Maximum per-joint drift in rad from the home_q captured after backToStart().",
    )
    parser.add_argument(
        "--startup_home_max_joint_speed",
        type=float,
        default=0.2,
        help="Maximum absolute measured joint speed in rad/s while confirming calibrated home.",
    )
    parser.add_argument("--zero_joints_hold_s", type=float, default=0.30)
    parser.add_argument("--zero_joints_timeout_s", type=float, default=12.0)
    parser.add_argument(
        "--zero_joints_command_mode",
        choices=["controller_home"],
        default="controller_home",
        help="Compatibility option; startup zero always uses the Z1 controller's backToStart().",
    )
    parser.add_argument(
        "--zero_joints_gripper",
        type=float,
        default=-math.pi / 2.0,
        help=(
            "Gripper target during startup zero. Default is the project-wide "
            "fully-open gripper pose, -pi/2; closed is 0.0."
        ),
    )
    add_bool_argument(
        parser,
        "--startup_gripper_open_once",
        default=True,
        help_text=(
            "During startup, send the fully-open gripper target only once, then "
            "neutralize to current feedback until ACT commands arrive. This avoids "
            "continuously pushing the gripper against its open stop before inference."
        ),
    )
    parser.add_argument("--dry_run_initial_ee_xyz", type=float, nargs=3, default=[0.35, 0.0, 0.25])
    parser.add_argument("--dry_run_initial_ee_quat", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0])
    parser.add_argument(
        "--arm_from_act_xyz",
        type=float,
        nargs=3,
        default=(-A2W_Z1_MOUNT_XYZ_ACT_FROM_ARM).tolist(),
        help=(
            "Translation of ACT/A2W-base frame expressed as a transform into the Z1 SDK arm frame. "
            "Default subtracts the simulated A2W->Z1 mount offset [0.174, 0, 0.142]."
        ),
    )
    parser.add_argument(
        "--arm_from_act_rpy",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Rotation of ACT/A2W-base frame into the Z1 SDK arm frame, roll pitch yaw in radians.",
    )
    parser.add_argument(
        "--act_ee_from_sdk_ee_xyz",
        type=float,
        nargs=3,
        default=Z1_SDK_EE_TO_ACT_EE_XYZ.tolist(),
        help=(
            "Tool translation from the Unitree SDK FK/IK EE frame to the ACT/sim ee_gripper_link frame. "
            "Default +0.086m in local x aligns Z1 SDK q=0 with the simulated ee_gripper_link q=0 after "
            "subtracting the A2W mount offset."
        ),
    )
    parser.add_argument(
        "--act_ee_from_sdk_ee_rpy",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Tool rotation from the Unitree SDK FK/IK EE frame to the ACT/sim ee_gripper_link frame.",
    )
    parser.add_argument(
        "--ik_workspace_check",
        action="store_true",
        help="Deprecated compatibility flag; the controller now always tries local IK then global IK.",
    )
    add_bool_argument(parser, "--ik_cache", default=True, help_text="Use observed FK pose->q cache when both IK modes fail.")
    parser.add_argument("--ik_cache_size", type=int, default=2000)
    parser.add_argument("--ik_cache_pos_tolerance", type=float, default=0.02, help="meters")
    parser.add_argument("--ik_cache_rot_tolerance_deg", type=float, default=12.0)
    parser.add_argument("--ik_cache_min_pos_delta", type=float, default=0.003, help="meters between deduped cache samples")
    parser.add_argument("--ik_cache_min_rot_delta_deg", type=float, default=2.0)
    parser.add_argument("--ik_cache_max_joint_delta", type=float, default=1.5, help="max abs joint delta allowed for cache fallback")
    parser.add_argument("--ik_global_max_joint_delta", type=float, default=1.5, help="max abs joint delta allowed for global IK fallback")
    parser.add_argument(
        "--ik_max_joint_delta",
        type=float,
        default=1.5,
        help="max abs joint delta allowed for every local/global/cache IK result relative to current q_cmd.",
    )
    add_bool_argument(
        parser,
        "--soft_ik_fallback",
        default=True,
        help_text=(
            "After exact local/global 6D IK fails, solve a bounded local soft IK "
            "that prioritizes ACT tool-point position and treats orientation as a low-weight objective."
        ),
    )
    parser.add_argument(
        "--soft_ik_max_iterations",
        type=int,
        default=10,
        help="bounded iteration count; representative NX fallback solves take about 3-6 ms.",
    )
    parser.add_argument("--soft_ik_damping", type=float, default=0.03)
    parser.add_argument(
        "--soft_ik_orientation_weight",
        type=float,
        default=0.02,
        help="meters per radian equivalent weight; position remains the dominant objective.",
    )
    parser.add_argument(
        "--soft_ik_seed_weight",
        type=float,
        default=1.0e-4,
        help="joint-space regularization toward the current q_cmd seed.",
    )
    parser.add_argument("--soft_ik_max_joint_step", type=float, default=0.08, help="rad per solver iteration")
    parser.add_argument("--soft_ik_position_tolerance", type=float, default=0.003, help="meters")
    parser.add_argument(
        "--soft_ik_min_position_improvement",
        type=float,
        default=0.001,
        help="meters of required position improvement when the closest pose cannot reach the tolerance.",
    )
    parser.add_argument(
        "--joint_target_max_delta",
        type=float,
        default=2.5,
        help="max abs delta allowed for an explicit real-time joint waypoint relative to current q_cmd.",
    )
    parser.add_argument(
        "--ik_recompute_pos_delta",
        type=float,
        default=0.001,
        help="meters; recompute IK when the raw ACT EE target position changes more than this.",
    )
    parser.add_argument(
        "--ik_recompute_rot_delta_deg",
        type=float,
        default=1.0,
        help="degrees; recompute IK when the raw ACT EE target orientation changes more than this.",
    )
    parser.add_argument(
        "--joint_command_mode",
        choices=["online_quintic"],
        default="online_quintic",
        help="Compatibility option; successful IK waypoints are interpolated by the online quintic generator.",
    )
    parser.add_argument("--max_joint_speed", type=float, default=3.0, help="rad/s cap before SDK joint speed limits.")
    parser.add_argument(
        "--max_joint_acceleration",
        type=float,
        default=15.0,
        help="rad/s^2 cap used to time-scale each online quintic segment.",
    )
    parser.add_argument(
        "--joint_trajectory_duration_s",
        type=float,
        default=0.04,
        help=(
            "Nominal duration for each online IK waypoint segment. The default 0.04 s "
            "maps one 25 Hz EE waypoint to twenty 500 Hz LOWCMD samples; far targets "
            "are automatically stretched to respect the speed and acceleration limits."
        ),
    )
    parser.add_argument(
        "--joint_waypoint_velocity_alpha",
        type=float,
        default=0.7,
        help="blend weight for the latest causal waypoint velocity estimate.",
    )
    parser.add_argument(
        "--joint_brake_duration_s",
        type=float,
        default=0.08,
        help="minimum smooth braking duration after an IK failure or stale command.",
    )
    parser.add_argument(
        "--max_gripper_speed",
        type=float,
        default=3.14,
        help=(
            "rad/s cap for the smooth Z1 gripper command trajectory. "
            "The Z1 controller's gripper limit is pi rad/s."
        ),
    )
    parser.add_argument(
        "--max_gripper_acceleration",
        type=float,
        default=120.0,
        help="rad/s^2 acceleration cap for smooth gripper starts and stops.",
    )
    parser.add_argument("--gripper_min", type=float, default=-math.pi / 2.0)
    parser.add_argument("--gripper_max", type=float, default=0.0)
    parser.add_argument("--gripper_scale", type=float, default=1.0)
    parser.add_argument("--gripper_offset", type=float, default=0.0)
    parser.add_argument("--initial_gripper", type=float, default=-math.pi / 2.0)
    add_bool_argument(
        parser,
        "--gripper_close_latch",
        default=True,
        help_text=(
            "If mapped ACT gripper target stays above --gripper_close_latch_threshold for "
            "--gripper_close_latch_confirm_steps action packets, send --gripper_close_latch_target. "
            "One subsequent packet below the threshold exits the latch immediately."
        ),
    )
    parser.add_argument(
        "--gripper_close_latch_threshold",
        type=float,
        default=-0.5,
        help="Mapped gripper target threshold in rad. Values >= this threshold count as close intent.",
    )
    parser.add_argument(
        "--gripper_close_latch_confirm_steps",
        type=int,
        default=3,
        help="Number of consecutive fresh ACT action packets required to force closed gripper.",
    )
    parser.add_argument(
        "--gripper_close_latch_target",
        type=float,
        default=0.1,
        help=(
            "Gripper target sent after close latch is confirmed. "
            "This may be above --gripper_max; the extra range is used only while force-closed."
        ),
    )
    add_bool_argument(parser, "--passive_on_exit", default=True)
    add_bool_argument(
        parser,
        "--back_to_start_on_exit",
        default=True,
        help_text="Call Z1 backToStart() before switching passive whenever the bridge exits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.gripper_close_latch_confirm_steps) <= 0:
        raise ValueError("--gripper_close_latch_confirm_steps must be positive.")
    if float(args.gripper_min) > float(args.gripper_max):
        raise ValueError("--gripper_min must be <= --gripper_max.")
    if float(args.gripper_close_latch_target) < float(args.gripper_min):
        raise ValueError("--gripper_close_latch_target must be >= --gripper_min.")

    shared = SharedBridgeState()
    stop_event = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    threads = [
        threading.Thread(target=arm_command_loop, args=(args, shared, stop_event), daemon=True, name="z1-command"),
        threading.Thread(target=io_loop, args=(args, shared, stop_event), daemon=True, name="z1-io"),
    ]
    for thread in threads:
        thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
            if stop_event.is_set():
                break
            for thread in threads:
                if not thread.is_alive() and not stop_event.is_set():
                    stop_event.set()
                    raise RuntimeError(f"worker thread exited unexpectedly: {thread.name}")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
