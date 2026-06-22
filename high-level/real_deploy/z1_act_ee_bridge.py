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

EE pose is in the ACT/base frame.  By default the ACT frame is treated as the
Z1 arm base frame; use --arm_from_act_xyz/--arm_from_act_rpy after measuring the
real A2-W/Z1 mount transform.

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
    startup_zero_error: str = ""
    last_error: str = ""

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
                "startup_zero_error": self.startup_zero_error,
                "last_error": self.last_error,
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
    error: str = "",
) -> None:
    with shared.lock:
        shared.startup_zero_requested = bool(requested)
        shared.startup_zero_active = bool(active)
        shared.startup_zero_done = bool(done)
        shared.startup_zero_max_err = float(max_err)
        shared.startup_zero_error = str(error)


def get_action_snapshot(shared: SharedBridgeState) -> tuple[np.ndarray, bool, float]:
    now = time.time()
    with shared.lock:
        age = float("inf") if not shared.has_action else now - shared.action_stamp
        return shared.action.copy(), bool(shared.has_action), age


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


def send_lowcmd_joint_target(
    arm,
    arm_model,
    sdk_lock: threading.Lock,
    q_target: np.ndarray,
    qd_target: np.ndarray,
    gripper_target: float,
    gripper_qd: float,
    joint_min: np.ndarray,
    joint_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_next = np.asarray(q_target, dtype=np.float64).reshape(6)
    qd_cmd = np.asarray(qd_target, dtype=np.float64).reshape(6)
    try:
        with sdk_lock:
            q_next, qd_cmd = arm_model.jointProtect(q_next, qd_cmd)
        q_next = np.asarray(q_next, dtype=np.float64).reshape(6)
        qd_cmd = np.asarray(qd_cmd, dtype=np.float64).reshape(6)
    except Exception:
        # Some SDK builds expose jointProtect but do not accept writable Eigen refs cleanly.
        q_next = np.clip(q_next, joint_min, joint_max)

    with sdk_lock:
        tau_cmd = arm_model.inverseDynamics(q_next, qd_cmd, np.zeros(6), np.zeros(6))
        arm.setArmCmd(q_next, qd_cmd, tau_cmd)
        arm.setGripperCmd(float(gripper_target), float(gripper_qd), 0.0)
        arm.sendRecv()
    return q_next, qd_cmd


def move_z1_to_startup_zero(
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
) -> tuple[np.ndarray, np.ndarray, float, int]:
    requested = bool(args.zero_joints_on_start)
    if not requested:
        update_shared_startup_zero_status(shared, requested=False, active=False, done=True)
        return q_cmd, np.zeros(6, dtype=np.float64), gripper_cmd, control_count

    q_zero = np.asarray(args.zero_joints_target, dtype=np.float64).reshape(6)
    q_zero = np.clip(q_zero, joint_min, joint_max)
    gripper_target = gripper_cmd
    if args.zero_joints_gripper is not None:
        gripper_target = float(np.clip(float(args.zero_joints_gripper), args.gripper_min, args.gripper_max))

    timeout_s = max(float(args.zero_joints_timeout_s), 0.0)
    tolerance = max(float(args.zero_joints_tolerance), 0.0)
    hold_s = max(float(args.zero_joints_hold_s), 0.0)
    deadline = time.monotonic() + timeout_s if timeout_s > 0.0 else float("inf")
    hold_start: float | None = None
    last_err = float("inf")
    qd_cmd = np.zeros(6, dtype=np.float64)
    next_t = time.monotonic()
    timed_out = False

    print(
        "z1_bridge startup_zero begin "
        f"target={np.round(q_zero, 4).tolist()} tolerance={tolerance:.4f} "
        f"timeout_s={timeout_s:.2f} hold_s={hold_s:.2f}",
        flush=True,
    )
    update_shared_startup_zero_status(shared, requested=True, active=True, done=False, max_err=last_err)

    while not stop_event.is_set():
        with sdk_lock:
            q_meas = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
        q_current = q_meas if np.isfinite(q_meas).all() else q_cmd
        last_err = float(np.max(np.abs(q_zero - q_current)))
        if last_err <= tolerance:
            if hold_start is None:
                hold_start = time.monotonic()
            if time.monotonic() - hold_start >= hold_s:
                break
        else:
            hold_start = None

        if time.monotonic() >= deadline:
            msg = f"startup_zero_timeout:max_err={last_err:.4f}"
            update_shared_startup_zero_status(shared, requested=True, active=False, done=False, max_err=last_err, error=msg)
            if bool(args.zero_joints_strict):
                raise RuntimeError(msg)
            print(f"warning: {msg}; proceeding because --no_zero_joints_strict was set.", file=sys.stderr, flush=True)
            timed_out = True
            break

        q_next, qd_cmd = clamp_joint_target(q_current, q_zero, period, joint_speed_limit)
        q_next, qd_cmd = send_lowcmd_joint_target(
            arm,
            arm_model,
            sdk_lock,
            q_next,
            qd_cmd,
            gripper_target,
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
            last_error="startup_zero_joints",
            ik_source="startup_zero",
        )
        update_shared_startup_zero_status(shared, requested=True, active=True, done=False, max_err=last_err)
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
    if np.isfinite(q_meas).all():
        q_cmd = q_meas
        last_err = float(np.max(np.abs(q_zero - q_meas)))
    qd_cmd = np.zeros(6, dtype=np.float64)
    q_cmd, qd_cmd = send_lowcmd_joint_target(
        arm,
        arm_model,
        sdk_lock,
        q_zero,
        qd_cmd,
        gripper_target,
        0.0,
        joint_min,
        joint_max,
    )
    update_shared_startup_zero_status(shared, requested=True, active=False, done=True, max_err=last_err)
    print(f"z1_bridge startup_zero done max_err={last_err:.4f}", flush=True)
    return q_zero.copy(), qd_cmd, gripper_target, control_count


def arm_state_read_loop(
    args: argparse.Namespace,
    shared: SharedBridgeState,
    stop_event: threading.Event,
    arm_stop_event: threading.Event,
    arm,
    arm_model,
    sdk_lock: threading.Lock,
    act_from_arm: np.ndarray,
) -> None:
    period = 1.0 / max(float(args.arm_state_hz), 1.0)
    next_t = time.monotonic()
    while not stop_event.is_set() and not arm_stop_event.is_set():
        try:
            with sdk_lock:
                q_actual = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
                qd_actual = np.asarray(arm.lowstate.getQd(), dtype=np.float64).reshape(6)
                gripper_actual = float(arm.lowstate.getGripperQ())
                ee_transform_arm = np.asarray(arm_model.forwardKinematics(q_actual, 6), dtype=np.float64).reshape(4, 4)
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
    period = 1.0 / max(float(args.dry_run_hz), 1.0)
    q_cmd = np.zeros(6, dtype=np.float64)
    qd_cmd = np.zeros(6, dtype=np.float64)
    gripper_cmd = float(np.clip(args.initial_gripper, args.gripper_min, args.gripper_max))
    joint_speed_limit = np.full(6, float(args.max_joint_speed), dtype=np.float64)
    ik_fail_count = 0
    control_count = 0

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
            action, has_action, action_age = get_action_snapshot(shared)
            if has_action and action_age <= args.command_timeout_s:
                ee_pos = action[EE_POS_SLICE].astype(np.float32)
                ee_quat = normalize_quat_xyzw(action[EE_QUAT_SLICE]).astype(np.float32)
                gripper_cmd = map_act_gripper_to_z1(float(action[GRIPPER_INDEX]), args)
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
        f"joint_command_mode={args.joint_command_mode} "
        f"max_joint_speed={np.round(joint_speed_limit, 3)}",
        flush=True,
    )
    with sdk_lock:
        arm.setFsmLowcmd()
        q_cmd = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
        qd_cmd = np.zeros(6, dtype=np.float64)
        gripper_cmd = float(np.clip(float(arm.lowstate.getGripperQ()), args.gripper_min, args.gripper_max))
    q_cmd, qd_cmd, gripper_cmd, control_count = move_z1_to_startup_zero(
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
    )
    with sdk_lock:
        initial_transform_arm = np.asarray(arm_model.forwardKinematics(q_cmd, 6), dtype=np.float64).reshape(4, 4)
    q_goal = q_cmd.copy()
    gripper_goal = gripper_cmd
    last_target_pos_act: np.ndarray | None = None
    last_target_quat_act: np.ndarray | None = None
    last_ik_source = ""
    pose_q_cache = PoseQCache(
        max_size=args.ik_cache_size,
        min_pos_delta=args.ik_cache_min_pos_delta,
        min_rot_delta_rad=math.radians(args.ik_cache_min_rot_delta_deg),
    )
    pose_q_cache.add_transform(act_from_arm @ initial_transform_arm, q_cmd)
    next_t = time.monotonic()
    arm_stop_event = threading.Event()
    state_thread = threading.Thread(
        target=arm_state_read_loop,
        args=(args, shared, stop_event, arm_stop_event, arm, arm_model, sdk_lock, act_from_arm),
        daemon=True,
        name="z1-state",
    )
    state_thread.start()

    try:
        while not stop_event.is_set():
            last_error = ""
            action, has_action, action_age = get_action_snapshot(shared)
            with sdk_lock:
                q_meas = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
                gripper_meas = float(arm.lowstate.getGripperQ())
                ee_transform_arm_meas = np.asarray(arm_model.forwardKinematics(q_meas, 6), dtype=np.float64).reshape(4, 4)
            q_seed = q_meas if np.isfinite(q_meas).all() else q_cmd
            pose_q_cache.add_transform(act_from_arm @ ee_transform_arm_meas, q_seed)

            ik_ok = False
            ik_source = ""
            q_target = q_goal.copy()
            if has_action and action_age <= args.command_timeout_s:
                target_pos_act = action[EE_POS_SLICE].astype(np.float64)
                target_quat_act = normalize_quat_xyzw(action[EE_QUAT_SLICE])
                gripper_goal = map_act_gripper_to_z1(float(action[GRIPPER_INDEX]), args)
                if should_resolve_ik(
                    target_pos_act,
                    target_quat_act,
                    last_target_pos_act,
                    last_target_quat_act,
                    args.ik_recompute_pos_delta,
                    math.radians(args.ik_recompute_rot_delta_deg),
                ):
                    target_transform_act = pose_to_transform(target_pos_act, target_quat_act)
                    target_transform_arm = arm_from_act @ target_transform_act

                    attempt_errors: list[str] = []
                    for check_in_workspace, source_name in ((False, "local"), (True, "global")):
                        try:
                            with sdk_lock:
                                has_ik, ik_result = arm_model.inverseKinematics(
                                    target_transform_arm,
                                    q_seed.astype(np.float64),
                                    bool(check_in_workspace),
                                )
                            if bool(has_ik):
                                candidate = np.asarray(ik_result, dtype=np.float64).reshape(6)
                                candidate = np.clip(candidate, joint_min, joint_max)
                                candidate_delta = float(np.max(np.abs(candidate - q_seed)))
                                if source_name == "global" and candidate_delta > float(args.ik_global_max_joint_delta):
                                    attempt_errors.append(f"global_jump:{candidate_delta:.3f}")
                                    continue
                                q_goal = candidate
                                last_target_pos_act = target_pos_act.copy()
                                last_target_quat_act = target_quat_act.copy()
                                last_ik_source = source_name
                                ik_ok = True
                                ik_source = source_name
                                break
                            attempt_errors.append(f"{source_name}_failed")
                        except Exception as exc:
                            attempt_errors.append(f"{source_name}_exception:{exc}")

                    if (not ik_ok) and bool(args.ik_cache):
                        cached_q, cache_stats = pose_q_cache.lookup(
                            target_transform_act,
                            q_seed,
                            pos_tolerance=args.ik_cache_pos_tolerance,
                            rot_tolerance_rad=math.radians(args.ik_cache_rot_tolerance_deg),
                            max_joint_delta=args.ik_cache_max_joint_delta,
                        )
                        if cached_q is not None:
                            q_goal = np.clip(cached_q, joint_min, joint_max)
                            last_target_pos_act = target_pos_act.copy()
                            last_target_quat_act = target_quat_act.copy()
                            last_ik_source = "cache"
                            ik_ok = True
                            ik_source = "cache"
                        else:
                            attempt_errors.append(
                                "cache_miss:"
                                f"n={int(cache_stats.get('cache_size', 0.0))}"
                            )

                    if not ik_ok:
                        ik_fail_count += 1
                        last_error = "ik_failed:" + ",".join(attempt_errors)
                else:
                    # Same EE target as the previous packet: do not run IK again.
                    # Keep repeatedly sending the already-latched q_goal.
                    ik_ok = True
                    ik_source = f"hold:{last_ik_source}" if last_ik_source else "hold"

                q_target = q_goal.copy()
                gripper_target = gripper_goal
            else:
                # If ACT stops publishing, keep holding the last commanded joint
                # target instead of recomputing IK or interpolating.
                q_target = q_goal.copy()
                gripper_target = gripper_goal
                if has_action:
                    last_error = f"stale_action:{action_age:.3f}s"
                else:
                    last_error = "no_action"

            if args.joint_command_mode == "rate_limited":
                q_next, qd_cmd = clamp_joint_target(q_cmd, q_target, period, joint_speed_limit)
                gripper_delta_limit = float(args.max_gripper_speed) * period
                gripper_cmd = float(
                    np.clip(
                        gripper_cmd + np.clip(gripper_target - gripper_cmd, -gripper_delta_limit, gripper_delta_limit),
                        args.gripper_min,
                        args.gripper_max,
                    )
                )
                gripper_qd = float(
                    np.clip(
                        (gripper_cmd - gripper_meas) / max(period, 1.0e-6),
                        -args.max_gripper_speed,
                        args.max_gripper_speed,
                    )
                )
            else:
                # Direct hold mode: send the same IK joint target every cycle.
                q_next = np.asarray(q_target, dtype=np.float64).reshape(6)
                qd_cmd = np.zeros(6, dtype=np.float64)
                gripper_cmd = float(np.clip(gripper_target, args.gripper_min, args.gripper_max))
                gripper_qd = 0.0

            q_next, qd_cmd = send_lowcmd_joint_target(
                arm,
                arm_model,
                sdk_lock,
                q_next,
                qd_cmd,
                gripper_cmd,
                gripper_qd,
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
                        action = parse_action_packet(data)
                        action[EE_QUAT_SLICE] = normalize_quat_xyzw(action[EE_QUAT_SLICE]).astype(np.float32)
                        with shared.lock:
                            shared.action = action.astype(np.float32)
                            shared.action_stamp = now
                            shared.action_source = f"{addr[0]}:{addr[1]}"
                            shared.has_action = True
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
        help_text="Before accepting ACT EE commands, move the six Z1 arm joints to zero.",
    )
    add_bool_argument(
        parser,
        "--zero_joints_strict",
        default=True,
        help_text="Abort startup if the zero-joint pose is not reached before --zero_joints_timeout_s.",
    )
    parser.add_argument("--zero_joints_target", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--zero_joints_tolerance", type=float, default=0.03, help="rad")
    parser.add_argument("--zero_joints_hold_s", type=float, default=0.30)
    parser.add_argument("--zero_joints_timeout_s", type=float, default=12.0)
    parser.add_argument(
        "--zero_joints_gripper",
        type=float,
        default=-math.pi / 2.0,
        help=(
            "Gripper target during startup zero. Default is the project-wide "
            "fully-open gripper pose, -pi/2; closed is 0.0."
        ),
    )
    parser.add_argument("--dry_run_initial_ee_xyz", type=float, nargs=3, default=[0.35, 0.0, 0.25])
    parser.add_argument("--dry_run_initial_ee_quat", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0])
    parser.add_argument("--arm_from_act_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--arm_from_act_rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0])
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
        "--ik_recompute_pos_delta",
        type=float,
        default=0.001,
        help="meters; in direct_hold mode, recompute IK only when the EE target position changes more than this.",
    )
    parser.add_argument(
        "--ik_recompute_rot_delta_deg",
        type=float,
        default=1.0,
        help="degrees; in direct_hold mode, recompute IK only when target orientation changes more than this.",
    )
    parser.add_argument(
        "--joint_command_mode",
        choices=["direct_hold", "rate_limited"],
        default="rate_limited",
        help="direct_hold sends the IK q_goal unchanged every cycle; rate_limited uses the old joint-speed interpolation.",
    )
    parser.add_argument("--max_joint_speed", type=float, default=0.4, help="rad/s cap before SDK joint speed limits.")
    parser.add_argument("--max_gripper_speed", type=float, default=0.2, help="rad/s cap for Z1 gripper.")
    parser.add_argument("--gripper_min", type=float, default=-math.pi / 2.0)
    parser.add_argument("--gripper_max", type=float, default=0.0)
    parser.add_argument("--gripper_scale", type=float, default=1.0)
    parser.add_argument("--gripper_offset", type=float, default=0.0)
    parser.add_argument("--initial_gripper", type=float, default=-math.pi / 2.0)
    add_bool_argument(parser, "--passive_on_exit", default=True)
    parser.add_argument("--back_to_start_on_exit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
