#!/usr/bin/env python3
"""Standalone Pinocchio FK/IK for the Unitree Z1 arm.

This module is intentionally independent from the Isaac Gym door scripts.  It
only knows about a URDF, joint names and target poses.  Public quaternions use
Isaac Gym's order: [x, y, z, w].
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


DEFAULT_Z1_URDF = Path(__file__).resolve().parents[1] / "data" / "asset" / "z1" / "urdf" / "z1_arm.urdf"
DEFAULT_Z1_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT_Z1_EE_LINK = "ee_gripper_link"


def normalize_quat_xyzw(quat: np.ndarray | list[float]) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def quat_conjugate_xyzw(quat: np.ndarray | list[float]) -> np.ndarray:
    quat = normalize_quat_xyzw(quat)
    return np.asarray([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def quat_multiply_xyzw(lhs: np.ndarray | list[float], rhs: np.ndarray | list[float]) -> np.ndarray:
    x1, y1, z1, w1 = normalize_quat_xyzw(lhs)
    x2, y2, z2, w2 = normalize_quat_xyzw(rhs)
    return normalize_quat_xyzw(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_to_rot_xyzw(quat: np.ndarray | list[float]) -> np.ndarray:
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


def rot_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 0.0)) * 2.0
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
            w = (rot[2, 1] - rot[1, 2]) / s
        elif idx == 1:
            s = math.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 0.0)) * 2.0
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
            w = (rot[0, 2] - rot[2, 0]) / s
        else:
            s = math.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 0.0)) * 2.0
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
            w = (rot[1, 0] - rot[0, 1]) / s
    return normalize_quat_xyzw([x, y, z, w])


def orientation_error_xyzw(desired: np.ndarray | list[float], current: np.ndarray | list[float]) -> np.ndarray:
    delta = quat_multiply_xyzw(desired, quat_conjugate_xyzw(current))
    sign = 1.0 if delta[3] >= 0.0 else -1.0
    return delta[:3] * sign


class Z1PinocchioIK:
    """Pinocchio-based Z1 FK/IK in the URDF root frame."""

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_Z1_URDF,
        joint_names: tuple[str, ...] | list[str] = DEFAULT_Z1_JOINT_NAMES,
        ee_link: str = DEFAULT_Z1_EE_LINK,
    ):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(f"Pinocchio import failed: {exc}") from exc

        self.pin = pin
        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        if not Path(self.urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        if not self.model.existFrame(ee_link):
            frames = ", ".join(frame.name for frame in self.model.frames)
            raise ValueError(f"EE frame {ee_link!r} not in URDF. Frames: {frames}")

        self.joint_names = list(joint_names)
        self.ee_link = str(ee_link)
        self.ee_frame_id = int(self.model.getFrameId(self.ee_link))
        self.joint_q_index = self._joint_q_index()
        missing = [name for name in self.joint_names if name not in self.joint_q_index]
        if missing:
            raise ValueError(f"URDF is missing controlled joints: {missing}")

        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        self.neutral = np.asarray(pin.neutral(self.model), dtype=np.float64)

    def _joint_q_index(self) -> dict[str, int]:
        out = {}
        for joint_id, joint_name in enumerate(self.model.names):
            if joint_id == 0:
                continue
            joint = self.model.joints[joint_id]
            if int(joint.nq) == 1:
                out[str(joint_name)] = int(joint.idx_q)
        return out

    def pack_q(self, joints: np.ndarray | list[float]) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64).reshape(-1)
        q = self.neutral.copy()
        for i, name in enumerate(self.joint_names):
            if i < joints.shape[0]:
                q[self.joint_q_index[name]] = joints[i]
        return self.clip_q(q)

    def unpack_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        return np.asarray([q[self.joint_q_index[name]] for name in self.joint_names], dtype=np.float64)

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(self.model.nq).copy()
        finite_lower = np.isfinite(self.lower)
        finite_upper = np.isfinite(self.upper)
        q[finite_lower] = np.maximum(q[finite_lower], self.lower[finite_lower])
        q[finite_upper] = np.minimum(q[finite_upper], self.upper[finite_upper])
        return q

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray([self.lower[self.joint_q_index[name]] for name in self.joint_names], dtype=np.float64)
        upper = np.asarray([self.upper[self.joint_q_index[name]] for name in self.joint_names], dtype=np.float64)
        return lower, upper

    def fk(self, joints: np.ndarray | list[float]) -> np.ndarray:
        """Return [x, y, z, qx, qy, qz, qw] in the URDF root frame."""
        q = self.pack_q(joints)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.ee_frame_id]
        return np.concatenate([np.asarray(pose.translation, dtype=np.float64), rot_to_quat_xyzw(pose.rotation)])

    def ik(
        self,
        target_pose: np.ndarray | list[float],
        seed_joints: np.ndarray | list[float] | None = None,
        position_only: bool = False,
        max_iter: int = 100,
        max_restarts: int = 12,
        dt: float = 0.4,
        damping: float = 1.0e-4,
        threshold: float = 1.0e-4,
        rotation_weight: float = 0.5,
        continuity_weight: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | None:
        """Solve IK for target [x, y, z, qx, qy, qz, qw].

        Returns the controlled joint vector in ``joint_names`` order, or None if
        the damped least-squares iteration does not converge.
        """
        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        if target_pose.shape[0] not in (3, 7):
            raise ValueError(f"target_pose must have 3 or 7 values, got {target_pose.shape[0]}")

        target_pos = target_pose[:3]
        if position_only or target_pose.shape[0] == 3:
            target_rot = np.eye(3, dtype=np.float64)
            weights = np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            target_rot = quat_to_rot_xyzw(target_pose[3:7])
            weights = np.asarray([1.0, 1.0, 1.0, rotation_weight, rotation_weight, rotation_weight], dtype=np.float64)
        target_se3 = self.pin.SE3(target_rot, target_pos)
        weight_matrix = np.diag(weights)
        active = np.abs(weights) > 0.0

        seeds = self._ik_seed_list(seed_joints, max_restarts=max_restarts, rng=rng)
        seed_control = self.unpack_q(seeds[0])
        continuity_weight = max(0.0, float(continuity_weight))
        loose_threshold = max(float(threshold) * 10.0, 1.0e-3)
        best = None
        best_norm = math.inf
        best_score = math.inf
        for seed in seeds:
            q_sol, err_norm, converged = self._ik_core(
                target_se3,
                weight_matrix,
                active,
                seed,
                max_iter=max_iter,
                dt=dt,
                damping=damping,
                threshold=threshold,
            )
            if converged and continuity_weight <= 0.0:
                return self.unpack_q(q_sol)
            if converged or err_norm < loose_threshold:
                q_control = self.unpack_q(q_sol)
                continuity_cost = float(np.linalg.norm(q_control - seed_control))
                score = float(err_norm) + continuity_weight * continuity_cost
                if score < best_score:
                    best_score = score
                    best_norm = err_norm
                    best = q_sol
            if err_norm < best_norm:
                best_norm = err_norm
                best = q_sol
        if best is not None and best_norm < loose_threshold:
            return self.unpack_q(best)
        return None

    def _ik_seed_list(self, seed_joints, max_restarts, rng):
        lower, upper = self.joint_limits()
        middle = np.where(np.isfinite(lower) & np.isfinite(upper), 0.5 * (lower + upper), 0.0)
        seeds = []
        if seed_joints is not None:
            seeds.append(np.asarray(seed_joints, dtype=np.float64).reshape(-1)[: len(self.joint_names)])
        seeds.append(np.zeros(len(self.joint_names), dtype=np.float64))
        seeds.append(middle)
        rng = np.random.default_rng(0) if rng is None else rng
        for _ in range(max(0, int(max_restarts) - len(seeds))):
            sample = middle.copy()
            finite = np.isfinite(lower) & np.isfinite(upper)
            sample[finite] = lower[finite] + rng.random(np.count_nonzero(finite)) * (upper[finite] - lower[finite])
            seeds.append(sample)
        return [self.pack_q(seed) for seed in seeds[: max(1, int(max_restarts))]]

    def _ik_core(self, target_se3, weight_matrix, active, q_seed, max_iter, dt, damping, threshold):
        q = self.clip_q(q_seed)
        best_q = q.copy()
        best_norm = math.inf
        for _ in range(max(1, int(max_iter))):
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.ee_frame_id]

            pos_err = target_se3.translation - current.translation
            rot_err_local = self.pin.log3(current.rotation.T @ target_se3.rotation)
            rot_err_world = current.rotation @ rot_err_local
            err = np.concatenate([pos_err, rot_err_world])
            weighted_err = weight_matrix @ err
            err_norm = float(np.linalg.norm(weighted_err[active]))
            if err_norm < best_norm:
                best_norm = err_norm
                best_q = q.copy()
            if err_norm < float(threshold):
                return q, err_norm, True

            jac = self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.ee_frame_id,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            weighted_jac = weight_matrix @ jac
            lhs = weighted_jac @ weighted_jac.T + (float(damping) ** 2) * np.eye(6)
            try:
                velocity = weighted_jac.T @ np.linalg.solve(lhs, weighted_err)
            except np.linalg.LinAlgError:
                velocity = weighted_jac.T @ np.linalg.lstsq(lhs, weighted_err, rcond=None)[0]
            q_next = self.clip_q(self.pin.integrate(self.model, q, velocity * float(dt)))
            if float(np.linalg.norm(q_next - q)) < 1.0e-10:
                break
            q = q_next
        return best_q, best_norm, False
