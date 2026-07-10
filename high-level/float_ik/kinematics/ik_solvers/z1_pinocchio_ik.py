#!/usr/bin/env python3
"""Standalone Pinocchio FK/IK for the Unitree Z1 arm.

The core ``solve_fk``/``solve_ik`` API follows
``high-level/float_ik/kinematics/pinocchio_kinematics.py``:

- pose format: ``[x, y, z, qw, qx, qy, qz]``
- default full pose constraint
- world/local/mixed pose masks
- DLS update ``(J.T @ J + damp * I) dq = J.T @ err``

For the Isaac Gym comparison and visualizer scripts, ``fk``/``ik`` keep the
previous public convention: ``[x, y, z, qx, qy, qz, qw]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


DEFAULT_Z1_URDF = Path(__file__).resolve().parents[3] / "data" / "asset" / "z1" / "urdf" / "z1_arm.urdf"
DEFAULT_Z1_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT_Z1_BASE_LINK = "base"
DEFAULT_Z1_EE_LINK = "ee_gripper_link"


def normalize_quat_xyzw(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def normalize_quat_wxyz(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_xyzw_to_wxyz(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat)
    return np.asarray([w, x, y, z], dtype=np.float64)


def quat_wxyz_to_xyzw(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(quat)
    return np.asarray([x, y, z, w], dtype=np.float64)


def quat_conjugate_xyzw(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    quat = normalize_quat_xyzw(quat)
    return np.asarray([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def quat_multiply_xyzw(lhs: Union[np.ndarray, List[float]], rhs: Union[np.ndarray, List[float]]) -> np.ndarray:
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


def quat_to_rot_xyzw(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
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


def quat_to_rot_wxyz(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    return quat_to_rot_xyzw(quat_wxyz_to_xyzw(quat))


def rot_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 0.0)) * 2.0
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
            w = (rot[2, 1] - rot[1, 2]) / s
        elif idx == 1:
            s = np.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 0.0)) * 2.0
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
            w = (rot[0, 2] - rot[2, 0]) / s
        else:
            s = np.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 0.0)) * 2.0
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
            w = (rot[1, 0] - rot[0, 1]) / s
    return normalize_quat_xyzw([x, y, z, w])


def rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    return quat_xyzw_to_wxyz(rot_to_quat_xyzw(rot))


def orientation_error_xyzw(desired: Union[np.ndarray, List[float]], current: Union[np.ndarray, List[float]]) -> np.ndarray:
    delta = quat_multiply_xyzw(desired, quat_conjugate_xyzw(current))
    sign = 1.0 if delta[3] >= 0.0 else -1.0
    return delta[:3] * sign


class Z1PinocchioIK:
    """Pinocchio-based Z1 FK/IK in the URDF root frame."""

    def __init__(
        self,
        urdf_path: Union[str, Path] = DEFAULT_Z1_URDF,
        joint_names: Union[Tuple[str, ...], List[str]] = DEFAULT_Z1_JOINT_NAMES,
        ee_link: str = DEFAULT_Z1_EE_LINK,
        base_link: str = DEFAULT_Z1_BASE_LINK,
        base_height: float = 0.0,
    ):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(f"Pinocchio import failed: {exc}") from exc

        self.pin = pin
        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        if not Path(self.urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.base_link = str(base_link)
        self.ee_link = str(ee_link)
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        if not self.model.existFrame(self.ee_link):
            frames = ", ".join(frame.name for frame in self.model.frames)
            raise ValueError(f"Frame {self.ee_link!r} does not exist in URDF. Frames: {frames}")

        self.ee_frame_id = int(self.model.getFrameId(self.ee_link))
        self.base_frame_id = int(self.model.getFrameId(self.base_link)) if self.model.existFrame(self.base_link) else 0
        self.num_joints = int(self.model.nq)
        self.joint_names = list(joint_names)
        self.arm_joints = len(self.joint_names)
        self.joint_q_index = self._joint_q_index()
        missing = [name for name in self.joint_names if name not in self.joint_q_index]
        if missing:
            raise ValueError(f"URDF is missing controlled joints: {missing}")

        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        self.neutral = np.asarray(pin.neutral(self.model), dtype=np.float64)
        self.ground_z_threshold = -float(base_height)
        self.ground_check_enabled = False
        self._ground_check_frame_ids = []
        for frame_id in range(self.model.nframes):
            frame = self.model.frames[frame_id]
            if frame.type == pin.FrameType.BODY and frame.name not in ("dummy_link", self.base_link):
                self._ground_check_frame_ids.append(frame_id)

    def _joint_q_index(self) -> Dict[str, int]:
        out = {}
        for joint_id, joint_name in enumerate(self.model.names):
            if joint_id == 0:
                continue
            joint = self.model.joints[joint_id]
            if int(joint.nq) == 1:
                out[str(joint_name)] = int(joint.idx_q)
        return out

    def _ensure_numpy(self, data) -> np.ndarray:
        if isinstance(data, list) or isinstance(data, tuple):
            return np.asarray(data, dtype=np.float64)
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            return data.cpu().numpy().astype(np.float64)
        if isinstance(data, np.ndarray):
            return data.astype(np.float64, copy=False)
        return np.asarray(data, dtype=np.float64)

    def _full_q(self, joints: Union[List[float], np.ndarray]) -> np.ndarray:
        joints = self._ensure_numpy(joints).reshape(-1)
        if joints.shape[0] == self.model.nq:
            return self.clip_q(joints)
        return self.pack_q(joints)

    def pack_q(self, joints: Union[List[float], np.ndarray]) -> np.ndarray:
        joints = self._ensure_numpy(joints).reshape(-1)
        q = self.neutral.copy()
        for i, name in enumerate(self.joint_names):
            if i < joints.shape[0]:
                q[self.joint_q_index[name]] = joints[i]
        return self.clip_q(q)

    def unpack_q(self, q: np.ndarray) -> np.ndarray:
        q = self._ensure_numpy(q).reshape(-1)
        return np.asarray([q[self.joint_q_index[name]] for name in self.joint_names], dtype=np.float64)

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        q = self._ensure_numpy(q).reshape(self.model.nq).copy()
        finite_lower = np.isfinite(self.lower)
        finite_upper = np.isfinite(self.upper)
        q[finite_lower] = np.maximum(q[finite_lower], self.lower[finite_lower])
        q[finite_upper] = np.minimum(q[finite_upper], self.upper[finite_upper])
        return q

    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.asarray([self.lower[self.joint_q_index[name]] for name in self.joint_names], dtype=np.float64)
        upper = np.asarray([self.upper[self.joint_q_index[name]] for name in self.joint_names], dtype=np.float64)
        return lower, upper

    def solve_fk(self, joint_angles: Union[List[float], np.ndarray], ee_link: Optional[str] = None) -> np.ndarray:
        """Forward kinematics returning [x, y, z, qw, qx, qy, qz]."""
        q = self._full_q(joint_angles)
        fid = self.ee_frame_id
        if ee_link is not None and ee_link != self.ee_link:
            fid = int(self.model.getFrameId(ee_link))

        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[fid]
        return np.concatenate([np.asarray(pose.translation, dtype=np.float64), rot_to_quat_wxyz(pose.rotation)])

    def fk(self, joints: Union[List[float], np.ndarray]) -> np.ndarray:
        """Forward kinematics returning [x, y, z, qx, qy, qz, qw]."""
        pose_wxyz = self.solve_fk(joints)
        return np.concatenate([pose_wxyz[:3], quat_wxyz_to_xyzw(pose_wxyz[3:7])])

    def _check_ground_penetration(self, q: np.ndarray, threshold: Optional[float] = None) -> Tuple[bool, str, float]:
        if threshold is None:
            threshold = self.ground_z_threshold
        q = self._full_q(q)

        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)

        min_z = float("inf")
        min_frame = ""
        for fid in self._ground_check_frame_ids:
            z = float(self.data.oMf[fid].translation[2])
            if z < min_z:
                min_z = z
                min_frame = self.model.frames[fid].name
        return min_z < float(threshold), min_frame, min_z

    def solve_ik(
        self,
        target_pose: Union[np.ndarray, List[float]],
        seed_joints: Optional[np.ndarray] = None,
        threshold: float = 1.0e-4,
        pose_constraint: Optional[Union[np.ndarray, List[float], Dict[str, Union[np.ndarray, List[float]]]]] = None,
        constraint_frame: str = "world",
        ee_link: Optional[str] = None,
        max_restarts: int = 30,
        strict_seed: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """IK solver matching kinematics/pinocchio_kinematics.py.

        Args:
            target_pose: [x, y, z, qw, qx, qy, qz].
            pose_constraint: [x,y,z, rx,ry,rz] weights, or
                {"local": [...], "world": [...]}.
        """
        mask_local = None
        mask_world = None

        if isinstance(pose_constraint, dict):
            mask_local = pose_constraint.get("local")
            mask_world = pose_constraint.get("world")
            if mask_local is not None:
                mask_local = np.asarray(mask_local, dtype=np.float64)
            if mask_world is not None:
                mask_world = np.asarray(mask_world, dtype=np.float64)
        else:
            constraint_arr = np.ones(6, dtype=np.float64) if pose_constraint is None else np.asarray(
                pose_constraint, dtype=np.float64
            )
            if constraint_arr.shape == (3,):
                constraint_arr = np.concatenate([constraint_arr, np.zeros(3, dtype=np.float64)])
            if constraint_frame == "local":
                mask_local = constraint_arr
            else:
                mask_world = constraint_arr

        max_iter = int(kwargs.get("max_iter", 1000))
        dt = float(kwargs.get("dt", 0.05))
        damp = float(kwargs.get("damp", kwargs.get("damping", 1.0e-6)))
        threshold = float(kwargs.get("threshold", threshold))
        pose_arr = self._ensure_numpy(target_pose).reshape(-1)
        if pose_arr.shape[0] != 7:
            raise ValueError(f"target_pose must be [x,y,z,qw,qx,qy,qz], got shape {pose_arr.shape}")
        fid = int(self.model.getFrameId(ee_link)) if ee_link else self.ee_frame_id

        seed_strategies = []
        if seed_joints is not None:
            seed_joints_array = self._full_q(seed_joints)
            seed_strategies.append(("user_seed", seed_joints_array.copy()))
            if strict_seed:
                max_restarts = 1

        if not strict_seed:
            seed_strategies.append(("zero", self.clip_q(np.zeros(self.model.nq, dtype=np.float64))))
            middle = np.where(
                np.isfinite(self.lower) & np.isfinite(self.upper),
                0.5 * (self.lower + self.upper),
                self.neutral,
            )
            seed_strategies.append(("middle", self.clip_q(middle)))
            for i in range(max(0, int(max_restarts) - len(seed_strategies))):
                seed_strategies.append((f"random_{i}", self.clip_q(self.pin.randomConfiguration(self.model))))

        seed_strategies = seed_strategies[: max(1, int(max_restarts))]
        ground_check = bool(kwargs.get("ground_check", self.ground_check_enabled))
        ground_z = float(kwargs.get("ground_z_threshold", self.ground_z_threshold))
        verbose = bool(kwargs.get("verbose", False))

        best_penetrating_sol = None
        rejected_count = 0
        for restart_idx, (strategy_name, q_init) in enumerate(seed_strategies):
            q_sol = self._solve_ik_core(
                q_init=q_init.copy(),
                target_pose=pose_arr,
                mask_local=mask_local,
                mask_world=mask_world,
                max_iter=max_iter,
                dt=dt,
                damp=damp,
                threshold=threshold,
                fid=fid,
            )
            if q_sol is None:
                continue

            if ground_check:
                penetrates, pen_frame, pen_z = self._check_ground_penetration(q_sol, ground_z)
                if penetrates:
                    rejected_count += 1
                    if best_penetrating_sol is None or pen_z > best_penetrating_sol[1]:
                        best_penetrating_sol = (q_sol.copy(), pen_z, pen_frame)
                    if verbose:
                        print(f"IK {strategy_name!r} rejected: {pen_frame} z={pen_z:.4f} < {ground_z}")
                    continue

            if verbose:
                print(
                    f"IK solved using strategy {strategy_name!r} "
                    f"(attempt {restart_idx}, {rejected_count} ground-rejected)"
                )
            return self.unpack_q(q_sol)

        if best_penetrating_sol is not None:
            sol, pen_z, pen_frame = best_penetrating_sol
            print(
                f"IK: all {rejected_count} converged solutions penetrate ground. "
                f"Returning least-penetrating: {pen_frame} z={pen_z:.4f}"
            )
            return self.unpack_q(sol)

        return None

    def _solve_ik_core(
        self,
        q_init: np.ndarray,
        target_pose: np.ndarray,
        mask_world: Optional[np.ndarray],
        mask_local: Optional[np.ndarray],
        max_iter: int,
        dt: float,
        damp: float,
        threshold: float,
        fid: int,
    ) -> Optional[np.ndarray]:
        q = self.clip_q(q_init)
        q_min = self.lower
        q_max = self.upper

        pos_des = target_pose[:3]
        quat_wxyz = target_pose[3:7]
        r_des = quat_to_rot_wxyz(quat_wxyz)
        o_m_des = self.pin.SE3(r_des, pos_des)

        w_world = np.diag(mask_world) if mask_world is not None else None
        w_local = np.diag(mask_local) if mask_local is not None else None

        for i in range(max(1, int(max_iter))):
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            o_m_tool = self.data.oMf[fid]

            jacobians = []
            errors = []

            if w_local is not None:
                d_m_local = o_m_tool.actInv(o_m_des)
                err_local_full = self.pin.log6(d_m_local).vector
                err_local = w_local @ err_local_full
                jac_local_full = self.pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    q,
                    fid,
                    self.pin.ReferenceFrame.LOCAL,
                )
                jacobians.append(w_local @ jac_local_full)
                errors.append(err_local)

            if w_world is not None:
                err_pos_world = pos_des - o_m_tool.translation
                rot_err_local = self.pin.log3(o_m_tool.rotation.T @ r_des)
                err_rot_world = o_m_tool.rotation @ rot_err_local
                err_world_full = np.concatenate([err_pos_world, err_rot_world])
                err_world = w_world @ err_world_full
                jac_world_full = self.pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    q,
                    fid,
                    self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                jacobians.append(w_world @ jac_world_full)
                errors.append(err_world)

            if not errors:
                break

            err_stack = np.concatenate(errors)
            jac_stack = np.vstack(jacobians)
            if float(np.linalg.norm(err_stack)) < float(threshold):
                return q

            hessian = jac_stack.T @ jac_stack + float(damp) * np.eye(self.model.nv)
            gradient = jac_stack.T @ err_stack
            try:
                dq = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

            q_new = self.pin.integrate(self.model, q, dq * float(dt))
            q_new = np.clip(q_new, q_min, q_max)
            if float(np.linalg.norm(q_new - q)) < 1.0e-6 and i < max_iter - 1:
                dt *= 0.5
            q = q_new

        return None

    def ik(
        self,
        target_pose: Union[np.ndarray, List[float]],
        seed_joints: Optional[Union[np.ndarray, List[float]]] = None,
        position_only: bool = False,
        max_iter: int = 1000,
        max_restarts: int = 30,
        dt: float = 0.05,
        damping: float = 1.0e-6,
        threshold: float = 1.0e-4,
        rotation_weight: float = 1.0,
        rng=None,
        strict_seed: bool = False,
        constraint_frame: str = "world",
        pose_constraint: Optional[Union[np.ndarray, List[float], Dict[str, Union[np.ndarray, List[float]]]]] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Compatibility IK wrapper using [x, y, z, qx, qy, qz, qw] targets."""
        del rng
        target_pose = self._ensure_numpy(target_pose).reshape(-1)
        if target_pose.shape[0] == 3:
            target_wxyz = np.concatenate([target_pose[:3], self.solve_fk(seed_joints if seed_joints is not None else np.zeros(6))[3:7]])
            pose_constraint = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0] if pose_constraint is None else pose_constraint
        elif target_pose.shape[0] == 7:
            target_wxyz = np.concatenate([target_pose[:3], quat_xyzw_to_wxyz(target_pose[3:7])])
            if pose_constraint is None:
                if position_only:
                    pose_constraint = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
                else:
                    pose_constraint = [
                        1.0,
                        1.0,
                        1.0,
                        float(rotation_weight),
                        float(rotation_weight),
                        float(rotation_weight),
                    ]
        else:
            raise ValueError(f"target_pose must have 3 or 7 values, got {target_pose.shape[0]}")

        return self.solve_ik(
            target_wxyz,
            seed_joints=None if seed_joints is None else self._ensure_numpy(seed_joints),
            threshold=float(threshold),
            pose_constraint=pose_constraint,
            constraint_frame=constraint_frame,
            max_restarts=int(max_restarts),
            strict_seed=bool(strict_seed),
            max_iter=int(max_iter),
            dt=float(dt),
            damp=float(damping),
            **kwargs,
        )
