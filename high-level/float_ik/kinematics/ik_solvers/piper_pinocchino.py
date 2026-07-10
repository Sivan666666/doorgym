#!/usr/bin/env python3
"""Pinocchio IK benchmark for Piper.

This is intentionally self-contained so it can run in the IsaacGym/b1z1 conda
env without depending on deformable_bench's BaseKinematics/loguru/transforms3d.

Pose format follows the copied pinocchio_kinematics.py:
    [x, y, z, qw, qx, qy, qz]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


DEFAULT_PIPER_URDF = Path("/home/sivan/cloth/deformable_bench/assets/robot/piper/urdf/piper_with_gripper.urdf")
DEFAULT_PIPER_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT_BASE_LINK = "base_link"
DEFAULT_EE_LINK = "tcp_link"


def normalize_quat_wxyz(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_to_rot_wxyz(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(quat)
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


def rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rot[2, 1] - rot[1, 2]) / scale
        y = (rot[0, 2] - rot[2, 0]) / scale
        z = (rot[1, 0] - rot[0, 1]) / scale
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            scale = np.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 0.0)) * 2.0
            x = 0.25 * scale
            y = (rot[0, 1] + rot[1, 0]) / scale
            z = (rot[0, 2] + rot[2, 0]) / scale
            w = (rot[2, 1] - rot[1, 2]) / scale
        elif idx == 1:
            scale = np.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 0.0)) * 2.0
            x = (rot[0, 1] + rot[1, 0]) / scale
            y = 0.25 * scale
            z = (rot[1, 2] + rot[2, 1]) / scale
            w = (rot[0, 2] - rot[2, 0]) / scale
        else:
            scale = np.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 0.0)) * 2.0
            x = (rot[0, 2] + rot[2, 0]) / scale
            y = (rot[1, 2] + rot[2, 1]) / scale
            z = 0.25 * scale
            w = (rot[1, 0] - rot[0, 1]) / scale
    return normalize_quat_wxyz([w, x, y, z])


class PiperPinocchioIK:
    def __init__(
        self,
        urdf_path: Union[str, Path] = DEFAULT_PIPER_URDF,
        base_link: str = DEFAULT_BASE_LINK,
        ee_link: str = DEFAULT_EE_LINK,
        joint_names: Tuple[str, ...] = DEFAULT_PIPER_JOINT_NAMES,
    ):
        import pinocchio as pin

        self.pin = pin
        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        if not Path(self.urdf_path).exists():
            raise FileNotFoundError(f"Piper URDF not found: {self.urdf_path}")

        self.base_link = str(base_link)
        self.ee_link = str(ee_link)
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        if not self.model.existFrame(self.ee_link):
            frames = ", ".join(frame.name for frame in self.model.frames)
            raise ValueError(f"Frame {self.ee_link!r} not in URDF. Frames: {frames}")
        self.ee_frame_id = int(self.model.getFrameId(self.ee_link))
        self.base_frame_id = int(self.model.getFrameId(self.base_link)) if self.model.existFrame(self.base_link) else 0
        self.joint_names = list(joint_names)
        self.joint_q_index = self._joint_q_index()
        missing = [name for name in self.joint_names if name not in self.joint_q_index]
        if missing:
            raise ValueError(f"URDF is missing controlled joints: {missing}")
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        self.neutral = np.asarray(pin.neutral(self.model), dtype=np.float64)

    def _joint_q_index(self) -> Dict[str, int]:
        out = {}
        for joint_id, joint_name in enumerate(self.model.names):
            if joint_id == 0:
                continue
            joint = self.model.joints[joint_id]
            if int(joint.nq) == 1:
                out[str(joint_name)] = int(joint.idx_q)
        return out

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(self.model.nq).copy()
        finite_lower = np.isfinite(self.lower)
        finite_upper = np.isfinite(self.upper)
        q[finite_lower] = np.maximum(q[finite_lower], self.lower[finite_lower])
        q[finite_upper] = np.minimum(q[finite_upper], self.upper[finite_upper])
        return q

    def pack_q(self, joints: Union[List[float], np.ndarray]) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64).reshape(-1)
        if joints.shape[0] == self.model.nq:
            return self.clip_q(joints)
        q = self.neutral.copy()
        for i, name in enumerate(self.joint_names):
            if i < joints.shape[0]:
                q[self.joint_q_index[name]] = joints[i]
        return self.clip_q(q)

    def unpack_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        return np.asarray([q[self.joint_q_index[name]] for name in self.joint_names], dtype=np.float64)

    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.asarray([self.lower[self.joint_q_index[name]] for name in self.joint_names], dtype=np.float64)
        upper = np.asarray([self.upper[self.joint_q_index[name]] for name in self.joint_names], dtype=np.float64)
        return lower, upper

    def solve_fk(self, joint_angles: Union[List[float], np.ndarray], ee_link: Optional[str] = None) -> np.ndarray:
        q = self.pack_q(joint_angles)
        fid = self.ee_frame_id
        if ee_link is not None and ee_link != self.ee_link:
            fid = int(self.model.getFrameId(ee_link))
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[fid]
        return np.concatenate([np.asarray(pose.translation, dtype=np.float64), rot_to_quat_wxyz(pose.rotation)])

    def solve_ik(
        self,
        target_pose: Union[List[float], np.ndarray],
        seed_joints: Optional[Union[List[float], np.ndarray]] = None,
        threshold: float = 1.0e-4,
        pose_constraint: Optional[Union[List[float], np.ndarray, Dict[str, Union[List[float], np.ndarray]]]] = None,
        constraint_frame: str = "world",
        ee_link: Optional[str] = None,
        max_restarts: int = 30,
        strict_seed: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
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
        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(7)
        fid = int(self.model.getFrameId(ee_link)) if ee_link else self.ee_frame_id

        seed_strategies = []
        if seed_joints is not None:
            seed_strategies.append(("user_seed", self.pack_q(seed_joints)))
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

        for _, q_init in seed_strategies:
            q_sol = self._solve_ik_core(
                q_init=q_init,
                target_pose=target_pose,
                mask_world=mask_world,
                mask_local=mask_local,
                max_iter=max_iter,
                dt=dt,
                damp=damp,
                threshold=threshold,
                fid=fid,
            )
            if q_sol is not None:
                return self.unpack_q(q_sol)
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
        pos_des = target_pose[:3]
        rot_des = quat_to_rot_wxyz(target_pose[3:7])
        o_m_des = self.pin.SE3(rot_des, pos_des)
        w_world = np.diag(mask_world) if mask_world is not None else None
        w_local = np.diag(mask_local) if mask_local is not None else None

        for i in range(max(1, int(max_iter))):
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            o_m_tool = self.data.oMf[fid]

            errors = []
            jacobians = []
            if w_local is not None:
                d_m_local = o_m_tool.actInv(o_m_des)
                err_local = w_local @ self.pin.log6(d_m_local).vector
                jac_local = self.pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    q,
                    fid,
                    self.pin.ReferenceFrame.LOCAL,
                )
                errors.append(err_local)
                jacobians.append(w_local @ jac_local)

            if w_world is not None:
                err_pos_world = pos_des - o_m_tool.translation
                rot_err_local = self.pin.log3(o_m_tool.rotation.T @ rot_des)
                err_rot_world = o_m_tool.rotation @ rot_err_local
                err_world = w_world @ np.concatenate([err_pos_world, err_rot_world])
                jac_world = self.pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    q,
                    fid,
                    self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                errors.append(err_world)
                jacobians.append(w_world @ jac_world)

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
            q_next = self.pin.integrate(self.model, q, dq * float(dt))
            q_next = np.clip(q_next, self.lower, self.upper)
            if float(np.linalg.norm(q_next - q)) < 1.0e-6 and i < max_iter - 1:
                dt *= 0.5
            q = q_next
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_PIPER_URDF))
    parser.add_argument("--base_link", type=str, default=DEFAULT_BASE_LINK)
    parser.add_argument("--ee_link", type=str, default=DEFAULT_EE_LINK)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_delta", type=float, default=0.25)
    parser.add_argument("--global_targets", action="store_true")
    parser.add_argument("--seed_mode", choices=("chain", "target", "middle", "zero"), default="chain")
    parser.add_argument("--rot_weight", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=80)
    parser.add_argument("--max_restarts", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--damping", type=float, default=1.0e-4)
    parser.add_argument("--threshold", type=float, default=1.0e-4)
    return parser.parse_args()


def summarize_ms(name: str, values_ms: List[float]) -> str:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.size == 0:
        return f"{name}: no samples"
    return (
        f"{name}: mean={np.mean(values):.3f} ms "
        f"p50={np.percentile(values, 50):.3f} ms "
        f"p90={np.percentile(values, 90):.3f} ms "
        f"p95={np.percentile(values, 95):.3f} ms "
        f"p99={np.percentile(values, 99):.3f} ms "
        f"max={np.max(values):.3f} ms"
    )


def pose_error(ik: PiperPinocchioIK, target_pose: np.ndarray, joints: np.ndarray) -> Tuple[float, float]:
    pose = ik.solve_fk(joints)
    pos_err = float(np.linalg.norm(target_pose[:3] - pose[:3]))
    rot_target = quat_to_rot_wxyz(target_pose[3:7])
    rot_current = quat_to_rot_wxyz(pose[3:7])
    rot_err = float(np.linalg.norm(ik.pin.log3(rot_current.T @ rot_target)))
    return pos_err, rot_err


def sample_benchmark_items(ik: PiperPinocchioIK, args, rng: np.random.Generator, count: int):
    lower, upper = ik.joint_limits()
    margin = np.minimum(0.05, np.maximum(upper - lower, 0.0) * 0.02)
    middle = np.clip(0.5 * (lower + upper), lower + margin, upper - margin)
    zero = np.clip(np.zeros_like(middle), lower + margin, upper - margin)
    chain_seed = middle.copy()
    items = []
    for _ in range(count):
        if args.global_targets:
            target_q = lower + margin + rng.random(middle.shape[0]) * (upper - lower - 2.0 * margin)
        else:
            delta = rng.uniform(-float(args.target_delta), float(args.target_delta), size=middle.shape[0])
            target_q = np.clip(chain_seed + delta, lower + margin, upper - margin)

        if args.seed_mode == "target":
            seed_q = target_q.copy()
        elif args.seed_mode == "middle":
            seed_q = middle.copy()
        elif args.seed_mode == "zero":
            seed_q = zero.copy()
        else:
            seed_q = chain_seed.copy()

        items.append((seed_q, ik.solve_fk(target_q), target_q))
        chain_seed = target_q
    return items


def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    ik = PiperPinocchioIK(args.urdf, base_link=args.base_link, ee_link=args.ee_link)
    items = sample_benchmark_items(ik, args, rng, int(args.samples) + int(args.warmup))
    pose_constraint = [1.0, 1.0, 1.0, float(args.rot_weight), float(args.rot_weight), float(args.rot_weight)]

    times_ms = []
    success_times_ms = []
    failure_times_ms = []
    pos_errors = []
    rot_errors = []
    failures = 0
    first_failure = None
    for i, (seed_q, target_pose, _) in enumerate(items):
        start_ns = time.perf_counter_ns()
        solution = ik.solve_ik(
            target_pose,
            seed_joints=seed_q,
            pose_constraint=pose_constraint,
            max_restarts=int(args.max_restarts),
            max_iter=int(args.max_iter),
            dt=float(args.dt),
            damp=float(args.damping),
            threshold=float(args.threshold),
        )
        elapsed_ms = (time.perf_counter_ns() - start_ns) * 1.0e-6
        if i < int(args.warmup):
            continue
        times_ms.append(elapsed_ms)
        if solution is None:
            failures += 1
            failure_times_ms.append(elapsed_ms)
            if first_failure is None:
                first_failure = i - int(args.warmup)
            continue
        success_times_ms.append(elapsed_ms)
        pos_err, rot_err = pose_error(ik, target_pose, solution)
        pos_errors.append(pos_err)
        rot_errors.append(rot_err)

    total_s = float(np.sum(times_ms)) / 1000.0
    success_only_s = float(np.sum(success_times_ms)) / 1000.0
    successful = len(success_times_ms)
    print("Piper Pinocchio IK benchmark")
    print(f"urdf={Path(args.urdf).expanduser().resolve()}")
    print(f"base_link={args.base_link} ee_link={args.ee_link}")
    print(
        f"samples={int(args.samples)} warmup={int(args.warmup)} "
        f"global_targets={bool(args.global_targets)} target_delta={float(args.target_delta):.3f} "
        f"seed_mode={args.seed_mode}"
    )
    print(
        f"constraint=[1,1,1,{float(args.rot_weight):.3f},{float(args.rot_weight):.3f},{float(args.rot_weight):.3f}] "
        f"max_iter={int(args.max_iter)} max_restarts={int(args.max_restarts)} "
        f"dt={float(args.dt):.3f} damping={float(args.damping):.1e} threshold={float(args.threshold):.1e}"
    )
    print(f"success={successful} failures={failures} first_failure={first_failure}")
    print(summarize_ms("latency_all", times_ms))
    print(summarize_ms("latency_success_only", success_times_ms))
    if failure_times_ms:
        print(summarize_ms("latency_failure_only", failure_times_ms))
    if total_s > 0.0:
        print(f"throughput_all_calls={len(times_ms) / total_s:.1f} calls/s")
        print(f"throughput_successful_wall={successful / total_s:.1f} successful solves/s including failure time")
    if success_only_s > 0.0:
        print(f"throughput_success_only={successful / success_only_s:.1f} successful solves/s excluding failure time")
    if pos_errors:
        print(
            "pos_err(m): "
            f"mean={np.mean(pos_errors):.6f} p95={np.percentile(pos_errors, 95):.6f} "
            f"max={np.max(pos_errors):.6f}"
        )
        print(
            "rot_err: "
            f"mean={np.mean(rot_errors):.6f} p95={np.percentile(rot_errors, 95):.6f} "
            f"max={np.max(rot_errors):.6f}"
        )


if __name__ == "__main__":
    main()
