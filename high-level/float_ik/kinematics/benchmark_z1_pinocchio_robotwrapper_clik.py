#!/usr/bin/env python3
"""Benchmark a RobotWrapper-based Pinocchio CLIK solver for Z1.

Pinocchio 3.x does not expose a high-level ``ikine``/``inverseKinematics`` API
like RTB or TRAC-IK.  The closest higher-level interface is RobotWrapper, which
wraps model/data construction and FK helpers.  The IK loop here follows the
standard Pinocchio CLIK/DLS pattern while using RobotWrapper for loading/FK.

Pose format inside the script is [x, y, z, qx, qy, qz, qw].
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
IK_SOLVERS_DIR = SCRIPT_DIR / "ik_solvers"
if str(IK_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(IK_SOLVERS_DIR))

from z1_pinocchio_ik import (
    DEFAULT_Z1_EE_LINK,
    DEFAULT_Z1_JOINT_NAMES,
    DEFAULT_Z1_URDF,
    normalize_quat_xyzw,
    quat_xyzw_to_wxyz,
    quat_wxyz_to_xyzw,
    rot_to_quat_wxyz,
)


def quat_to_rot_xyzw(quat: np.ndarray) -> np.ndarray:
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


def quat_angle_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = normalize_quat_xyzw(lhs)
    rhs = normalize_quat_xyzw(rhs)
    dot = abs(float(np.dot(lhs, rhs)))
    return float(2.0 * math.acos(np.clip(dot, -1.0, 1.0)))


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "p50": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def fmt_stats(stats: Dict[str, float], digits: int = 3) -> str:
    if math.isnan(stats["mean"]):
        return "no samples"
    return (
        f"mean={stats['mean']:.{digits}f} p50={stats['p50']:.{digits}f} "
        f"p95={stats['p95']:.{digits}f} p99={stats['p99']:.{digits}f} max={stats['max']:.{digits}f}"
    )


class Z1RobotWrapperCLIK:
    def __init__(self, urdf_path: Path, ee_link: str = DEFAULT_Z1_EE_LINK):
        import pinocchio as pin
        from pinocchio.robot_wrapper import RobotWrapper
        from pinocchio.shortcuts import buildModelsFromUrdf

        self.pin = pin
        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        model = buildModelsFromUrdf(self.urdf_path, geometry_types=[])[0]
        self.robot = RobotWrapper(model=model)
        self.model = self.robot.model
        self.data = self.robot.data
        self.ee_link = str(ee_link)
        if not self.model.existFrame(self.ee_link):
            raise ValueError(f"Frame {self.ee_link!r} not found in {self.urdf_path}")
        self.ee_frame_id = int(self.model.getFrameId(self.ee_link))
        self.joint_names = list(DEFAULT_Z1_JOINT_NAMES)
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

    def pack_q(self, joints: np.ndarray) -> np.ndarray:
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

    def fk(self, joints: np.ndarray) -> np.ndarray:
        q = self.pack_q(joints)
        pose = self.robot.framePlacement(q, self.ee_frame_id, update_kinematics=True)
        quat_wxyz = rot_to_quat_wxyz(pose.rotation)
        return np.concatenate([np.asarray(pose.translation, dtype=np.float64), quat_wxyz_to_xyzw(quat_wxyz)])

    def ik(
        self,
        target_pose_xyzw: np.ndarray,
        seed_joints: np.ndarray,
        max_iter: int,
        dt: float,
        damping: float,
        threshold: float,
        rot_weight: float,
    ) -> Optional[np.ndarray]:
        q = self.pack_q(seed_joints)
        target_pose_xyzw = np.asarray(target_pose_xyzw, dtype=np.float64).reshape(7)
        pos_des = target_pose_xyzw[:3]
        rot_des = quat_to_rot_xyzw(target_pose_xyzw[3:7])
        weights = np.diag([1.0, 1.0, 1.0, float(rot_weight), float(rot_weight), float(rot_weight)])

        for i in range(max(1, int(max_iter))):
            o_m_tool = self.robot.framePlacement(q, self.ee_frame_id, update_kinematics=True)
            err_pos = pos_des - o_m_tool.translation
            rot_err_local = self.pin.log3(o_m_tool.rotation.T @ rot_des)
            err_rot_world = o_m_tool.rotation @ rot_err_local
            err = weights @ np.concatenate([err_pos, err_rot_world])
            if float(np.linalg.norm(err)) < float(threshold):
                return self.unpack_q(q)

            jac = self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.ee_frame_id,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            jac = weights @ jac
            hessian = jac.T @ jac + float(damping) * np.eye(self.model.nv)
            gradient = jac.T @ err
            try:
                dq = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            q_next = self.pin.integrate(self.model, q, dq * float(dt))
            q_next = np.clip(q_next, self.lower, self.upper)
            if float(np.linalg.norm(q_next - q)) < 1.0e-8 and i < max_iter - 1:
                dt *= 0.5
            q = q_next
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--ee_link", type=str, default=DEFAULT_Z1_EE_LINK)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_delta", type=float, default=0.25)
    parser.add_argument("--global_targets", action="store_true")
    parser.add_argument("--seed_mode", choices=("chain", "home", "middle", "zero", "target"), default="chain")
    parser.add_argument("--home_q", type=float, nargs=6, default=(0.0, 1.05, -1.45, 0.75, 0.0, 0.0))
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=1.0e-6)
    parser.add_argument("--threshold", type=float, default=1.0e-4)
    parser.add_argument("--rot_weight", type=float, default=0.5)
    return parser.parse_args()


def sample_items(ik: Z1RobotWrapperCLIK, args: argparse.Namespace, rng: np.random.Generator, count: int):
    lower, upper = ik.joint_limits()
    margin = np.minimum(0.05, np.maximum(upper - lower, 0.0) * 0.02)
    middle = np.clip(0.5 * (lower + upper), lower + margin, upper - margin)
    zero = np.clip(np.zeros_like(middle), lower + margin, upper - margin)
    home = np.clip(np.asarray(args.home_q, dtype=np.float64), lower + margin, upper - margin)
    chain_seed = home.copy()
    items = []
    for _ in range(count):
        if bool(args.global_targets):
            target_q = lower + margin + rng.random(home.shape[0]) * (upper - lower - 2.0 * margin)
        else:
            delta = rng.uniform(-float(args.target_delta), float(args.target_delta), size=home.shape[0])
            delta[-1] *= 1.6
            target_q = np.clip(chain_seed + delta, lower + margin, upper - margin)

        if args.seed_mode == "target":
            seed_q = target_q.copy()
        elif args.seed_mode == "middle":
            seed_q = middle.copy()
        elif args.seed_mode == "zero":
            seed_q = zero.copy()
        elif args.seed_mode == "home":
            seed_q = home.copy()
        else:
            seed_q = chain_seed.copy()
        items.append((seed_q, target_q, ik.fk(target_q)))
        chain_seed = target_q
    return items


def main() -> int:
    import pinocchio as pin

    args = parse_args()
    ik_like_names = [name for name in dir(pin) if "ik" in name.lower() or "inversekin" in name.lower()]
    print("Pinocchio high-level IK API check")
    print(f"pinocchio_version={pin.__version__}")
    print(f"ik_like_symbols={ik_like_names}")
    print("note=no built-in high-level IK solver was found; benchmarking RobotWrapper-based CLIK/DLS.")

    rng = np.random.default_rng(int(args.seed))
    ik = Z1RobotWrapperCLIK(Path(args.urdf), ee_link=args.ee_link)
    items = sample_items(ik, args, rng, max(0, int(args.warmup)) + max(1, int(args.samples)))

    times_ms: List[float] = []
    success_times_ms: List[float] = []
    failure_times_ms: List[float] = []
    pos_errors: List[float] = []
    rot_errors: List[float] = []
    failures = 0
    first_failure = None
    for i, (seed_q, _target_q, target_pose) in enumerate(items):
        start_ns = time.perf_counter_ns()
        solution = ik.ik(
            target_pose,
            seed_joints=seed_q,
            max_iter=int(args.max_iter),
            dt=float(args.dt),
            damping=float(args.damping),
            threshold=float(args.threshold),
            rot_weight=float(args.rot_weight),
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
        achieved = ik.fk(solution)
        pos_errors.append(float(np.linalg.norm(achieved[:3] - target_pose[:3])))
        rot_errors.append(quat_angle_xyzw(target_pose[3:7], achieved[3:7]))

    successful = len(success_times_ms)
    total_s = float(np.sum(times_ms)) / 1000.0
    success_s = float(np.sum(success_times_ms)) / 1000.0

    print()
    print("Z1 Pinocchio RobotWrapper CLIK benchmark")
    print(f"urdf={Path(args.urdf).expanduser().resolve()}")
    print(
        f"samples={int(args.samples)} warmup={int(args.warmup)} seed={int(args.seed)} "
        f"global_targets={bool(args.global_targets)} target_delta={float(args.target_delta):.3f} "
        f"seed_mode={args.seed_mode}"
    )
    print(
        f"max_iter={int(args.max_iter)} dt={float(args.dt):.3f} damping={float(args.damping):.1e} "
        f"threshold={float(args.threshold):.1e} rot_weight={float(args.rot_weight):.3f}"
    )
    print(f"success={successful}/{int(args.samples)} failures={failures} first_failure={first_failure}")
    print(f"latency_all(ms):     {fmt_stats(summarize(times_ms), digits=3)}")
    print(f"latency_success(ms): {fmt_stats(summarize(success_times_ms), digits=3)}")
    if failure_times_ms:
        print(f"latency_failure(ms): {fmt_stats(summarize(failure_times_ms), digits=3)}")
    if total_s > 0.0:
        print(f"throughput_all_calls={len(times_ms) / total_s:.1f} Hz")
        print(f"throughput_successful_wall={successful / total_s:.1f} Hz")
    if success_s > 0.0:
        print(f"throughput_success_only={successful / success_s:.1f} Hz")
    print(f"pos_err(m):          {fmt_stats(summarize(pos_errors), digits=9)}")
    print(f"rot_err(rad):        {fmt_stats(summarize(rot_errors), digits=9)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
