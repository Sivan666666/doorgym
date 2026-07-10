#!/usr/bin/env python3
"""TRAC-IK FK/IK and benchmark for the Unitree Z1 arm.

Reference style:
    /home/sivan/cloth/deformable_bench/robot/kinematics/tracik_kinematics.py

Pose format for solve_fk/solve_ik:
    [x, y, z, qw, qx, qy, qz]

Compatibility wrappers fk/ik use:
    [x, y, z, qx, qy, qz, qw]
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from trac_ik import TracIK
except ImportError as exc:
    TracIK = None
    _TRACIK_IMPORT_ERROR = exc
else:
    _TRACIK_IMPORT_ERROR = None


HIGH_LEVEL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_Z1_URDF = HIGH_LEVEL_ROOT / "data" / "asset" / "z1" / "urdf" / "z1_arm.urdf"
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


def quat_to_rot_wxyz(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    quat_xyzw = quat_wxyz_to_xyzw(quat)
    return R.from_quat(quat_xyzw).as_matrix()


def rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(np.asarray(rot, dtype=np.float64).reshape(3, 3)).as_quat()
    return quat_xyzw_to_wxyz(quat_xyzw)


def orientation_error_wxyz(
    desired: Union[np.ndarray, List[float]],
    current: Union[np.ndarray, List[float]],
) -> float:
    desired_rot = R.from_quat(quat_wxyz_to_xyzw(desired))
    current_rot = R.from_quat(quat_wxyz_to_xyzw(current))
    return float((current_rot.inv() * desired_rot).magnitude())


class Z1TracIKKinematics:
    """TRAC-IK based Z1 FK/IK in the URDF root frame."""

    def __init__(
        self,
        urdf_path: Union[str, Path] = DEFAULT_Z1_URDF,
        base_link: str = DEFAULT_Z1_BASE_LINK,
        ee_link: str = DEFAULT_Z1_EE_LINK,
        timeout: float = 0.005,
        epsilon: float = 1.0e-5,
        solver_type: str = "Speed",
        joint_names: Union[Tuple[str, ...], List[str]] = DEFAULT_Z1_JOINT_NAMES,
    ):
        if TracIK is None:
            raise RuntimeError(
                "trac_ik is not installed in this Python environment. "
                "For this workspace use: conda activate b1z1"
            ) from _TRACIK_IMPORT_ERROR

        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        if not Path(self.urdf_path).exists():
            raise FileNotFoundError(f"Z1 URDF not found: {self.urdf_path}")

        self.base_link = str(base_link)
        self.ee_link = str(ee_link)
        self.joint_names = list(joint_names)
        self.timeout = float(timeout)
        self.epsilon = float(epsilon)
        self.solver_type = str(solver_type)

        self.ik_solver = TracIK(
            base_link_name=self.base_link,
            tip_link_name=self.ee_link,
            urdf_path=self.urdf_path,
            timeout=self.timeout,
            epsilon=self.epsilon,
            solver_type=self.solver_type,
        )
        self.num_joints = int(self.ik_solver.dof)
        self.arm_joints = self.num_joints
        self.lower_limits, self.upper_limits = self.ik_solver.joint_limits
        self.lower_limits = np.asarray(self.lower_limits, dtype=np.float64)
        self.upper_limits = np.asarray(self.upper_limits, dtype=np.float64)
        self.limits = list(zip(self.lower_limits.tolist(), self.upper_limits.tolist()))

        if self.num_joints != len(self.joint_names):
            warnings.warn(
                f"Expected {len(self.joint_names)} Z1 arm joints, got {self.num_joints}.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _ensure_numpy(self, data) -> np.ndarray:
        if isinstance(data, (list, tuple)):
            return np.asarray(data, dtype=np.float64)
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            return data.cpu().numpy().astype(np.float64)
        return np.asarray(data, dtype=np.float64)

    def _clip_q(self, q: Union[List[float], np.ndarray]) -> np.ndarray:
        q_arr = self._ensure_numpy(q).reshape(-1)
        if q_arr.shape[0] > self.num_joints:
            q_arr = q_arr[: self.num_joints]
        elif q_arr.shape[0] < self.num_joints:
            q_arr = np.pad(q_arr, (0, self.num_joints - q_arr.shape[0]), mode="constant")
        return np.clip(q_arr.astype(np.float64, copy=False), self.lower_limits, self.upper_limits)

    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lower_limits.copy(), self.upper_limits.copy()

    def middle_q(self) -> np.ndarray:
        return 0.5 * (self.lower_limits + self.upper_limits)

    def solve_fk(
        self,
        joint_angles: Union[List[float], np.ndarray],
        ee_link: Optional[str] = None,
    ) -> np.ndarray:
        """Forward kinematics returning [x, y, z, qw, qx, qy, qz]."""
        if ee_link is not None and ee_link != self.ee_link:
            raise ValueError("pytracik fixes the tip link at construction time; create a new solver.")
        q = self._clip_q(joint_angles)
        position, rotation_matrix = self.ik_solver.fk(q)
        return np.concatenate([np.asarray(position, dtype=np.float64), rot_to_quat_wxyz(rotation_matrix)])

    def fk(self, joints: Union[List[float], np.ndarray]) -> np.ndarray:
        """Compatibility FK returning [x, y, z, qx, qy, qz, qw]."""
        pose_wxyz = self.solve_fk(joints)
        return np.concatenate([pose_wxyz[:3], quat_wxyz_to_xyzw(pose_wxyz[3:7])])

    def _pose_constraint_mask(
        self,
        pose_constraint: Optional[Union[np.ndarray, List[float], Dict[str, Union[np.ndarray, List[float]]]]],
        position_only: bool = False,
    ) -> np.ndarray:
        if position_only:
            return np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if isinstance(pose_constraint, dict):
            mask = pose_constraint.get("world")
            if mask is None:
                warnings.warn(
                    "TRAC-IK does not support local/mixed pose constraints; using full pose target.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return np.ones(6, dtype=np.float64)
        elif pose_constraint is None:
            return np.ones(6, dtype=np.float64)
        else:
            mask = pose_constraint

        mask_arr = np.asarray(mask, dtype=np.float64).reshape(-1)
        if mask_arr.shape == (3,):
            mask_arr = np.concatenate([mask_arr, np.zeros(3, dtype=np.float64)])
        if mask_arr.shape != (6,):
            raise ValueError(f"pose_constraint must have 3 or 6 values, got {mask_arr.shape}")
        return mask_arr

    def _effective_target_pose(
        self,
        target_pose: np.ndarray,
        seed: np.ndarray,
        pose_constraint: Optional[Union[np.ndarray, List[float], Dict[str, Union[np.ndarray, List[float]]]]],
        position_only: bool,
    ) -> np.ndarray:
        mask = self._pose_constraint_mask(pose_constraint, position_only=position_only)
        if np.all(mask[3:6] > 0.0):
            return target_pose
        if np.any(mask[3:6] > 0.0):
            warnings.warn(
                "TRAC-IK does not support partial orientation constraints; using full target orientation.",
                RuntimeWarning,
                stacklevel=3,
            )
            return target_pose
        seed_pose = self.solve_fk(seed)
        return np.concatenate([target_pose[:3], seed_pose[3:7]])

    def solve_ik(
        self,
        target_pose: Union[np.ndarray, List[float]],
        seed_joints: Optional[Union[np.ndarray, List[float]]] = None,
        threshold: float = 1.0e-5,
        pose_constraint: Optional[Union[np.ndarray, List[float], Dict[str, Union[np.ndarray, List[float]]]]] = None,
        constraint_frame: str = "world",
        ee_link: Optional[str] = None,
        max_restarts: int = 1,
        strict_seed: bool = True,
        position_only: bool = False,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Inverse kinematics for [x, y, z, qw, qx, qy, qz] targets.

        TRAC-IK solves a full pose target. If position_only=True, or if
        pose_constraint has zero orientation weights, this wrapper keeps the
        seed end-effector orientation and solves only the requested position
        in practice.
        """
        del threshold, kwargs
        if ee_link is not None and ee_link != self.ee_link:
            raise ValueError("pytracik fixes the tip link at construction time; create a new solver.")
        if constraint_frame != "world":
            warnings.warn(
                "TRAC-IK wrapper only supports world-frame constraints.",
                RuntimeWarning,
                stacklevel=2,
            )

        pose_arr = self._ensure_numpy(target_pose).reshape(-1)
        if pose_arr.shape[0] != 7:
            raise ValueError(f"target_pose must be [x,y,z,qw,qx,qy,qz], got {pose_arr.shape}")

        if seed_joints is None:
            seed = self._clip_q(np.zeros(self.num_joints, dtype=np.float64))
        else:
            seed = self._clip_q(seed_joints)

        target = self._effective_target_pose(pose_arr, seed, pose_constraint, bool(position_only))
        target_pos = np.asarray(target[:3], dtype=np.float64)
        target_rot = quat_to_rot_wxyz(target[3:7])

        seed_strategies = [seed]
        if not strict_seed and int(max_restarts) > 1:
            seed_strategies.append(self._clip_q(np.zeros(self.num_joints, dtype=np.float64)))
            seed_strategies.append(self._clip_q(self.middle_q()))
            generator = rng if rng is not None else np.random.default_rng(0)
            while len(seed_strategies) < int(max_restarts):
                random_seed = generator.uniform(self.lower_limits, self.upper_limits)
                seed_strategies.append(self._clip_q(random_seed))

        for seed_values in seed_strategies[: max(1, int(max_restarts))]:
            try:
                result = self.ik_solver.ik(
                    tgt_pos=target_pos,
                    tgt_rot=target_rot,
                    seed_jnt_values=seed_values,
                )
            except Exception as exc:
                warnings.warn(f"TRAC-IK solver exception: {exc}", RuntimeWarning, stacklevel=2)
                return None
            if result is not None:
                result = np.asarray(result, dtype=np.float64).reshape(-1)
                if result.shape[0] == self.num_joints and np.all(np.isfinite(result)):
                    return self._clip_q(result)
        return None

    def ik(
        self,
        target_pose: Union[np.ndarray, List[float]],
        seed_joints: Optional[Union[np.ndarray, List[float]]] = None,
        position_only: bool = False,
        threshold: float = 1.0e-5,
        rotation_weight: float = 1.0,
        max_restarts: int = 1,
        strict_seed: bool = True,
        rng: Optional[np.random.Generator] = None,
        pose_constraint: Optional[Union[np.ndarray, List[float], Dict[str, Union[np.ndarray, List[float]]]]] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Compatibility IK wrapper using [x, y, z, qx, qy, qz, qw] targets."""
        pose_arr = self._ensure_numpy(target_pose).reshape(-1)
        if pose_arr.shape[0] == 3:
            seed = self._clip_q(seed_joints if seed_joints is not None else np.zeros(self.num_joints))
            target_wxyz = np.concatenate([pose_arr[:3], self.solve_fk(seed)[3:7]])
            position_only = True
        elif pose_arr.shape[0] == 7:
            target_wxyz = np.concatenate([pose_arr[:3], quat_xyzw_to_wxyz(pose_arr[3:7])])
        else:
            raise ValueError(f"target_pose must have 3 or 7 values, got {pose_arr.shape[0]}")

        if pose_constraint is None and (position_only or float(rotation_weight) == 0.0):
            pose_constraint = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

        return self.solve_ik(
            target_wxyz,
            seed_joints=seed_joints,
            threshold=float(threshold),
            pose_constraint=pose_constraint,
            max_restarts=int(max_restarts),
            strict_seed=bool(strict_seed),
            position_only=bool(position_only),
            rng=rng,
            **kwargs,
        )


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


def pose_error(kin: Z1TracIKKinematics, target_pose: np.ndarray, joints: np.ndarray) -> Tuple[float, float]:
    pose = kin.solve_fk(joints)
    pos_err = float(np.linalg.norm(np.asarray(target_pose[:3]) - pose[:3]))
    rot_err = orientation_error_wxyz(target_pose[3:7], pose[3:7])
    return pos_err, rot_err


def sample_items(
    kin: Z1TracIKKinematics,
    args: argparse.Namespace,
    rng: np.random.Generator,
    count: int,
):
    lower, upper = kin.joint_limits()
    margin = np.minimum(0.05, np.maximum(upper - lower, 0.0) * 0.02)
    middle = np.clip(0.5 * (lower + upper), lower + margin, upper - margin)
    zero = np.clip(np.zeros_like(middle), lower + margin, upper - margin)
    home = np.clip(np.asarray(args.home_q, dtype=np.float64), lower + margin, upper - margin)
    chain_seed = home.copy()
    items = []
    for _ in range(count):
        if args.global_targets:
            target_q = lower + margin + rng.random(middle.shape[0]) * (upper - lower - 2.0 * margin)
        else:
            delta = rng.uniform(-float(args.target_delta), float(args.target_delta), size=middle.shape[0])
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

        items.append((seed_q, kin.solve_fk(target_q), target_q))
        chain_seed = target_q
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--base_link", type=str, default=DEFAULT_Z1_BASE_LINK)
    parser.add_argument("--ee_link", type=str, default=DEFAULT_Z1_EE_LINK)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_delta", type=float, default=0.35)
    parser.add_argument("--global_targets", action="store_true")
    parser.add_argument("--seed_mode", choices=("chain", "target", "middle", "zero", "home"), default="chain")
    parser.add_argument("--timeout", type=float, default=0.005)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--solver_type", choices=("Speed", "Distance", "Manip1", "Manip2"), default="Speed")
    parser.add_argument("--max_restarts", type=int, default=1)
    parser.add_argument("--allow_restarts", action="store_true")
    parser.add_argument("--position_only", action="store_true")
    parser.add_argument("--threshold", type=float, default=1.0e-5)
    parser.add_argument("--home_q", type=float, nargs=6, default=(0.0, 1.05, -1.45, 0.75, 0.0, 0.0))
    parser.add_argument("--assert_success_rate", type=float, default=0.0)
    parser.add_argument("--assert_pos_err", type=float, default=np.inf)
    parser.add_argument("--assert_rot_err", type=float, default=np.inf)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    kin = Z1TracIKKinematics(
        args.urdf,
        base_link=args.base_link,
        ee_link=args.ee_link,
        timeout=float(args.timeout),
        epsilon=float(args.epsilon),
        solver_type=args.solver_type,
    )
    total_count = max(0, int(args.warmup)) + max(1, int(args.samples))
    items = sample_items(kin, args, rng, total_count)

    times_ms = []
    success_times_ms = []
    failure_times_ms = []
    pos_errors = []
    rot_errors = []
    failures = 0
    first_failure = None

    for i, (seed_q, target_pose, _target_q) in enumerate(items):
        start_ns = time.perf_counter_ns()
        solution = kin.solve_ik(
            target_pose,
            seed_joints=seed_q,
            threshold=float(args.threshold),
            max_restarts=max(1, int(args.max_restarts)),
            strict_seed=not bool(args.allow_restarts),
            position_only=bool(args.position_only),
            rng=rng,
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
        pos_err, rot_err = pose_error(kin, target_pose, solution)
        pos_errors.append(pos_err)
        rot_errors.append(rot_err)

    successful = len(times_ms) - failures
    total_s = float(np.sum(times_ms)) / 1000.0
    success_only_s = float(np.sum(success_times_ms)) / 1000.0
    throughput_all = len(times_ms) / total_s if total_s > 0.0 else float("inf")
    throughput_success = successful / total_s if total_s > 0.0 else float("inf")
    throughput_success_only = successful / success_only_s if success_only_s > 0.0 else float("inf")
    success_rate = successful / max(1, len(times_ms))

    print("Z1 TRAC-IK benchmark")
    print(f"urdf={Path(args.urdf).expanduser().resolve()}")
    print(f"base_link={args.base_link} ee_link={args.ee_link} dof={kin.num_joints}")
    print(
        f"samples={int(args.samples)} warmup={int(args.warmup)} seed={int(args.seed)} "
        f"global_targets={bool(args.global_targets)} seed_mode={args.seed_mode}"
    )
    print(
        f"solver_type={args.solver_type} timeout={float(args.timeout):.4f}s "
        f"epsilon={float(args.epsilon):.1e} max_restarts={int(args.max_restarts)} "
        f"allow_restarts={bool(args.allow_restarts)} position_only={bool(args.position_only)}"
    )
    print(f"success={successful} failures={failures} success_rate={success_rate:.3f} first_failure={first_failure}")
    print(summarize_ms("latency_all", times_ms))
    if success_times_ms:
        print(summarize_ms("latency_success_only", success_times_ms))
    if failure_times_ms:
        print(summarize_ms("latency_failure_only", failure_times_ms))
    print(f"throughput_all_calls={throughput_all:.1f} calls/s")
    print(f"throughput_successful_wall={throughput_success:.1f} successful solves/s including failure time")
    print(f"throughput_success_only={throughput_success_only:.1f} successful solves/s excluding failure time")

    max_pos_err = float("inf")
    max_rot_err = float("inf")
    if pos_errors:
        max_pos_err = float(np.max(pos_errors))
        max_rot_err = float(np.max(rot_errors))
        print(
            "pos_err(m): "
            f"mean={np.mean(pos_errors):.9f} p95={np.percentile(pos_errors, 95):.9f} "
            f"max={max_pos_err:.9f}"
        )
        print(
            "rot_err(rad): "
            f"mean={np.mean(rot_errors):.9f} p95={np.percentile(rot_errors, 95):.9f} "
            f"max={max_rot_err:.9f}"
        )

    if success_rate < float(args.assert_success_rate):
        return 2
    if max_pos_err > float(args.assert_pos_err):
        return 3
    if max_rot_err > float(args.assert_rot_err):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
