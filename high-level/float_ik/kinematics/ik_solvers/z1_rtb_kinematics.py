#!/usr/bin/env python3
"""Robotics Toolbox IK benchmark for Unitree Z1.

Reference style:
    /home/sivan/cloth/deformable_bench/robot/kinematics/rtb_kinematics.py

Pose format:
    [x, y, z, qw, qx, qy, qz]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3, UnitQuaternion


HIGH_LEVEL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_Z1_URDF = HIGH_LEVEL_ROOT / "data" / "asset" / "z1" / "urdf" / "z1_arm.urdf"
DEFAULT_BASE_LINK = "base"
DEFAULT_EE_LINK = "ee_gripper_link"


def normalize_quat_wxyz(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_to_rot_wxyz(quat: Union[np.ndarray, List[float]]) -> np.ndarray:
    return UnitQuaternion(normalize_quat_wxyz(quat)).R


class Z1RtbKinematics:
    def __init__(self, urdf_path: Union[str, Path] = DEFAULT_Z1_URDF, base_link: str = DEFAULT_BASE_LINK, ee_link: str = DEFAULT_EE_LINK):
        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        if not Path(self.urdf_path).exists():
            raise FileNotFoundError(f"Z1 URDF not found: {self.urdf_path}")
        self.base_link = str(base_link)
        self.ee_link = str(ee_link)
        self.robot = rtb.ERobot.URDF(self.urdf_path)
        self.ets = self.robot.ets(start=self.base_link, end=self.ee_link)
        self.n = int(self.ets.n)
        if self.n != 6:
            print(f"Warning: expected 6 Z1 arm joints on chain, got ets.n={self.n}")

        if self.robot.qlim is None:
            self.lower = np.full(self.n, -np.pi, dtype=np.float64)
            self.upper = np.full(self.n, np.pi, dtype=np.float64)
        else:
            qlim = np.asarray(self.robot.qlim, dtype=np.float64)
            self.lower = qlim[0, : self.n].copy()
            self.upper = qlim[1, : self.n].copy()

    def _ensure_numpy(self, data) -> np.ndarray:
        if isinstance(data, (list, tuple)):
            return np.asarray(data, dtype=np.float64)
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            return data.cpu().numpy().astype(np.float64)
        return np.asarray(data, dtype=np.float64)

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        q = self._ensure_numpy(q).reshape(-1)
        if q.shape[0] > self.n:
            q = q[: self.n]
        elif q.shape[0] < self.n:
            q = np.pad(q, (0, self.n - q.shape[0]), mode="constant")
        return np.clip(q.astype(np.float64, copy=False), self.lower, self.upper)

    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lower.copy(), self.upper.copy()

    def solve_fk(self, joint_angles: Union[List[float], np.ndarray], ee_link: Optional[str] = None) -> np.ndarray:
        """FK: joints -> [x, y, z, qw, qx, qy, qz]."""
        q = self._clip_q(joint_angles)
        if ee_link is None or ee_link == self.ee_link:
            transform = self.ets.fkine(q)
        else:
            transform = self.robot.fkine(q, end=ee_link)
        quat = UnitQuaternion(transform.R).A
        return np.concatenate([np.asarray(transform.t, dtype=np.float64), normalize_quat_wxyz(quat)])

    def solve_ik(
        self,
        target_pose: Union[List[float], np.ndarray],
        seed_joints: Optional[Union[List[float], np.ndarray]] = None,
        threshold: float = 1.0e-4,
        pose_constraint: Optional[Union[List[float], np.ndarray, Dict[str, Union[List[float], np.ndarray]]]] = None,
        constraint_frame: str = "world",
        ee_link: Optional[str] = None,
        ilimit: int = 80,
        joint_limits: bool = True,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """IK: [x, y, z, qw, qx, qy, qz] -> joints, using RTB ikine_LM."""
        del kwargs
        pose_arr = self._ensure_numpy(target_pose).reshape(7)
        ee = ee_link if ee_link is not None else self.ee_link
        pos = pose_arr[:3]
        quat = normalize_quat_wxyz(pose_arr[3:7])
        goal = SE3(pos) * UnitQuaternion(quat).SE3()

        if seed_joints is None:
            q0 = np.zeros(self.n, dtype=np.float64)
        else:
            q0 = self._clip_q(seed_joints)

        mask = np.ones(6, dtype=np.float64)
        if isinstance(pose_constraint, dict):
            if "world" in pose_constraint:
                mask = np.asarray(pose_constraint["world"], dtype=np.float64)
            else:
                print("[RTB] complex non-world dict constraints are not supported by ikine_LM; using full mask.")
        elif pose_constraint is not None:
            if isinstance(pose_constraint, bool):
                mask = np.ones(6, dtype=np.float64) if pose_constraint else np.asarray([1, 1, 1, 0, 0, 0], dtype=np.float64)
            else:
                mask = np.asarray(pose_constraint, dtype=np.float64)
                if mask.shape == (3,):
                    mask = np.concatenate([mask, np.zeros(3, dtype=np.float64)])
            if constraint_frame == "local":
                print("[RTB] local constraint frame is not standard in ikine_LM; using mask in RTB/world convention.")

        sol = self.robot.ikine_LM(
            goal,
            q0=q0,
            mask=mask,
            end=ee,
            tol=float(threshold),
            ilimit=int(ilimit),
            joint_limits=bool(joint_limits),
        )
        if sol.success:
            return np.asarray(sol.q, dtype=np.float64)[: self.n]
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--base_link", type=str, default=DEFAULT_BASE_LINK)
    parser.add_argument("--ee_link", type=str, default=DEFAULT_EE_LINK)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_delta", type=float, default=0.25)
    parser.add_argument("--global_targets", action="store_true")
    parser.add_argument("--seed_mode", choices=("chain", "target", "middle", "zero"), default="chain")
    parser.add_argument("--rot_weight", type=float, default=0.5)
    parser.add_argument("--ilimit", type=int, default=80)
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


def pose_error(kin: Z1RtbKinematics, target_pose: np.ndarray, joints: np.ndarray) -> Tuple[float, float]:
    pose = kin.solve_fk(joints)
    pos_err = float(np.linalg.norm(target_pose[:3] - pose[:3]))
    rot_target = quat_to_rot_wxyz(target_pose[3:7])
    rot_current = quat_to_rot_wxyz(pose[3:7])
    cos_angle = np.clip((np.trace(rot_current.T @ rot_target) - 1.0) * 0.5, -1.0, 1.0)
    rot_err = float(np.arccos(cos_angle))
    return pos_err, rot_err


def sample_items(kin: Z1RtbKinematics, args, rng: np.random.Generator, count: int):
    lower, upper = kin.joint_limits()
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
            delta[-1] *= 1.6
            target_q = np.clip(chain_seed + delta, lower + margin, upper - margin)

        if args.seed_mode == "target":
            seed_q = target_q.copy()
        elif args.seed_mode == "middle":
            seed_q = middle.copy()
        elif args.seed_mode == "zero":
            seed_q = zero.copy()
        else:
            seed_q = chain_seed.copy()

        items.append((seed_q, kin.solve_fk(target_q), target_q))
        chain_seed = target_q
    return items


def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    kin = Z1RtbKinematics(args.urdf, base_link=args.base_link, ee_link=args.ee_link)
    items = sample_items(kin, args, rng, int(args.samples) + int(args.warmup))
    mask = [1.0, 1.0, 1.0, float(args.rot_weight), float(args.rot_weight), float(args.rot_weight)]

    times_ms = []
    success_times_ms = []
    failure_times_ms = []
    pos_errors = []
    rot_errors = []
    failures = 0
    first_failure = None
    for i, (seed_q, target_pose, _) in enumerate(items):
        start_ns = time.perf_counter_ns()
        solution = kin.solve_ik(
            target_pose,
            seed_joints=seed_q,
            pose_constraint=mask,
            threshold=float(args.threshold),
            ilimit=int(args.ilimit),
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

    successful = len(success_times_ms)
    total_s = float(np.sum(times_ms)) / 1000.0
    success_only_s = float(np.sum(success_times_ms)) / 1000.0
    print("Z1 Robotics Toolbox IK benchmark")
    print(f"urdf={Path(args.urdf).expanduser().resolve()}")
    print(f"base_link={args.base_link} ee_link={args.ee_link} chain_n={kin.n}")
    print(
        f"samples={int(args.samples)} warmup={int(args.warmup)} "
        f"global_targets={bool(args.global_targets)} target_delta={float(args.target_delta):.3f} "
        f"seed_mode={args.seed_mode}"
    )
    print(
        f"mask=[1,1,1,{float(args.rot_weight):.3f},{float(args.rot_weight):.3f},{float(args.rot_weight):.3f}] "
        f"ilimit={int(args.ilimit)} threshold={float(args.threshold):.1e}"
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
            "rot_err(rad): "
            f"mean={np.mean(rot_errors):.6f} p95={np.percentile(rot_errors, 95):.6f} "
            f"max={np.max(rot_errors):.6f}"
        )


if __name__ == "__main__":
    main()
