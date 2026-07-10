#!/usr/bin/env python3
"""Benchmark standalone Pinocchio IK latency for the Unitree Z1 arm."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from z1_pinocchio_ik import (
    DEFAULT_Z1_EE_LINK,
    DEFAULT_Z1_URDF,
    Z1PinocchioIK,
    orientation_error_xyzw,
    quat_xyzw_to_wxyz,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--ee_link", type=str, default=DEFAULT_Z1_EE_LINK)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_delta", type=float, default=0.45)
    parser.add_argument("--global_targets", action="store_true")
    parser.add_argument(
        "--api",
        type=str,
        default="solve_ik",
        choices=("solve_ik", "ik"),
        help="solve_ik matches kinematics/pinocchio_kinematics.py; ik is the IsaacGym xyzw compatibility wrapper.",
    )
    parser.add_argument("--rot_weight", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--max_restarts", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=1.0e-6)
    parser.add_argument("--threshold", type=float, default=1.0e-4)
    parser.add_argument("--home_q", type=float, nargs=6, default=(0.0, 1.05, -1.45, 0.75, 0.0, 0.0))
    return parser.parse_args()


def sample_pairs(ik, args, rng, count):
    lower, upper = ik.joint_limits()
    margin = np.minimum(0.05, np.maximum(upper - lower, 0.0) * 0.02)
    seed_q = np.clip(np.asarray(args.home_q, dtype=np.float64), lower + margin, upper - margin)
    pairs = []
    for _ in range(count):
        if args.global_targets:
            target_q = lower + margin + rng.random(seed_q.shape[0]) * (upper - lower - 2.0 * margin)
        else:
            delta = rng.uniform(-float(args.target_delta), float(args.target_delta), size=seed_q.shape[0])
            delta[5] *= 1.6
            target_q = np.clip(seed_q + delta, lower + margin, upper - margin)
        pairs.append((seed_q.copy(), ik.fk(target_q)))
        seed_q = target_q
    return pairs


def summarize_ms(name, values_ms):
    values_ms = np.asarray(values_ms, dtype=np.float64)
    return (
        f"{name}: mean={np.mean(values_ms):.3f} ms "
        f"p50={np.percentile(values_ms, 50):.3f} ms "
        f"p90={np.percentile(values_ms, 90):.3f} ms "
        f"p95={np.percentile(values_ms, 95):.3f} ms "
        f"p99={np.percentile(values_ms, 99):.3f} ms "
        f"max={np.max(values_ms):.3f} ms"
    )


def pose_error(ik, target_pose, joints):
    pose = ik.fk(joints)
    pos_err = float(np.linalg.norm(np.asarray(target_pose[:3]) - np.asarray(pose[:3])))
    rot_err = float(np.linalg.norm(orientation_error_xyzw(target_pose[3:7], pose[3:7])))
    return pos_err, rot_err


def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    ik = Z1PinocchioIK(Path(args.urdf), ee_link=args.ee_link)
    total_count = max(0, int(args.warmup)) + max(1, int(args.samples))
    pairs = sample_pairs(ik, args, rng, total_count)

    times_ms = []
    success_times_ms = []
    failure_times_ms = []
    pos_errors = []
    rot_errors = []
    failures = 0
    first_failure = None

    for i, (seed_q, target_pose) in enumerate(pairs):
        start_ns = time.perf_counter_ns()
        if args.api == "solve_ik":
            target_pose_wxyz = np.concatenate([target_pose[:3], quat_xyzw_to_wxyz(target_pose[3:7])])
            solution = ik.solve_ik(
                target_pose_wxyz,
                seed_joints=seed_q,
                max_iter=int(args.max_iter),
                max_restarts=int(args.max_restarts),
                dt=float(args.dt),
                damp=float(args.damping),
                threshold=float(args.threshold),
            )
        else:
            solution = ik.ik(
                target_pose,
                seed_joints=seed_q,
                position_only=False,
                max_iter=int(args.max_iter),
                max_restarts=int(args.max_restarts),
                dt=float(args.dt),
                damping=float(args.damping),
                threshold=float(args.threshold),
                rotation_weight=float(args.rot_weight),
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
        pos_err, rot_err = pose_error(ik, target_pose, solution)
        pos_errors.append(pos_err)
        rot_errors.append(rot_err)

    successful = len(times_ms) - failures
    total_s = float(np.sum(times_ms)) / 1000.0
    success_only_s = float(np.sum(success_times_ms)) / 1000.0
    throughput = successful / total_s if total_s > 0.0 else float("inf")
    throughput_all = len(times_ms) / total_s if total_s > 0.0 else float("inf")
    throughput_success_only = successful / success_only_s if success_only_s > 0.0 else float("inf")

    print("Z1 Pinocchio IK speed benchmark")
    print(f"urdf={Path(args.urdf).expanduser().resolve()}")
    print(
        f"samples={int(args.samples)} warmup={int(args.warmup)} "
        f"global_targets={bool(args.global_targets)}"
    )
    if args.api == "solve_ik":
        constraint_text = "constraint=[1,1,1,1,1,1]"
    else:
        constraint_text = f"rot_weight={float(args.rot_weight):.3f}"
    print(
        f"api={args.api} {constraint_text} max_iter={int(args.max_iter)} "
        f"max_restarts={int(args.max_restarts)} dt={float(args.dt):.3f} "
        f"damping={float(args.damping):.1e} threshold={float(args.threshold):.1e}"
    )
    print(f"success={successful} failures={failures} first_failure={first_failure}")
    print(summarize_ms("latency_all", times_ms))
    if success_times_ms:
        print(summarize_ms("latency_success_only", success_times_ms))
    if failure_times_ms:
        print(summarize_ms("latency_failure_only", failure_times_ms))
    print(f"throughput_all_calls={throughput_all:.1f} calls/s")
    print(f"throughput_successful_wall={throughput:.1f} successful solves/s including failure time")
    print(f"throughput_success_only={throughput_success_only:.1f} successful solves/s excluding failure time")
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
