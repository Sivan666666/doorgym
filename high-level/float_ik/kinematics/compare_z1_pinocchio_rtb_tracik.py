#!/usr/bin/env python3
"""Compare Z1 Pinocchio, Robotics Toolbox, and TRAC-IK on reachable 6D IK targets."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
IK_SOLVERS_DIR = SCRIPT_DIR / "ik_solvers"
if str(IK_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(IK_SOLVERS_DIR))

from z1_pinocchio_ik import DEFAULT_Z1_EE_LINK, DEFAULT_Z1_URDF, Z1PinocchioIK, normalize_quat_xyzw
from z1_rtb_kinematics import Z1RtbKinematics
from z1_tracik_kinematics import Z1TracIKKinematics


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat_xyzw(quat)
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def pose_xyzw_to_wxyz(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    return np.concatenate([pose[:3], xyzw_to_wxyz(pose[3:7])])


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


def fmt_ms(stats: Dict[str, float]) -> str:
    if math.isnan(stats["mean"]):
        return "no samples"
    return (
        f"mean={stats['mean']:.3f} p50={stats['p50']:.3f} "
        f"p95={stats['p95']:.3f} p99={stats['p99']:.3f} max={stats['max']:.3f}"
    )


def fmt_err(stats: Dict[str, float], digits: int = 6) -> str:
    if math.isnan(stats["mean"]):
        return "no samples"
    return (
        f"mean={stats['mean']:.{digits}f} p50={stats['p50']:.{digits}f} "
        f"p95={stats['p95']:.{digits}f} p99={stats['p99']:.{digits}f} "
        f"max={stats['max']:.{digits}f}"
    )


@dataclass
class Sample:
    seed_q: np.ndarray
    target_q: np.ndarray
    target_pose_xyzw: np.ndarray


@dataclass
class Solver:
    name: str
    solve: Callable[[np.ndarray, np.ndarray, np.random.Generator], Optional[np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--ee_link", type=str, default=DEFAULT_Z1_EE_LINK)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_delta", type=float, nargs="+", default=(0.1, 0.25, 0.45))
    parser.add_argument("--global_targets", action="store_true")
    parser.add_argument("--seed_mode", choices=("chain", "target", "middle", "zero", "home"), default="chain")
    parser.add_argument("--home_q", type=float, nargs=6, default=(0.0, 1.05, -1.45, 0.75, 0.0, 0.0))
    parser.add_argument("--rot_weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=1.0e-4)
    parser.add_argument("--pin_max_iter", type=int, default=1000)
    parser.add_argument("--pin_restarts", type=int, default=1)
    parser.add_argument("--pin_dt", type=float, default=0.05)
    parser.add_argument("--pin_damping", type=float, default=1.0e-6)
    parser.add_argument("--rtb_ilimit", type=int, default=80)
    parser.add_argument("--tracik_timeout", type=float, default=0.005)
    parser.add_argument("--tracik_epsilon", type=float, default=1.0e-5)
    parser.add_argument("--tracik_solver_type", choices=("Speed", "Distance", "Manip1", "Manip2"), default="Speed")
    parser.add_argument("--tracik_restarts", type=int, default=1)
    parser.add_argument("--tracik_allow_restarts", action="store_true")
    return parser.parse_args()


def sample_reachable_targets(
    verifier: Z1PinocchioIK,
    args: argparse.Namespace,
    rng: np.random.Generator,
    target_delta: float,
    count: int,
) -> List[Sample]:
    lower, upper = verifier.joint_limits()
    margin = np.minimum(0.05, np.maximum(upper - lower, 0.0) * 0.02)
    middle = np.clip(0.5 * (lower + upper), lower + margin, upper - margin)
    zero = np.clip(np.zeros_like(middle), lower + margin, upper - margin)
    home = np.clip(np.asarray(args.home_q, dtype=np.float64), lower + margin, upper - margin)
    chain_seed = home.copy()
    samples: List[Sample] = []

    for _ in range(count):
        if bool(args.global_targets):
            target_q = lower + margin + rng.random(home.shape[0]) * (upper - lower - 2.0 * margin)
        else:
            delta = rng.uniform(-float(target_delta), float(target_delta), size=home.shape[0])
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

        samples.append(
            Sample(
                seed_q=seed_q,
                target_q=target_q,
                target_pose_xyzw=np.asarray(verifier.fk(target_q), dtype=np.float64),
            )
        )
        chain_seed = target_q
    return samples


def run_solver(
    solver: Solver,
    verifier: Z1PinocchioIK,
    samples: List[Sample],
    warmup: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(int(seed))
    times_ms: List[float] = []
    success_times_ms: List[float] = []
    failure_times_ms: List[float] = []
    pos_errors: List[float] = []
    rot_errors: List[float] = []
    failures = 0
    first_failure = None

    for index, sample in enumerate(samples):
        start_ns = time.perf_counter_ns()
        solution = solver.solve(sample.target_pose_xyzw, sample.seed_q, rng)
        elapsed_ms = (time.perf_counter_ns() - start_ns) * 1.0e-6

        if index < int(warmup):
            continue

        times_ms.append(elapsed_ms)
        if solution is None:
            failures += 1
            failure_times_ms.append(elapsed_ms)
            if first_failure is None:
                first_failure = index - int(warmup)
            continue

        q = np.asarray(solution, dtype=np.float64).reshape(-1)[:6]
        success_times_ms.append(elapsed_ms)
        achieved = np.asarray(verifier.fk(q), dtype=np.float64)
        pos_errors.append(float(np.linalg.norm(achieved[:3] - sample.target_pose_xyzw[:3])))
        rot_errors.append(quat_angle_xyzw(sample.target_pose_xyzw[3:7], achieved[3:7]))

    successful = len(success_times_ms)
    total_s = float(np.sum(times_ms)) / 1000.0
    success_s = float(np.sum(success_times_ms)) / 1000.0
    return {
        "name": solver.name,
        "success": successful,
        "failures": failures,
        "first_failure": first_failure,
        "latency_all": summarize(times_ms),
        "latency_success": summarize(success_times_ms),
        "latency_failure": summarize(failure_times_ms),
        "pos_err": summarize(pos_errors),
        "rot_err": summarize(rot_errors),
        "calls_per_s": (len(times_ms) / total_s) if total_s > 0.0 else math.inf,
        "successful_per_s_wall": (successful / total_s) if total_s > 0.0 else math.inf,
        "successful_per_s_success_only": (successful / success_s) if success_s > 0.0 else math.inf,
    }


def print_result(result: Dict[str, object], total: int) -> None:
    success = int(result["success"])
    failures = int(result["failures"])
    success_rate = success / max(1, total)
    print(
        f"{result['name']}: success={success}/{total} ({success_rate:.3f}) "
        f"failures={failures} first_failure={result['first_failure']}"
    )
    print(f"  latency_all(ms):     {fmt_ms(result['latency_all'])}")
    print(f"  latency_success(ms): {fmt_ms(result['latency_success'])}")
    if failures:
        print(f"  latency_failure(ms): {fmt_ms(result['latency_failure'])}")
    print(
        "  frequency: "
        f"all_calls={float(result['calls_per_s']):.1f} Hz "
        f"successful_wall={float(result['successful_per_s_wall']):.1f} Hz "
        f"success_only={float(result['successful_per_s_success_only']):.1f} Hz"
    )
    print(f"  pos_err(m):          {fmt_err(result['pos_err'], digits=9)}")
    print(f"  rot_err(rad):        {fmt_err(result['rot_err'], digits=9)}")


def main() -> int:
    args = parse_args()
    urdf = Path(args.urdf).expanduser().resolve()
    verifier = Z1PinocchioIK(urdf, ee_link=args.ee_link)
    pin = verifier
    rtb = Z1RtbKinematics(urdf, ee_link=args.ee_link)
    tracik = Z1TracIKKinematics(
        urdf,
        ee_link=args.ee_link,
        timeout=float(args.tracik_timeout),
        epsilon=float(args.tracik_epsilon),
        solver_type=args.tracik_solver_type,
    )

    mask = [1.0, 1.0, 1.0, float(args.rot_weight), float(args.rot_weight), float(args.rot_weight)]

    solvers = [
        Solver(
            "pinocchio",
            lambda target_xyzw, seed_q, rng: pin.ik(
                target_xyzw,
                seed_joints=seed_q,
                position_only=False,
                max_iter=int(args.pin_max_iter),
                max_restarts=int(args.pin_restarts),
                dt=float(args.pin_dt),
                damping=float(args.pin_damping),
                threshold=float(args.threshold),
                rotation_weight=float(args.rot_weight),
                rng=rng,
            ),
        ),
        Solver(
            "rtb",
            lambda target_xyzw, seed_q, rng: rtb.solve_ik(
                pose_xyzw_to_wxyz(target_xyzw),
                seed_joints=seed_q,
                pose_constraint=mask,
                threshold=float(args.threshold),
                ilimit=int(args.rtb_ilimit),
            ),
        ),
        Solver(
            "tracik",
            lambda target_xyzw, seed_q, rng: tracik.solve_ik(
                pose_xyzw_to_wxyz(target_xyzw),
                seed_joints=seed_q,
                threshold=float(args.threshold),
                max_restarts=max(1, int(args.tracik_restarts)),
                strict_seed=not bool(args.tracik_allow_restarts),
                position_only=False,
                rng=rng,
            ),
        ),
    ]

    print("Z1 IK solver comparison on reachable 6D targets")
    print(f"urdf={urdf}")
    print(
        f"samples={int(args.samples)} warmup={int(args.warmup)} seed={int(args.seed)} "
        f"seed_mode={args.seed_mode} global_targets={bool(args.global_targets)}"
    )
    print(
        f"rot_weight={float(args.rot_weight):.3f} threshold={float(args.threshold):.1e} "
        f"pin_restarts={int(args.pin_restarts)} rtb_ilimit={int(args.rtb_ilimit)} "
        f"tracik_timeout={float(args.tracik_timeout):.4f}s tracik_solver={args.tracik_solver_type}"
    )

    total_count = max(0, int(args.warmup)) + max(1, int(args.samples))
    for delta in args.target_delta:
        rng = np.random.default_rng(int(args.seed))
        samples = sample_reachable_targets(verifier, args, rng, float(delta), total_count)
        print()
        print(f"target_delta={float(delta):.3f}")
        for solver_index, solver in enumerate(solvers):
            result = run_solver(
                solver,
                verifier,
                samples,
                warmup=int(args.warmup),
                seed=int(args.seed) + 1009 * solver_index,
            )
            print_result(result, total=int(args.samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
