#!/usr/bin/env python3
"""Compare Piper Pinocchio, Robotics Toolbox, and TRAC-IK on reachable 6D IK targets."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3, UnitQuaternion
from scipy.spatial.transform import Rotation as R
from trac_ik import TracIK


SCRIPT_DIR = Path(__file__).resolve().parent
IK_SOLVERS_DIR = SCRIPT_DIR / "ik_solvers"
if str(IK_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(IK_SOLVERS_DIR))

from piper_pinocchino import (
    DEFAULT_BASE_LINK,
    DEFAULT_EE_LINK,
    DEFAULT_PIPER_URDF,
    PiperPinocchioIK,
    normalize_quat_wxyz,
    quat_to_rot_wxyz,
)


def rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(np.asarray(rot, dtype=np.float64).reshape(3, 3)).as_quat()
    return normalize_quat_wxyz([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def quat_angle_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = normalize_quat_wxyz(lhs)
    rhs = normalize_quat_wxyz(rhs)
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


def fmt_err(stats: Dict[str, float], digits: int = 9) -> str:
    if math.isnan(stats["mean"]):
        return "no samples"
    return (
        f"mean={stats['mean']:.{digits}f} p50={stats['p50']:.{digits}f} "
        f"p95={stats['p95']:.{digits}f} p99={stats['p99']:.{digits}f} "
        f"max={stats['max']:.{digits}f}"
    )


class PiperRtbIK:
    def __init__(self, urdf: Path, base_link: str, ee_link: str):
        self.urdf_path = str(Path(urdf).expanduser().resolve())
        self.base_link = str(base_link)
        self.ee_link = str(ee_link)
        self.robot = rtb.ERobot.URDF(self.urdf_path)
        self.ets = self.robot.ets(start=self.base_link, end=self.ee_link)
        self.n = int(self.ets.n)
        if self.n != 6:
            print(f"Warning: expected 6 Piper arm joints on chain, got ets.n={self.n}")
        if self.robot.qlim is None:
            self.lower = np.full(self.n, -np.pi, dtype=np.float64)
            self.upper = np.full(self.n, np.pi, dtype=np.float64)
        else:
            qlim = np.asarray(self.robot.qlim, dtype=np.float64)
            self.lower = qlim[0, : self.n].copy()
            self.upper = qlim[1, : self.n].copy()

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] > self.n:
            q = q[: self.n]
        elif q.shape[0] < self.n:
            q = np.pad(q, (0, self.n - q.shape[0]), mode="constant")
        return np.clip(q, self.lower, self.upper)

    def solve_fk(self, q: np.ndarray) -> np.ndarray:
        transform = self.ets.fkine(self._clip_q(q))
        quat = UnitQuaternion(transform.R).A
        return np.concatenate([np.asarray(transform.t, dtype=np.float64), normalize_quat_wxyz(quat)])

    def solve_ik(
        self,
        target_pose: np.ndarray,
        seed_joints: np.ndarray,
        mask: List[float],
        threshold: float,
        ilimit: int,
    ) -> Optional[np.ndarray]:
        pose = np.asarray(target_pose, dtype=np.float64).reshape(7)
        goal = SE3(pose[:3]) * UnitQuaternion(normalize_quat_wxyz(pose[3:7])).SE3()
        sol = self.robot.ikine_LM(
            goal,
            q0=self._clip_q(seed_joints),
            mask=np.asarray(mask, dtype=np.float64),
            end=self.ee_link,
            tol=float(threshold),
            ilimit=int(ilimit),
            joint_limits=True,
        )
        if sol.success:
            return np.asarray(sol.q, dtype=np.float64)[: self.n]
        return None


class PiperTracIK:
    def __init__(self, urdf: Path, base_link: str, ee_link: str, timeout: float, epsilon: float, solver_type: str):
        self.urdf_path = str(Path(urdf).expanduser().resolve())
        self.base_link = str(base_link)
        self.ee_link = str(ee_link)
        self.solver = TracIK(
            base_link_name=self.base_link,
            tip_link_name=self.ee_link,
            urdf_path=self.urdf_path,
            timeout=float(timeout),
            epsilon=float(epsilon),
            solver_type=str(solver_type),
        )
        self.n = int(self.solver.dof)
        self.lower = np.asarray(self.solver.joint_limits[0], dtype=np.float64)
        self.upper = np.asarray(self.solver.joint_limits[1], dtype=np.float64)

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] > self.n:
            q = q[: self.n]
        elif q.shape[0] < self.n:
            q = np.pad(q, (0, self.n - q.shape[0]), mode="constant")
        return np.clip(q, self.lower, self.upper)

    def solve_fk(self, q: np.ndarray) -> np.ndarray:
        position, rotation_matrix = self.solver.fk(self._clip_q(q))
        return np.concatenate([np.asarray(position, dtype=np.float64), rot_to_quat_wxyz(rotation_matrix)])

    def solve_ik(self, target_pose: np.ndarray, seed_joints: np.ndarray) -> Optional[np.ndarray]:
        pose = np.asarray(target_pose, dtype=np.float64).reshape(7)
        try:
            result = self.solver.ik(
                tgt_pos=pose[:3],
                tgt_rot=quat_to_rot_wxyz(pose[3:7]),
                seed_jnt_values=self._clip_q(seed_joints),
            )
        except Exception:
            return None
        if result is None:
            return None
        result = np.asarray(result, dtype=np.float64).reshape(-1)
        if result.shape[0] != self.n or not np.all(np.isfinite(result)):
            return None
        return self._clip_q(result)


@dataclass
class Sample:
    seed_q: np.ndarray
    target_q: np.ndarray
    target_pose: np.ndarray


@dataclass
class Solver:
    name: str
    solve: Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_PIPER_URDF))
    parser.add_argument("--base_link", type=str, default=DEFAULT_BASE_LINK)
    parser.add_argument("--ee_link", type=str, default=DEFAULT_EE_LINK)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global_targets", action="store_true")
    parser.add_argument("--target_delta", type=float, default=0.25)
    parser.add_argument("--seed_mode", choices=("home", "middle", "zero", "target", "chain"), default="home")
    parser.add_argument("--home_q", type=float, nargs=6, default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    parser.add_argument("--rot_weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=1.0e-4)
    parser.add_argument("--pin_max_iter", type=int, default=80)
    parser.add_argument("--pin_restarts", type=int, default=1)
    parser.add_argument("--pin_allow_restarts", action="store_true")
    parser.add_argument("--pin_dt", type=float, default=0.4)
    parser.add_argument("--pin_damping", type=float, default=1.0e-4)
    parser.add_argument("--rtb_ilimit", type=int, default=80)
    parser.add_argument("--tracik_timeout", type=float, default=0.005)
    parser.add_argument("--tracik_epsilon", type=float, default=1.0e-5)
    parser.add_argument("--tracik_solver_type", choices=("Speed", "Distance", "Manip1", "Manip2"), default="Speed")
    return parser.parse_args()


def sample_reachable_targets(
    verifier: PiperPinocchioIK,
    args: argparse.Namespace,
    rng: np.random.Generator,
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
            delta = rng.uniform(-float(args.target_delta), float(args.target_delta), size=home.shape[0])
            target_q = np.clip(chain_seed + delta, lower + margin, upper - margin)

        if args.seed_mode == "target":
            seed_q = target_q.copy()
        elif args.seed_mode == "middle":
            seed_q = middle.copy()
        elif args.seed_mode == "zero":
            seed_q = zero.copy()
        elif args.seed_mode == "chain":
            seed_q = chain_seed.copy()
        else:
            seed_q = home.copy()

        samples.append(Sample(seed_q=seed_q, target_q=target_q, target_pose=verifier.solve_fk(target_q)))
        chain_seed = target_q
    return samples


def run_solver(solver: Solver, verifier: PiperPinocchioIK, samples: List[Sample], warmup: int) -> Dict[str, object]:
    times_ms: List[float] = []
    success_times_ms: List[float] = []
    failure_times_ms: List[float] = []
    pos_errors: List[float] = []
    rot_errors: List[float] = []
    failures = 0
    first_failure = None

    for index, sample in enumerate(samples):
        start_ns = time.perf_counter_ns()
        solution = solver.solve(sample.target_pose, sample.seed_q)
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
        achieved = verifier.solve_fk(q)
        pos_errors.append(float(np.linalg.norm(achieved[:3] - sample.target_pose[:3])))
        rot_errors.append(quat_angle_wxyz(sample.target_pose[3:7], achieved[3:7]))

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
    print(
        f"{result['name']}: success={success}/{total} ({success / max(1, total):.3f}) "
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
    print(f"  pos_err(m):          {fmt_err(result['pos_err'])}")
    print(f"  rot_err(rad):        {fmt_err(result['rot_err'])}")


def main() -> int:
    args = parse_args()
    urdf = Path(args.urdf).expanduser().resolve()
    verifier = PiperPinocchioIK(urdf, base_link=args.base_link, ee_link=args.ee_link)
    rtb_ik = PiperRtbIK(urdf, base_link=args.base_link, ee_link=args.ee_link)
    tracik_ik = PiperTracIK(
        urdf,
        base_link=args.base_link,
        ee_link=args.ee_link,
        timeout=float(args.tracik_timeout),
        epsilon=float(args.tracik_epsilon),
        solver_type=args.tracik_solver_type,
    )
    mask = [1.0, 1.0, 1.0, float(args.rot_weight), float(args.rot_weight), float(args.rot_weight)]
    solvers = [
        Solver(
            "pinocchio",
            lambda target_pose, seed_q: verifier.solve_ik(
                target_pose,
                seed_joints=seed_q,
                pose_constraint=mask,
                max_restarts=int(args.pin_restarts),
                max_iter=int(args.pin_max_iter),
                dt=float(args.pin_dt),
                damp=float(args.pin_damping),
                threshold=float(args.threshold),
                strict_seed=not bool(args.pin_allow_restarts),
            ),
        ),
        Solver(
            "rtb",
            lambda target_pose, seed_q: rtb_ik.solve_ik(
                target_pose,
                seed_joints=seed_q,
                mask=mask,
                threshold=float(args.threshold),
                ilimit=int(args.rtb_ilimit),
            ),
        ),
        Solver("tracik", lambda target_pose, seed_q: tracik_ik.solve_ik(target_pose, seed_q)),
    ]

    rng = np.random.default_rng(int(args.seed))
    total_count = max(0, int(args.warmup)) + max(1, int(args.samples))
    samples = sample_reachable_targets(verifier, args, rng, total_count)

    print("Piper IK solver comparison on reachable 6D targets")
    print(f"urdf={urdf}")
    print(f"base_link={args.base_link} ee_link={args.ee_link}")
    print(
        f"samples={int(args.samples)} warmup={int(args.warmup)} seed={int(args.seed)} "
        f"seed_mode={args.seed_mode} global_targets={bool(args.global_targets)}"
    )
    print(
        f"home_q={np.array2string(np.asarray(args.home_q, dtype=np.float64), precision=3, suppress_small=True)} "
        f"rot_weight={float(args.rot_weight):.3f} threshold={float(args.threshold):.1e}"
    )
    print(
        f"pin: max_iter={int(args.pin_max_iter)} restarts={int(args.pin_restarts)} "
        f"allow_restarts={bool(args.pin_allow_restarts)} "
        f"dt={float(args.pin_dt):.3f} damping={float(args.pin_damping):.1e}; "
        f"rtb_ilimit={int(args.rtb_ilimit)}; "
        f"tracik_timeout={float(args.tracik_timeout):.4f}s solver={args.tracik_solver_type}"
    )

    for solver in solvers:
        result = run_solver(solver, verifier, samples, warmup=int(args.warmup))
        print()
        print_result(result, total=int(args.samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
