#!/usr/bin/env python3
"""Joint-space trajectory smoothing for IK waypoint sequences.

Given IK solutions q[0], q[1], ... this script builds a time-parameterized
joint trajectory and samples position, velocity, acceleration, and jerk.

Default method:
    quintic  - global piecewise 5th-order spline, C4 at waypoints
               and therefore continuous velocity/acceleration/jerk.

Optional method:
    quintic_hermite - local 5th-order Hermite segments, C2 at waypoints.
    septic          - piecewise 7th-order polynomial, C3 at waypoints.

Notes:
    A single quintic segment has smooth position/velocity/acceleration/jerk
    inside the segment.  Multi-segment jerk continuity needs a global spline
    solve, which is what --method quintic now does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _as_2d_waypoints(waypoints: np.ndarray) -> np.ndarray:
    q = np.asarray(waypoints, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.ndim != 2:
        raise ValueError(f"waypoints must be [N, dof], got shape {q.shape}")
    if q.shape[0] < 2:
        raise ValueError("need at least two IK waypoint joint vectors")
    if not np.all(np.isfinite(q)):
        raise ValueError("waypoints contain non-finite values")
    return q


def parse_waypoints_text(text: str) -> np.ndarray:
    """Parse 'q0,q1,...; q0,q1,...' into [N, dof]."""
    rows = []
    for row_text in str(text).split(";"):
        row_text = row_text.strip()
        if not row_text:
            continue
        rows.append([float(v.strip()) for v in row_text.split(",") if v.strip()])
    if not rows:
        raise ValueError("empty --waypoints")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all waypoint rows must have the same length")
    return _as_2d_waypoints(np.asarray(rows, dtype=np.float64))


def load_waypoints(path: Path, key: str = "q") -> np.ndarray:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        return _as_2d_waypoints(np.load(path))
    if path.suffix == ".npz":
        data = np.load(path)
        if key not in data:
            raise KeyError(f"key {key!r} not found in {path}; keys={list(data.keys())}")
        return _as_2d_waypoints(data[key])
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if key not in data:
                raise KeyError(f"key {key!r} not found in {path}; keys={list(data.keys())}")
            data = data[key]
        return _as_2d_waypoints(np.asarray(data, dtype=np.float64))
    raise ValueError(f"unsupported input suffix {path.suffix!r}; use .npy, .npz, or .json")


def save_trajectory(path: Path, result: Dict[str, np.ndarray]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        np.savez(path, **result)
        return
    if path.suffix == ".npy":
        np.save(path, result["q"])
        return
    if path.suffix == ".json":
        serializable = {k: np.asarray(v).tolist() for k, v in result.items()}
        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        return
    raise ValueError(f"unsupported output suffix {path.suffix!r}; use .npz, .npy, or .json")


def build_segment_times(
    waypoints: np.ndarray,
    segment_time: float,
    max_vel: Optional[float] = None,
    min_segment_time: float = 1.0e-3,
) -> np.ndarray:
    """Return monotonic waypoint times [N]."""
    q = _as_2d_waypoints(waypoints)
    durations = np.full(q.shape[0] - 1, float(segment_time), dtype=np.float64)
    if max_vel is not None and float(max_vel) > 0.0:
        dq_inf = np.max(np.abs(np.diff(q, axis=0)), axis=1)
        durations = np.maximum(durations, dq_inf / float(max_vel))
    durations = np.maximum(durations, float(min_segment_time))
    return np.concatenate([[0.0], np.cumsum(durations)])


def estimate_waypoint_derivatives(
    waypoints: np.ndarray,
    times: np.ndarray,
    endpoint_mode: str = "zero",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate qd/qdd/qddd at waypoints.

    For quintic, qd and qdd are used.  For septic, qddd is used as well.
    Endpoint mode 'zero' starts/stops from rest.
    """
    q = _as_2d_waypoints(waypoints)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if t.shape[0] != q.shape[0]:
        raise ValueError(f"times length {t.shape[0]} != waypoint count {q.shape[0]}")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")

    n, dof = q.shape
    qd = np.zeros((n, dof), dtype=np.float64)
    qdd = np.zeros((n, dof), dtype=np.float64)
    qddd = np.zeros((n, dof), dtype=np.float64)
    slopes = np.diff(q, axis=0) / np.diff(t)[:, None]

    if n > 2:
        for i in range(1, n - 1):
            dt_prev = t[i] - t[i - 1]
            dt_next = t[i + 1] - t[i]
            qd[i] = (dt_next * slopes[i - 1] + dt_prev * slopes[i]) / (dt_prev + dt_next)
            qdd[i] = 2.0 * (slopes[i] - slopes[i - 1]) / (dt_prev + dt_next)

    if endpoint_mode == "finite_difference":
        qd[0] = slopes[0]
        qd[-1] = slopes[-1]
        if n > 2:
            qdd[0] = qdd[1]
            qdd[-1] = qdd[-2]
    elif endpoint_mode != "zero":
        raise ValueError(f"unknown endpoint_mode {endpoint_mode!r}")

    # Keep jerk zero at knots by default.  With septic this gives C3 knots.
    qddd[:] = 0.0
    return qd, qdd, qddd


def polynomial_coefficients(
    q0: np.ndarray,
    q1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    j0: Optional[np.ndarray],
    j1: Optional[np.ndarray],
    duration: float,
    method: str,
) -> np.ndarray:
    """Return coefficients c[k, dof] for q(t)=sum_k c[k] t^k."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    T = float(duration)
    if T <= 0.0:
        raise ValueError("duration must be positive")

    method = str(method).lower()
    if method in ("quintic_hermite", "quintic-local", "quintic_local"):
        coeff = np.zeros((6, q0.shape[0]), dtype=np.float64)
        coeff[0] = q0
        coeff[1] = v0
        coeff[2] = 0.5 * a0
        rhs = np.vstack(
            [
                q1 - (coeff[0] + coeff[1] * T + coeff[2] * T**2),
                v1 - (coeff[1] + 2.0 * coeff[2] * T),
                a1 - (2.0 * coeff[2]),
            ]
        )
        mat = np.asarray(
            [
                [T**3, T**4, T**5],
                [3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
                [6.0 * T, 12.0 * T**2, 20.0 * T**3],
            ],
            dtype=np.float64,
        )
        coeff[3:6] = np.linalg.solve(mat, rhs)
        return coeff

    if method == "septic":
        j0 = np.zeros_like(q0) if j0 is None else np.asarray(j0, dtype=np.float64)
        j1 = np.zeros_like(q0) if j1 is None else np.asarray(j1, dtype=np.float64)
        coeff = np.zeros((8, q0.shape[0]), dtype=np.float64)
        coeff[0] = q0
        coeff[1] = v0
        coeff[2] = 0.5 * a0
        coeff[3] = j0 / 6.0
        rhs = np.vstack(
            [
                q1 - sum(coeff[k] * T**k for k in range(4)),
                v1 - sum(k * coeff[k] * T ** (k - 1) for k in range(1, 4)),
                a1 - sum(k * (k - 1) * coeff[k] * T ** (k - 2) for k in range(2, 4)),
                j1 - sum(k * (k - 1) * (k - 2) * coeff[k] * T ** (k - 3) for k in range(3, 4)),
            ]
        )
        mat = np.asarray(
            [
                [T**4, T**5, T**6, T**7],
                [4.0 * T**3, 5.0 * T**4, 6.0 * T**5, 7.0 * T**6],
                [12.0 * T**2, 20.0 * T**3, 30.0 * T**4, 42.0 * T**5],
                [24.0 * T, 60.0 * T**2, 120.0 * T**3, 210.0 * T**4],
            ],
            dtype=np.float64,
        )
        coeff[4:8] = np.linalg.solve(mat, rhs)
        return coeff

    raise ValueError(f"unknown method {method!r}; expected quintic or septic")


def _poly_derivative_basis(degree: int, derivative: int, tau: float) -> np.ndarray:
    row = np.zeros(degree + 1, dtype=np.float64)
    for k in range(derivative, degree + 1):
        factor = 1.0
        for d in range(derivative):
            factor *= float(k - d)
        row[k] = factor * float(tau) ** (k - derivative)
    return row


def global_quintic_c3_coefficients(
    waypoints: np.ndarray,
    times: np.ndarray,
    endpoint_mode: str = "zero",
) -> np.ndarray:
    """Solve a global piecewise-quintic spline with C4 knot continuity.

    C4 continuity is a convenient square-system closure for quintic segments;
    it implies the requested C3 jerk continuity.
    """
    q = _as_2d_waypoints(waypoints)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if t.shape[0] != q.shape[0]:
        raise ValueError(f"times length {t.shape[0]} != waypoint count {q.shape[0]}")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if endpoint_mode not in ("zero", "finite_difference"):
        raise ValueError(f"unknown endpoint_mode {endpoint_mode!r}")

    num_segments = q.shape[0] - 1
    dof = q.shape[1]
    degree = 5
    unknowns = (degree + 1) * num_segments
    equation_columns = []
    equation_values = []
    rhs = []

    def add_row(segment: int, derivative: int, tau: float, value: np.ndarray) -> None:
        start = segment * (degree + 1)
        equation_columns.append(np.arange(start, start + degree + 1, dtype=np.int64))
        equation_values.append(_poly_derivative_basis(degree, derivative, tau))
        rhs.append(np.asarray(value, dtype=np.float64).reshape(dof))

    def add_continuity(prev_segment: int, next_segment: int, derivative: int, tau_prev: float) -> None:
        prev_start = prev_segment * (degree + 1)
        next_start = next_segment * (degree + 1)
        equation_columns.append(
            np.concatenate(
                [
                    np.arange(prev_start, prev_start + degree + 1, dtype=np.int64),
                    np.arange(next_start, next_start + degree + 1, dtype=np.int64),
                ]
            )
        )
        equation_values.append(
            np.concatenate(
                [
                    _poly_derivative_basis(degree, derivative, tau_prev),
                    -_poly_derivative_basis(degree, derivative, 0.0),
                ]
            )
        )
        rhs.append(np.zeros(dof, dtype=np.float64))

    # Position interpolation for each segment.
    for i in range(num_segments):
        duration = float(t[i + 1] - t[i])
        add_row(i, 0, 0.0, q[i])
        add_row(i, 0, duration, q[i + 1])

    # C1/C2/C3/C4 continuity at interior knots.
    for i in range(1, num_segments):
        duration_prev = float(t[i] - t[i - 1])
        for derivative in range(1, 5):
            add_continuity(i - 1, i, derivative, duration_prev)

    if endpoint_mode == "finite_difference":
        endpoint_qd, endpoint_qdd, _ = estimate_waypoint_derivatives(q, t, endpoint_mode=endpoint_mode)
        start_velocity = endpoint_qd[0]
        start_acceleration = endpoint_qdd[0]
        end_velocity = endpoint_qd[-1]
        end_acceleration = endpoint_qdd[-1]
    else:
        start_velocity = np.zeros(dof, dtype=np.float64)
        start_acceleration = np.zeros(dof, dtype=np.float64)
        end_velocity = np.zeros(dof, dtype=np.float64)
        end_acceleration = np.zeros(dof, dtype=np.float64)

    add_row(0, 1, 0.0, start_velocity)
    add_row(0, 2, 0.0, start_acceleration)
    last_duration = float(t[-1] - t[-2])
    add_row(num_segments - 1, 1, last_duration, end_velocity)
    add_row(num_segments - 1, 2, last_duration, end_acceleration)

    B = np.vstack(rhs)
    if len(equation_columns) != unknowns:
        raise RuntimeError(f"quintic system is not square: rows={len(equation_columns)} unknowns={unknowns}")

    if unknowns >= 256:
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import spsolve
        except ImportError as exc:
            raise RuntimeError(
                "Large global quintic trajectories require scipy for sparse solving"
            ) from exc

        row_indices = np.concatenate(
            [np.full(columns.shape, row, dtype=np.int64) for row, columns in enumerate(equation_columns)]
        )
        column_indices = np.concatenate(equation_columns)
        values = np.concatenate(equation_values)
        A = coo_matrix((values, (row_indices, column_indices)), shape=(unknowns, unknowns)).tocsr()
        coeff_flat = np.asarray(spsolve(A, B), dtype=np.float64)
    else:
        A = np.zeros((unknowns, unknowns), dtype=np.float64)
        for row, (columns, values) in enumerate(zip(equation_columns, equation_values)):
            A[row, columns] += values
        coeff_flat = np.linalg.solve(A, B)
    return coeff_flat.reshape(num_segments, degree + 1, dof)


def waypoint_derivatives_from_coefficients(coeffs: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    num_segments = coeffs.shape[0]
    dof = coeffs.shape[2]
    qd = np.zeros((num_segments + 1, dof), dtype=np.float64)
    qdd = np.zeros_like(qd)
    qddd = np.zeros_like(qd)
    first = evaluate_polynomial(coeffs[0], np.asarray([0.0]))
    qd[0], qdd[0], qddd[0] = first[1][0], first[2][0], first[3][0]
    for i in range(num_segments):
        duration = float(t[i + 1] - t[i])
        values = evaluate_polynomial(coeffs[i], np.asarray([duration]))
        qd[i + 1], qdd[i + 1], qddd[i + 1] = values[1][0], values[2][0], values[3][0]
    return qd, qdd, qddd


def evaluate_polynomial(coeff: np.ndarray, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coeff = np.asarray(coeff, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    dof = coeff.shape[1]
    q = np.zeros((tau.shape[0], dof), dtype=np.float64)
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    qddd = np.zeros_like(q)
    for k in range(coeff.shape[0]):
        q += coeff[k] * tau[:, None] ** k
        if k >= 1:
            qd += k * coeff[k] * tau[:, None] ** (k - 1)
        if k >= 2:
            qdd += k * (k - 1) * coeff[k] * tau[:, None] ** (k - 2)
        if k >= 3:
            qddd += k * (k - 1) * (k - 2) * coeff[k] * tau[:, None] ** (k - 3)
    return q, qd, qdd, qddd


def smooth_joint_waypoints(
    waypoints: np.ndarray,
    dt: float = 0.02,
    segment_time: float = 1.0,
    times: Optional[np.ndarray] = None,
    method: str = "quintic",
    endpoint_mode: str = "zero",
    max_vel: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Smooth IK joint waypoints into sampled q/qd/qdd/qddd trajectory."""
    q_wp = _as_2d_waypoints(waypoints)
    if times is None:
        t_wp = build_segment_times(q_wp, segment_time=segment_time, max_vel=max_vel)
    else:
        t_wp = np.asarray(times, dtype=np.float64).reshape(-1)
    method = str(method).lower()
    if method == "quintic":
        global_coeffs = global_quintic_c3_coefficients(q_wp, t_wp, endpoint_mode=endpoint_mode)
        qd_wp, qdd_wp, qddd_wp = waypoint_derivatives_from_coefficients(global_coeffs, t_wp)
    else:
        global_coeffs = None
        qd_wp, qdd_wp, qddd_wp = estimate_waypoint_derivatives(q_wp, t_wp, endpoint_mode=endpoint_mode)

    sample_t = []
    sample_q = []
    sample_qd = []
    sample_qdd = []
    sample_qddd = []
    segment_index = []
    coeffs = []
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    for i in range(q_wp.shape[0] - 1):
        T = float(t_wp[i + 1] - t_wp[i])
        if global_coeffs is not None:
            coeff = global_coeffs[i]
        else:
            coeff = polynomial_coefficients(
                q_wp[i],
                q_wp[i + 1],
                qd_wp[i],
                qd_wp[i + 1],
                qdd_wp[i],
                qdd_wp[i + 1],
                qddd_wp[i],
                qddd_wp[i + 1],
                T,
                method=method,
            )
        coeffs.append(coeff)
        # Avoid creating an extra sample when T/dt is an integer perturbed by
        # floating-point roundoff (for example 0.04 / 0.02).
        count = max(2, int(np.ceil(T / dt - 1.0e-12)) + 1)
        tau = np.linspace(0.0, T, count)
        if i > 0:
            tau = tau[1:]
        q, qd, qdd, qddd = evaluate_polynomial(coeff, tau)
        sample_t.append(t_wp[i] + tau)
        sample_q.append(q)
        sample_qd.append(qd)
        sample_qdd.append(qdd)
        sample_qddd.append(qddd)
        segment_index.append(np.full(tau.shape[0], i, dtype=np.int64))

    return {
        "t": np.concatenate(sample_t),
        "q": np.vstack(sample_q),
        "qd": np.vstack(sample_qd),
        "qdd": np.vstack(sample_qdd),
        "qddd": np.vstack(sample_qddd),
        "waypoint_t": t_wp,
        "waypoint_q": q_wp,
        "waypoint_qd": qd_wp,
        "waypoint_qdd": qdd_wp,
        "waypoint_qddd": qddd_wp,
        "segment_index": np.concatenate(segment_index),
        "coefficients": np.asarray(coeffs, dtype=np.float64),
    }


def continuity_report(result: Dict[str, np.ndarray]) -> Dict[str, float]:
    coeffs = np.asarray(result["coefficients"], dtype=np.float64)
    t_wp = np.asarray(result["waypoint_t"], dtype=np.float64)
    jumps: Dict[str, float] = {"q": 0.0, "qd": 0.0, "qdd": 0.0, "qddd": 0.0}
    for i in range(coeffs.shape[0] - 1):
        T = t_wp[i + 1] - t_wp[i]
        left = evaluate_polynomial(coeffs[i], np.asarray([T]))
        right = evaluate_polynomial(coeffs[i + 1], np.asarray([0.0]))
        for name, li, ri in zip(("q", "qd", "qdd", "qddd"), left, right):
            jumps[name] = max(jumps[name], float(np.max(np.abs(li[0] - ri[0]))))
    return jumps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default="", help="Input .npy/.npz/.json waypoints [N,dof].")
    parser.add_argument("--input_key", type=str, default="q", help="Key for .npz/.json input.")
    parser.add_argument("--waypoints", type=str, default="", help="Inline waypoints: '0,0; 1,0.5; 2,0'.")
    parser.add_argument("--output", type=str, default="", help="Optional output .npz/.npy/.json.")
    parser.add_argument("--method", choices=("quintic", "quintic_hermite", "septic"), default="quintic")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--segment_time", type=float, default=1.0)
    parser.add_argument("--max_vel", type=float, default=0.0, help="Optional per-joint max velocity for segment timing.")
    parser.add_argument("--endpoint_mode", choices=("zero", "finite_difference"), default="zero")
    parser.add_argument("--demo", action="store_true", help="Use a built-in 6-DOF demo waypoint sequence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input:
        waypoints = load_waypoints(Path(args.input), key=args.input_key)
    elif args.waypoints:
        waypoints = parse_waypoints_text(args.waypoints)
    elif args.demo:
        waypoints = np.asarray(
            [
                [0.0, 1.05, -1.45, 0.75, 0.0, 0.0],
                [-0.35, 1.55, -1.10, 0.15, 0.45, -0.30],
                [0.25, 1.20, -1.75, 0.90, -0.35, 0.55],
                [0.0, 1.05, -1.45, 0.75, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    else:
        raise SystemExit("Provide --input, --waypoints, or --demo.")

    max_vel = float(args.max_vel) if float(args.max_vel) > 0.0 else None
    result = smooth_joint_waypoints(
        waypoints,
        dt=float(args.dt),
        segment_time=float(args.segment_time),
        method=args.method,
        endpoint_mode=args.endpoint_mode,
        max_vel=max_vel,
    )
    jumps = continuity_report(result)
    print("Joint trajectory smoothing")
    print(f"method={args.method} dt={float(args.dt):.4f}s segment_time={float(args.segment_time):.4f}s")
    print(f"waypoints={result['waypoint_q'].shape[0]} dof={result['waypoint_q'].shape[1]}")
    print(f"samples={result['q'].shape[0]} duration={result['t'][-1]:.4f}s")
    print(
        "max_abs: "
        f"vel={np.max(np.abs(result['qd'])):.6f} "
        f"acc={np.max(np.abs(result['qdd'])):.6f} "
        f"jerk={np.max(np.abs(result['qddd'])):.6f}"
    )
    print(
        "knot jumps: "
        f"q={jumps['q']:.3e} qd={jumps['qd']:.3e} "
        f"qdd={jumps['qdd']:.3e} qddd={jumps['qddd']:.3e}"
    )
    if args.method == "quintic_hermite" and jumps["qddd"] > 1.0e-8:
        print("note=quintic_hermite is C2 across multiple segments; use --method quintic or septic for C3 jerk continuity.")
    if args.output:
        save_trajectory(Path(args.output), result)
        print(f"wrote {Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
