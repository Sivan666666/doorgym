"""Rollout diagnostics and artifact summaries for door digital-twin runs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _round_list(value: Any, precision: int = 5) -> list[float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return np.round(arr, precision).tolist()


def door_open_degrees(door_pos: Any, door_motion_sign: float) -> float:
    arr = np.asarray([] if door_pos is None else door_pos, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return math.degrees(float(door_motion_sign) * float(arr[0]))


def handle_rotation_degrees(door_pos: Any, handle_lower: float = 0.0) -> float:
    arr = np.asarray([] if door_pos is None else door_pos, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return 0.0
    return math.degrees(abs(float(arr[1]) - float(handle_lower)))


@dataclass
class RolloutReport:
    env_id: int
    door_name: str
    success: bool
    failure_stage: str
    door_open_deg: float
    handle_rotation_deg: float
    ee_handle_dist: float
    ee_tracking_error: float
    base_collision: bool
    body_passed: bool
    camera_available: bool
    handle_unlocked: bool
    steps: int
    final_phase: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    skill_program: dict[str, Any] | None = None
    door_spec: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RolloutTracker:
    """Accumulate low-bandwidth rollout diagnostics without storing full video."""

    def __init__(
        self,
        env_id: int,
        door_name: str,
        door_spec: dict[str, Any] | None = None,
        skill_program: dict[str, Any] | None = None,
        *,
        pass_open_angle_deg: float = 80.0,
        door_motion_sign: float = -1.0,
        handle_lower: float = 0.0,
        handle_unlock_threshold: float = 0.0,
        require_traverse: bool = False,
        camera_required: bool = False,
        save_trace: bool = False,
        base_start: Any = None,
        base_push: Any = None,
    ):
        self.env_id = int(env_id)
        self.door_name = str(door_name)
        self.door_spec = door_spec
        self.skill_program = skill_program
        self.asset_supported = bool((door_spec or {}).get("supported", True))
        self.pass_open_angle_deg = float(pass_open_angle_deg)
        self.door_motion_sign = float(door_motion_sign)
        self.handle_lower = float(handle_lower)
        self.handle_unlock_threshold = float(handle_unlock_threshold)
        self.require_traverse = bool(require_traverse)
        self.camera_required = bool(camera_required)
        self.save_trace = bool(save_trace)
        self.base_start = None if base_start is None else np.asarray(base_start, dtype=np.float32).reshape(2)
        self.base_push = None if base_push is None else np.asarray(base_push, dtype=np.float32).reshape(2)

        self.steps = 0
        self.final_phase = "init"
        self.max_door_open_deg = 0.0
        self.max_handle_rotation_deg = 0.0
        self.max_ee_tracking_error = 0.0
        self.max_pre_push_ee_tracking_error = 0.0
        self.min_ee_handle_dist = math.inf
        self.min_grasp_ee_handle_dist = math.inf
        self.max_push_ee_handle_dist = 0.0
        self.base_collision = False
        self.body_passed = False
        self.camera_available = not self.camera_required
        self.handle_unlocked = False
        self.joint_limit_hit = False
        self.pre_push_joint_limit_hit = False
        self.last_base_xy: list[float] | None = None
        self.last_base_command = {"vx": 0.0, "vyaw": 0.0}
        self.max_abs_base_vx = 0.0
        self.max_abs_base_vyaw = 0.0
        self.last_target_pos: list[float] | None = None
        self.last_ee_pos: list[float] | None = None
        self.last_handle_goal: list[float] | None = None
        self.artifacts: dict[str, Any] = {"keyframes": [], "expert_trajectories": []}
        self.trace: list[dict[str, Any]] = []

    def mark_camera_available(self, available: bool) -> None:
        self.camera_available = bool(self.camera_available or available)

    def add_artifact(self, kind: str, record: dict[str, Any]) -> None:
        if kind == "keyframe":
            self.artifacts.setdefault("keyframes", []).append(dict(record))
        elif kind == "expert_trajectory":
            self.artifacts.setdefault("expert_trajectories", []).append(dict(record))
        elif kind == "base_collision_event":
            self.artifacts.setdefault("base_collision_events", []).append(dict(record))
        else:
            self.artifacts[kind] = record

    def update(
        self,
        *,
        step: int,
        phase: str,
        door_pos: Any,
        handle_goal: Any,
        target_pos: Any,
        ee_pos: Any,
        ee_tracking_error: float,
        base_xy: Any,
        base_collision: bool,
        camera_available: bool,
        base_vx: float = 0.0,
        base_vyaw: float = 0.0,
        dof_positions: Any = None,
        lower: Any = None,
        upper: Any = None,
    ) -> None:
        self.steps = max(self.steps, int(step) + 1)
        self.final_phase = str(phase)
        self.base_collision = bool(self.base_collision or base_collision)
        self.mark_camera_available(camera_available)
        door_deg = door_open_degrees(door_pos, self.door_motion_sign)
        self.max_door_open_deg = max(self.max_door_open_deg, door_deg)
        self.max_handle_rotation_deg = max(
            self.max_handle_rotation_deg,
            handle_rotation_degrees(door_pos, self.handle_lower),
        )
        arr = np.asarray([] if door_pos is None else door_pos, dtype=np.float32).reshape(-1)
        if arr.size >= 2:
            self.handle_unlocked = bool(
                self.handle_unlocked or float(arr[1] - self.handle_lower) >= self.handle_unlock_threshold
            )
        self.max_ee_tracking_error = max(self.max_ee_tracking_error, float(ee_tracking_error))
        pre_push_phase = phase in (
            "walk",
            "initial_hold",
            "grasp",
            "grasp_hold",
            "close_gripper",
            "rotate_handle",
        )
        if pre_push_phase:
            self.max_pre_push_ee_tracking_error = max(
                self.max_pre_push_ee_tracking_error,
                float(ee_tracking_error),
            )

        handle = np.asarray(handle_goal, dtype=np.float32).reshape(3) if handle_goal is not None else None
        ee = np.asarray(ee_pos, dtype=np.float32).reshape(3) if ee_pos is not None else None
        if handle is not None and ee is not None:
            dist = float(np.linalg.norm(ee - handle))
            self.min_ee_handle_dist = min(self.min_ee_handle_dist, dist)
            if phase in ("grasp", "grasp_hold", "close_gripper", "rotate_handle"):
                self.min_grasp_ee_handle_dist = min(self.min_grasp_ee_handle_dist, dist)
            if phase == "push_door":
                self.max_push_ee_handle_dist = max(self.max_push_ee_handle_dist, dist)

        base = np.asarray(base_xy, dtype=np.float32).reshape(2) if base_xy is not None else None
        if base is not None:
            self.last_base_xy = _round_list(base)
            if self.base_start is not None and self.base_push is not None:
                path = self.base_push - self.base_start
                denom = max(float(np.linalg.norm(path)), 1.0e-6)
                progress = float(np.dot(base - self.base_start, path / denom))
                target = float(np.linalg.norm(path))
                self.body_passed = bool(self.body_passed or (target <= 1.0e-5 or progress >= target - 0.05))
        self.last_base_command = {"vx": float(base_vx), "vyaw": float(base_vyaw)}
        self.max_abs_base_vx = max(self.max_abs_base_vx, abs(float(base_vx)))
        self.max_abs_base_vyaw = max(self.max_abs_base_vyaw, abs(float(base_vyaw)))
        self.last_target_pos = None if target_pos is None else _round_list(target_pos)
        self.last_ee_pos = None if ee_pos is None else _round_list(ee_pos)
        self.last_handle_goal = None if handle_goal is None else _round_list(handle_goal)

        if dof_positions is not None and lower is not None and upper is not None:
            q = np.asarray(dof_positions, dtype=np.float32).reshape(-1)
            lo = np.asarray(lower, dtype=np.float32).reshape(-1)
            hi = np.asarray(upper, dtype=np.float32).reshape(-1)
            n = min(q.size, lo.size, hi.size)
            if n > 0:
                span = np.maximum(hi[:n] - lo[:n], 1.0e-6)
                margin = np.minimum(q[:n] - lo[:n], hi[:n] - q[:n])
                joint_limit_now = bool(np.any(margin / span < 0.015))
                self.joint_limit_hit = bool(self.joint_limit_hit or joint_limit_now)
                if pre_push_phase:
                    self.pre_push_joint_limit_hit = bool(
                        self.pre_push_joint_limit_hit or joint_limit_now
                    )

        if self.save_trace and (not self.trace or self.trace[-1]["phase"] != phase or int(step) % 25 == 0):
            self.trace.append(
                {
                    "step": int(step),
                    "phase": str(phase),
                    "door_open_deg": float(self.max_door_open_deg),
                    "handle_rotation_deg": float(self.max_handle_rotation_deg),
                    "ee_handle_dist": (
                        None if math.isinf(self.min_ee_handle_dist) else float(self.min_ee_handle_dist)
                    ),
                    "ee_tracking_error": float(ee_tracking_error),
                    "base_xy": self.last_base_xy,
                    "base_command": dict(self.last_base_command),
                    "target_pos": self.last_target_pos,
                    "ee_pos": self.last_ee_pos,
                    "handle_goal": self.last_handle_goal,
                }
            )

    def _failure_stage(self, success: bool) -> str:
        if success:
            return ""
        if not self.asset_supported:
            return "asset_invalid"
        if self.camera_required and not self.camera_available:
            return "camera_unavailable"
        if self.base_collision:
            return "base_collision"
        if self.max_pre_push_ee_tracking_error > 0.12 or self.pre_push_joint_limit_hit:
            return "arm_joint_limit_or_ik_bad"
        if self.min_grasp_ee_handle_dist > 0.09:
            return "grasp_miss"
        if not self.handle_unlocked:
            return "handle_not_unlocked"
        if self.max_ee_tracking_error > 0.12 or self.joint_limit_hit:
            return "arm_joint_limit_or_ik_bad"
        if self.max_push_ee_handle_dist > 0.20:
            return "contact_lost"
        if self.max_door_open_deg < self.pass_open_angle_deg:
            return "door_push_insufficient"
        if self.require_traverse and not self.body_passed:
            return "body_blocked"
        return "timeout"

    def finalize(self) -> RolloutReport:
        body_ok = (not self.require_traverse) or self.body_passed
        camera_ok = (not self.camera_required) or self.camera_available
        success = bool(
            self.asset_supported
            and self.max_door_open_deg >= self.pass_open_angle_deg
            and not self.base_collision
            and body_ok
            and camera_ok
        )
        metrics = {
            "min_grasp_ee_handle_dist": None if math.isinf(self.min_grasp_ee_handle_dist) else self.min_grasp_ee_handle_dist,
            "max_push_ee_handle_dist": self.max_push_ee_handle_dist,
            "joint_limit_hit": self.joint_limit_hit,
            "pre_push_joint_limit_hit": self.pre_push_joint_limit_hit,
            "max_pre_push_ee_tracking_error": self.max_pre_push_ee_tracking_error,
            "secondary_failures": self._secondary_failures(),
            "last_base_xy": self.last_base_xy,
            "last_base_command": self.last_base_command,
            "max_abs_base_vx": self.max_abs_base_vx,
            "max_abs_base_vyaw": self.max_abs_base_vyaw,
            "last_target_pos": self.last_target_pos,
            "last_ee_pos": self.last_ee_pos,
            "last_handle_goal": self.last_handle_goal,
            "trace": self.trace,
        }
        return RolloutReport(
            env_id=self.env_id,
            door_name=self.door_name,
            success=success,
            failure_stage=self._failure_stage(success),
            door_open_deg=float(self.max_door_open_deg),
            handle_rotation_deg=float(self.max_handle_rotation_deg),
            ee_handle_dist=(
                float(self.min_ee_handle_dist) if not math.isinf(self.min_ee_handle_dist) else math.inf
            ),
            ee_tracking_error=float(self.max_ee_tracking_error),
            base_collision=bool(self.base_collision),
            body_passed=bool(self.body_passed),
            camera_available=bool(self.camera_available),
            handle_unlocked=bool(self.handle_unlocked),
            steps=int(self.steps),
            final_phase=self.final_phase,
            artifacts=self.artifacts,
            metrics=metrics,
            skill_program=self.skill_program,
            door_spec=self.door_spec,
        )

    def _secondary_failures(self) -> list[str]:
        failures = []
        if self.max_pre_push_ee_tracking_error > 0.12 or self.pre_push_joint_limit_hit:
            failures.append("arm_joint_limit_or_ik_bad")
        if self.min_grasp_ee_handle_dist > 0.09:
            failures.append("grasp_miss")
        if not self.handle_unlocked:
            failures.append("handle_not_unlocked")
        if self.base_collision:
            failures.append("base_collision")
        if self.max_push_ee_handle_dist > 0.20:
            failures.append("contact_lost")
        if self.max_door_open_deg < self.pass_open_angle_deg:
            failures.append("door_push_insufficient")
        if self.require_traverse and not self.body_passed:
            failures.append("body_blocked")
        primary = self._failure_stage(False)
        return [failure for failure in failures if failure != primary]


def write_rollout_reports(
    trackers: list[RolloutTracker],
    log_dir: str | Path,
    *,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(log_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = [tracker.finalize() for tracker in trackers]
    for report in reports:
        path = out_dir / f"env_{report.env_id:04d}_report.json"
        path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report.artifacts["report"] = str(path)
    success_count = sum(1 for report in reports if report.success)
    successful_env_ids = [int(report.env_id) for report in reports if report.success]
    failure_counts: dict[str, int] = {}
    for report in reports:
        key = report.failure_stage or "success"
        failure_counts[key] = failure_counts.get(key, 0) + 1
    expert_trajectories = []
    for report in reports:
        for artifact in report.artifacts.get("expert_trajectories", []):
            expert_trajectories.append({"env_id": int(report.env_id), **dict(artifact)})
    summary = {
        "schema_version": "door_twin_rollout_summary_v1",
        "num_envs": len(reports),
        "success_count": success_count,
        "success_rate": float(success_count) / float(max(1, len(reports))),
        "successful_env_ids": successful_env_ids,
        "expert_trajectories": expert_trajectories,
        "failure_counts": failure_counts,
        "best_skill_program": next((report.skill_program for report in reports if report.skill_program), None),
        "run_metadata": dict(run_metadata or {}),
        "reports": [report.to_dict() for report in reports],
    }
    summary_path = out_dir / "rollout_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
