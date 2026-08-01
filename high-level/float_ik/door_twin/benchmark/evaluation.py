"""Rollout aggregation and two-stage DoorTwin benchmark scoring."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .schema import Candidate, DoorCase, hinge_range_degrees


def synthetic_failed_summary(seed: int, returncode: int, log_tail: str) -> dict[str, Any]:
    report = {
        "env_id": 0,
        "benchmark_seed": int(seed),
        "success": False,
        "failure_stage": "asset_invalid",
        "door_open_deg": 0.0,
        "handle_rotation_deg": 0.0,
        "ee_handle_dist": math.inf,
        "ee_tracking_error": 0.0,
        "base_collision": False,
        "body_passed": False,
        "camera_available": False,
        "handle_unlocked": False,
        "steps": 0,
        "final_phase": "process_failed",
        "artifacts": {},
        "metrics": {"secondary_failures": [], "trace": []},
    }
    return {
        "schema_version": "door_twin_rollout_summary_v1",
        "num_envs": 1,
        "success_count": 0,
        "success_rate": 0.0,
        "failure_counts": {"asset_invalid": 1},
        "reports": [report],
        "process_failures": [{"seed": int(seed), "returncode": int(returncode), "log_tail": log_tail}],
    }


def load_seed_summary(path: str | Path, seed: int) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for report in data.get("reports", []) or []:
        report["benchmark_seed"] = int(seed)
    return data


def load_batch_summary(path: str | Path, seeds: list[int]) -> dict[str, Any]:
    """Load one parallel Isaac Gym rollout and label each report deterministically."""

    if not seeds:
        raise ValueError("A batched rollout requires at least one benchmark seed")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    reports = list(data.get("reports", []) or [])
    if len(reports) != len(seeds):
        raise ValueError(
            f"Parallel rollout produced {len(reports)} reports for {len(seeds)} requested seeds"
        )
    for index, report in enumerate(reports):
        env_id = int(report.get("env_id", index))
        label_index = env_id if 0 <= env_id < len(seeds) else index
        report["benchmark_seed"] = int(seeds[label_index])
        report["benchmark_base_seed"] = int(seeds[0])
    data["benchmark_seeds"] = [int(seed) for seed in seeds]
    data["benchmark_base_seed"] = int(seeds[0])
    return data


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    process_failures: list[dict[str, Any]] = []
    for summary in summaries:
        reports.extend(summary.get("reports", []) or [])
        process_failures.extend(summary.get("process_failures", []) or [])
    success_count = sum(bool(report.get("success", False)) for report in reports)
    counts = Counter(str(report.get("failure_stage", "") or "success") for report in reports)
    return {
        "schema_version": "door_twin_benchmark_aggregate_v1",
        "num_envs": len(reports),
        "success_count": int(success_count),
        "success_rate": float(success_count) / float(max(1, len(reports))),
        "failure_counts": dict(counts),
        "reports": reports,
        "process_failures": process_failures,
    }


def stage_metrics(summary: dict[str, Any], *, pass_open_angle_deg: float = 80.0) -> dict[str, Any]:
    reports = list(summary.get("reports", []) or [])
    n = max(1, len(reports))
    grasp = 0
    for report in reports:
        distance = report.get("metrics", {}).get("min_grasp_ee_handle_dist")
        if distance is None:
            distance = report.get("ee_handle_dist")
        if distance is not None and float(distance) <= 0.09:
            grasp += 1
    unlock = sum(bool(report.get("handle_unlocked", False)) for report in reports)
    opened = sum(float(report.get("door_open_deg", 0.0)) >= pass_open_angle_deg for report in reports)
    traverse = sum(bool(report.get("body_passed", False)) for report in reports)
    success = sum(bool(report.get("success", False)) for report in reports)
    return {
        "trials": len(reports),
        "grasp_count": grasp,
        "unlock_count": unlock,
        "open_count": opened,
        "traverse_count": traverse,
        "success_count": success,
        "grasp_rate": grasp / n,
        "unlock_rate": unlock / n,
        "open_rate": opened / n,
        "traverse_rate": traverse / n,
        "success_rate": success / n,
    }


def evaluate_asset(
    candidate: Candidate,
    door: DoorCase,
    summary: dict[str, Any],
    *,
    door_z_offset: float = 0.0,
    motion_fallback_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reports = list(summary.get("reports", []) or [])
    first = reports[0] if reports else {}
    runtime = dict(first.get("door_spec", {}) or candidate.runtime_spec)
    gt = dict(door.hidden_ground_truth or {})
    load_success = bool(reports and int(first.get("steps", 0)) >= 100 and first.get("failure_stage") != "asset_invalid")
    semantic_fields = ("door_body_name", "handle_body_name", "door_dof_name", "handle_dof_name")
    semantic_checks = {}
    for key in semantic_fields:
        expected = gt.get(key)
        semantic_checks[key] = True if expected is None else str(runtime.get(key, "")) == str(expected)
    predicted_goal = np.asarray(candidate.handle_bounding.get("goal_pos", (math.inf,) * 3), dtype=np.float64)
    expected_goal = np.asarray(gt.get("handle_goal_pos", predicted_goal), dtype=np.float64)
    goal_error = float(np.linalg.norm(predicted_goal - expected_goal)) if predicted_goal.size == 3 else math.inf
    # Fresh manifests historically copied this value from the automatically
    # generated base candidate. It is a frozen reference, not an independent
    # annotation. Keep its delta for diagnostics, but only gate structure
    # success when the manifest explicitly marks it as externally verified.
    goal_reference_type = str(gt.get("handle_goal_reference_type", "generated_initial_candidate"))
    goal_check_required = bool(gt.get("handle_goal_verified", False))
    goal_check_passed = bool(goal_error <= 0.05) if goal_check_required else None
    actor_offset = candidate.runtime_spec.get("actor_position_offset", (0.0, 0.0, 0.0))
    ground_clearance = float(door_z_offset) + float(actor_offset[2] if len(actor_offset) >= 3 else 0.0)
    hinge_range = hinge_range_degrees(candidate)
    fallback_reports = list((motion_fallback_summary or {}).get("reports", []) or [])
    hinge_probe_moved = max((float(report.get("door_open_deg", 0.0)) for report in reports), default=0.0)
    hinge_fallback_moved = max(
        (float(report.get("door_open_deg", 0.0)) for report in fallback_reports),
        default=0.0,
    )
    hinge_moved = max(hinge_probe_moved, hinge_fallback_moved)
    structure_success = bool(
        load_success
        and all(semantic_checks.values())
        and (not goal_check_required or bool(goal_check_passed))
        and ground_clearance >= -0.02
        and hinge_range >= 30.0
        and hinge_moved >= 30.0
    )
    return {
        "load_success": load_success,
        "structure_success": structure_success,
        "semantic_checks": semantic_checks,
        # Preserve the legacy key for old report readers.
        "handle_goal_error_m": goal_error,
        "frozen_reference_handle_goal_delta_m": goal_error,
        "handle_goal_reference_type": goal_reference_type,
        "handle_goal_check_required": goal_check_required,
        "handle_goal_check_passed": goal_check_passed,
        "ground_clearance_m": ground_clearance,
        "hinge_range_deg": hinge_range,
        "asset_probe_hinge_motion_deg": hinge_moved,
        "isolated_torque_probe_hinge_motion_deg": hinge_probe_moved,
        "interaction_fallback_hinge_motion_deg": hinge_fallback_moved,
        "hinge_motion_source": "isolated_torque_probe" if hinge_probe_moved >= 30.0 else "interaction_fallback",
    }
