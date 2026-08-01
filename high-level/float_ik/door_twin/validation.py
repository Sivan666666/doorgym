"""Fast static validation and three-level probe summaries for DoorTwin."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .skill import MIN_DOOR_TWIN_FORWARD_DISTANCE_M, PATCH_BOUNDS, PRIMITIVE_ORDER, SkillProgram


STATIC_SCHEMA_VERSION = "door_twin_static_validation_v1"
PROBE_SCHEMA_VERSION = "door_twin_probe_summary_v1"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    blocking: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticValidationReport:
    candidate_hash: str
    checks: dict[str, bool]
    issues: list[ValidationIssue]
    measurements: dict[str, Any]

    @property
    def passed(self) -> bool:
        return not any(issue.blocking for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATIC_SCHEMA_VERSION,
            "candidate_hash": self.candidate_hash,
            "passed": self.passed,
            "checks": self.checks,
            "issues": [asdict(issue) for issue in self.issues],
            "measurements": self.measurements,
        }


def _finite_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _joint_range_deg(joint: dict[str, Any] | None) -> float:
    if joint is None:
        return 0.0
    if str(joint.get("type", "")) == "continuous":
        return 360.0
    lower, upper = joint.get("lower"), joint.get("upper")
    if lower is None or upper is None:
        return 0.0
    return math.degrees(abs(float(upper) - float(lower)))


def _mesh_paths(urdf_path: Path) -> list[tuple[str, Path | None]]:
    root = ET.parse(urdf_path).getroot()
    result = []
    for mesh in root.findall(".//mesh"):
        filename = str(mesh.get("filename", "")).strip()
        if not filename:
            result.append((filename, None))
        elif filename.startswith(("package://", "http://", "https://")):
            result.append((filename, None))
        else:
            path = Path(filename).expanduser()
            result.append((filename, path.resolve() if path.is_absolute() else (urdf_path.parent / path).resolve()))
    return result


def validate_candidate_static(candidate: Any, *, handle_goal_tolerance_m: float = 0.02) -> StaticValidationReport:
    issues: list[ValidationIssue] = []
    checks: dict[str, bool] = {}
    measurements: dict[str, Any] = {}
    asset_path = candidate.asset_path()

    checks["asset_exists"] = asset_path.is_file()
    if not asset_path.is_file():
        issues.append(ValidationIssue("ASSET_MISSING", f"URDF does not exist: {asset_path}"))
        return StaticValidationReport(candidate.fingerprint, checks, issues, measurements)

    try:
        structure = candidate.urdf_structure()
        ET.parse(asset_path)
        checks["urdf_parse"] = True
    except (ET.ParseError, OSError, ValueError) as exc:
        checks["urdf_parse"] = False
        issues.append(ValidationIssue("URDF_PARSE", f"Unable to parse URDF: {exc}"))
        return StaticValidationReport(candidate.fingerprint, checks, issues, measurements)

    missing_meshes = []
    unresolved_meshes = []
    for filename, path in _mesh_paths(asset_path):
        if path is None:
            unresolved_meshes.append(filename)
        elif not path.is_file():
            missing_meshes.append(filename)
    checks["mesh_files"] = not missing_meshes
    if missing_meshes:
        issues.append(ValidationIssue("MESH_MISSING", "URDF references missing mesh files", evidence={"files": missing_meshes}))
    if unresolved_meshes:
        issues.append(
            ValidationIssue(
                "MESH_URI_UNRESOLVED",
                "External/package mesh URIs could not be checked statically",
                blocking=False,
                evidence={"files": unresolved_meshes},
            )
        )

    links = set(structure.get("links", []))
    joints = {str(joint.get("name", "")): joint for joint in structure.get("joints", [])}
    runtime = dict(candidate.runtime_spec)
    for key in ("door_body_name", "handle_body_name"):
        name = str(runtime.get(key, ""))
        valid = bool(name and name in links)
        checks[key] = valid
        if not valid:
            issues.append(ValidationIssue("BODY_MAPPING", f"{key}={name!r} is not a URDF link", evidence={"field": key}))

    door_joint_name = str(runtime.get("door_dof_name", ""))
    door_joint = joints.get(door_joint_name)
    door_joint_valid = door_joint is not None and str(door_joint.get("type", "")) in {"revolute", "continuous"}
    checks["door_dof_name"] = door_joint_valid
    if not door_joint_valid:
        issues.append(ValidationIssue("DOOR_JOINT_MAPPING", f"door_dof_name={door_joint_name!r} is not a movable hinge"))
    hinge_range = _joint_range_deg(door_joint)
    measurements["hinge_range_deg"] = hinge_range
    checks["hinge_range"] = hinge_range >= 30.0
    if hinge_range < 30.0:
        issues.append(ValidationIssue("HINGE_RANGE", f"Door hinge range is only {hinge_range:.2f} deg"))
    if door_joint is not None:
        axis = _finite_vector(str(door_joint.get("axis", "")).split(), 3)
        axis_valid = axis is not None and float(np.linalg.norm(axis)) > 1.0e-6
        checks["hinge_axis"] = axis_valid
        if not axis_valid:
            issues.append(ValidationIssue("HINGE_AXIS", f"Door hinge axis is invalid: {door_joint.get('axis')!r}"))

    handle_joint_name = str(runtime.get("handle_dof_name", ""))
    if handle_joint_name:
        handle_joint = joints.get(handle_joint_name)
        valid = handle_joint is not None and str(handle_joint.get("type", "")) in {"revolute", "continuous", "prismatic"}
        checks["handle_dof_name"] = valid
        if not valid:
            issues.append(ValidationIssue("HANDLE_JOINT_MAPPING", f"handle_dof_name={handle_joint_name!r} is not movable"))
    else:
        checks["handle_dof_name"] = True
        measurements["handle_type"] = "fixed"

    scale = runtime.get("actor_scale", 1.0)
    yaw = runtime.get("actor_yaw_offset", 0.0)
    offset = _finite_vector(runtime.get("actor_position_offset", [0.0, 0.0, 0.0]), 3)
    pose_valid = (
        isinstance(scale, (int, float))
        and math.isfinite(float(scale))
        and 0.25 <= float(scale) <= 3.0
        and isinstance(yaw, (int, float))
        and math.isfinite(float(yaw))
        and offset is not None
    )
    checks["actor_pose"] = pose_valid
    if not pose_valid:
        issues.append(ValidationIssue("ACTOR_POSE", "actor scale/yaw/position offset is invalid"))
    ground_offset = math.inf if offset is None else float(offset[2])
    measurements["ground_clearance_m"] = ground_offset
    checks["ground_clearance"] = ground_offset >= -0.02
    if ground_offset < -0.02:
        issues.append(ValidationIssue("GROUND_PENETRATION", f"Configured actor z offset penetrates ground by {-ground_offset:.3f} m"))

    handle_min = _finite_vector(candidate.handle_bounding.get("handle_min"), 3)
    handle_max = _finite_vector(candidate.handle_bounding.get("handle_max"), 3)
    handle_goal = _finite_vector(candidate.handle_bounding.get("goal_pos"), 3)
    goal_valid = handle_min is not None and handle_max is not None and handle_goal is not None
    if goal_valid:
        assert handle_min is not None and handle_max is not None and handle_goal is not None
        goal_valid = all(
            min(lo, hi) - handle_goal_tolerance_m <= goal <= max(lo, hi) + handle_goal_tolerance_m
            for lo, hi, goal in zip(handle_min, handle_max, handle_goal)
        )
    checks["handle_goal"] = goal_valid
    if not goal_valid:
        issues.append(ValidationIssue("HANDLE_GOAL", "handle goal is non-finite or outside handle bbox", evidence={"tolerance_m": handle_goal_tolerance_m}))

    _validate_skill(candidate.skill_program, checks, issues)
    return StaticValidationReport(candidate.fingerprint, checks, issues, measurements)


def _validate_skill(program: SkillProgram, checks: dict[str, bool], issues: list[ValidationIssue]) -> None:
    allowed = set(PRIMITIVE_ORDER)
    unknown = [primitive.name for primitive in program.primitives if primitive.name not in allowed]
    checks["skill_primitives"] = not unknown
    if unknown:
        issues.append(ValidationIssue("SKILL_PRIMITIVE", "Unknown skill primitive", evidence={"names": unknown}))

    order_map = {
        ("MoveTo", "approach"): 0,
        ("ApproachDoor", ""): 1,
        ("MoveEEToHandle", ""): 2,
        ("CloseGripper", ""): 3,
        ("RotateHandle", ""): 4,
        ("PushDoor", ""): 5,
        ("MoveTo", "push"): 6,
        ("TraverseDoor", ""): 7,
        ("MoveTo", "traverse"): 8,
        ("ReleaseAndRetract", ""): 9,
    }
    sequence = [order_map.get((primitive.name, str(primitive.params.get("stage", "")).lower()), -1) for primitive in program.primitives]
    known_sequence = [value for value in sequence if value >= 0]
    ordered = known_sequence == sorted(known_sequence)
    checks["skill_order"] = ordered
    if not ordered:
        issues.append(ValidationIssue("SKILL_ORDER", "Skill primitives are not in executable task order", evidence={"order": sequence}))

    finite = True
    for primitive in program.primitives:
        for key, value in primitive.params.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                finite = False
            if isinstance(value, (list, tuple)):
                try:
                    finite = finite and all(math.isfinite(float(item)) for item in value)
                except (TypeError, ValueError):
                    finite = False
    checks["skill_finite"] = finite
    if not finite:
        issues.append(ValidationIssue("SKILL_NONFINITE", "Skill contains non-finite numeric values"))

    bounds_ok = True
    for primitive in program.primitives:
        stage = str(primitive.params.get("stage", "")).lower()
        prefix = f"{primitive.name}:{stage}" if primitive.name == "MoveTo" and stage else primitive.name
        for parameter, value in primitive.params.items():
            path = f"{prefix}.{parameter}"
            if path not in PATCH_BOUNDS:
                continue
            lo, hi = PATCH_BOUNDS[path]
            values = value if isinstance(value, (list, tuple)) else [value]
            lower = lo if isinstance(lo, list) else [lo]
            upper = hi if isinstance(hi, list) else [hi]
            if len(values) != len(lower) or any(float(v) < float(a) or float(v) > float(b) for v, a, b in zip(values, lower, upper)):
                bounds_ok = False
                issues.append(ValidationIssue("SKILL_BOUNDS", f"{path} is outside allowed bounds", evidence={"value": value, "bounds": [lo, hi]}))
    checks["skill_bounds"] = bounds_ok

    move_to_handle = program.primitive("MoveEEToHandle")
    pregrasp = None if move_to_handle is None else move_to_handle.params.get("pregrasp_offset")
    grasp = None if move_to_handle is None else move_to_handle.params.get("grasp_offset")
    matched_handle_z = bool(
        isinstance(pregrasp, (list, tuple))
        and isinstance(grasp, (list, tuple))
        and len(pregrasp) == 3
        and len(grasp) == 3
        and abs(float(pregrasp[2]) - float(grasp[2])) <= 1.0e-6
    )
    checks["pregrasp_grasp_z_match"] = matched_handle_z
    if not matched_handle_z:
        issues.append(
            ValidationIssue(
                "HANDLE_APPROACH_Z_MISMATCH",
                "MoveEEToHandle pregrasp_offset.z must equal grasp_offset.z",
                evidence={"pregrasp_offset": pregrasp, "grasp_offset": grasp},
            )
        )

    push = program.primitive("PushDoor")
    push_move = next((item for item in program.primitives_named("MoveTo") if item.params.get("stage") == "push"), None)
    traverse = program.primitive("TraverseDoor")
    traverse_move = next((item for item in program.primitives_named("MoveTo") if item.params.get("stage") == "traverse"), None)
    duplicate = bool(push and push_move and any(key in push.params for key in ("base_distance", "base_v")))
    duplicate = duplicate or bool(traverse and traverse_move and any(key in traverse.params for key in ("base_v", "distance")))
    checks["single_base_command_source"] = not duplicate
    if duplicate:
        issues.append(ValidationIssue("DUPLICATE_BASE_COMMAND", "Base motion is defined by both semantic and MoveTo primitives"))

    push_distance = None if push_move is None else push_move.params.get("distance")
    traverse_distance = None if traverse_move is None else traverse_move.params.get("distance")
    complete_traverse = bool(traverse is not None and push_distance is not None and traverse_distance is not None)
    forward_distance = (
        float(push_distance) + float(traverse_distance)
        if complete_traverse
        else 0.0
    )
    forward_distance_ok = bool(
        complete_traverse
        and forward_distance + 1.0e-9 >= MIN_DOOR_TWIN_FORWARD_DISTANCE_M
    )
    checks["minimum_forward_distance"] = forward_distance_ok
    if not forward_distance_ok:
        issues.append(
            ValidationIssue(
                "FORWARD_DISTANCE_INSUFFICIENT",
                (
                    "Push plus traverse distance must be at least "
                    f"{MIN_DOOR_TWIN_FORWARD_DISTANCE_M:.2f} m"
                ),
                evidence={
                    "push_distance_m": push_distance,
                    "traverse_distance_m": traverse_distance,
                    "total_distance_m": forward_distance if complete_traverse else None,
                    "minimum_distance_m": MIN_DOOR_TWIN_FORWARD_DISTANCE_M,
                },
            )
        )


def summarize_physics_probes(
    candidate_hash: str,
    *,
    load_summary: dict[str, Any],
    hinge_summary: dict[str, Any],
    handle_summary: dict[str, Any] | None,
    grasp_summary: dict[str, Any],
    has_handle_dof: bool,
    hinge_threshold_deg: float = 30.0,
    grasp_tracking_threshold_m: float = 0.08,
) -> dict[str, Any]:
    load_reports = list(load_summary.get("reports", []) or [])
    hinge_reports = list(hinge_summary.get("reports", []) or [])
    handle_reports = list((handle_summary or {}).get("reports", []) or [])
    grasp_reports = list(grasp_summary.get("reports", []) or [])
    load_ok = bool(
        load_reports
        and not load_summary.get("process_failures")
        and int(load_reports[0].get("steps", 0)) >= 100
        and load_reports[0].get("failure_stage") != "asset_invalid"
        and not bool(load_reports[0].get("initial_penetration", False))
    )
    hinge_torque_deg = max((float(report.get("door_open_deg", 0.0)) for report in hinge_reports), default=0.0)
    # Some Isaac Gym builds leave an untouched articulation asleep when only
    # DOF effort is applied. The handle-motion probe is still a bounded physical
    # interaction and provides a valid fallback proof that the hinge moves.
    hinge_interaction_deg = max(
        (float(report.get("door_open_deg", 0.0)) for report in handle_reports),
        default=0.0,
    )
    hinge_deg = max(hinge_torque_deg, hinge_interaction_deg)
    hinge_ok = hinge_deg >= hinge_threshold_deg
    handle_deg = max((float(report.get("handle_rotation_deg", 0.0)) for report in handle_reports), default=0.0)
    handle_ok = True if not has_handle_dof else any(bool(report.get("handle_unlocked", False)) for report in handle_reports)
    tracking_values = []
    for report in grasp_reports:
        metrics = report.get("metrics", {}) or {}
        phase_metrics = metrics.get("phase_metrics", {}) or {}
        # Reachability is about the pregrasp/grasp endpoint, not the transient
        # peak while the arm first leaves its home pose or rotates the handle.
        # Those phases are evaluated by the full rollout diagnostics.
        grasp_phase = phase_metrics.get("grasp", {}) or {}
        value = grasp_phase.get(
            "max_ee_tracking_error",
            metrics.get("max_pre_push_ee_tracking_error", report.get("ee_tracking_error")),
        )
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            tracking_values.append(value)
    tracking = max(tracking_values, default=None)
    collision = False
    for report in grasp_reports:
        phase_metrics = (report.get("metrics", {}) or {}).get("phase_metrics", {}) or {}
        collision = collision or any(
            bool((phase_metrics.get(name, {}) or {}).get("base_collision", False))
            for name in ("walk", "initial_hold", "grasp", "close_gripper")
        )
    joint_limit = any(bool(report.get("metrics", {}).get("pre_push_joint_limit_hit", False)) for report in grasp_reports)
    grasp_distances = []
    for report in grasp_reports:
        value = report.get("metrics", {}).get("min_grasp_ee_handle_dist", report.get("ee_handle_dist"))
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            grasp_distances.append(value)
    min_grasp_distance = min(grasp_distances, default=None)
    grasp_ok = bool(
        grasp_reports
        and tracking is not None
        and tracking <= grasp_tracking_threshold_m
        and min_grasp_distance is not None
        and min_grasp_distance <= 0.09
        and not collision
        and not joint_limit
    )
    checks = {
        "load_stability": load_ok,
        "hinge_motion": hinge_ok,
        "handle_motion": handle_ok,
        "grasp_reachability": grasp_ok,
    }
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "candidate_hash": candidate_hash,
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": {
            "hinge_motion_deg": hinge_deg,
            "hinge_torque_probe_deg": hinge_torque_deg,
            "hinge_interaction_probe_deg": hinge_interaction_deg,
            "hinge_motion_source": "torque_probe" if hinge_torque_deg >= hinge_threshold_deg else "interaction_probe",
            "handle_motion_deg": handle_deg,
            "max_pre_push_ee_tracking_error_m": tracking,
            "min_grasp_ee_handle_distance_m": min_grasp_distance,
            "base_collision": collision,
            "pre_push_joint_limit_hit": joint_limit,
        },
    }


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
