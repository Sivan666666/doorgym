"""Skill-program schema and local-frame waypoint generation."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


PRIMITIVE_ORDER = (
    "MoveTo",
    "ApproachDoor",
    "MoveEEToHandle",
    "CloseGripper",
    "RotateHandle",
    "PushDoor",
    "HoldDoor",
    "TraverseDoor",
    "ReleaseAndRetract",
)


def _as_float_list(value: Any, length: int, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"Expected a {length}D list, got {value!r}")
    return [float(v) for v in value]


def _seconds_or_steps(value: Any, *, dt: float = 0.02, default_steps: int) -> int:
    if value is None:
        return int(default_steps)
    if isinstance(value, str) and value.endswith("s"):
        return max(1, int(round(float(value[:-1]) / dt)))
    value = float(value)
    if value <= 0:
        return int(default_steps)
    if value <= 20.0:
        return max(1, int(round(value / dt)))
    return max(1, int(round(value)))


@dataclass
class SkillPrimitive:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any) -> "SkillPrimitive":
        if isinstance(obj, str):
            return cls(name=obj, params={})
        if not isinstance(obj, dict):
            raise ValueError(f"Skill primitive must be an object or string, got {type(obj).__name__}")
        name = obj.get("name") or obj.get("type") or obj.get("primitive")
        if not name:
            raise ValueError(f"Skill primitive is missing name/type: {obj!r}")
        params = {k: v for k, v in obj.items() if k not in ("name", "type", "primitive")}
        return cls(name=str(name), params=params)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, **copy.deepcopy(self.params)}


@dataclass
class SkillProgram:
    version: str = "door_skill_program_v1"
    primitives: list[SkillPrimitive] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any) -> "SkillProgram":
        if isinstance(obj, list):
            primitives = [SkillPrimitive.from_obj(item) for item in obj]
            return cls(primitives=primitives)
        if not isinstance(obj, dict):
            raise ValueError(f"Skill program must be a list or object, got {type(obj).__name__}")
        raw_primitives = obj.get("skill_program", obj.get("primitives", obj.get("program")))
        if raw_primitives is None:
            raise ValueError("Skill program JSON must contain skill_program/primitives/program")
        primitives = [SkillPrimitive.from_obj(item) for item in raw_primitives]
        return cls(
            version=str(obj.get("version", "door_skill_program_v1")),
            primitives=primitives,
            metadata=dict(obj.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "metadata": copy.deepcopy(self.metadata),
            "skill_program": [primitive.to_dict() for primitive in self.primitives],
        }

    def primitive(self, name: str) -> SkillPrimitive | None:
        for primitive in self.primitives:
            if primitive.name == name:
                return primitive
        return None

    def primitives_named(self, name: str) -> list[SkillPrimitive]:
        return [primitive for primitive in self.primitives if primitive.name == name]


@dataclass
class SkillTrajectoryProfile:
    pregrasp_local: list[float]
    grasp_local: list[float]
    handle_goal_bias_world: list[float]
    rotate_local_delta: list[float]
    base_moves: dict[str, dict[str, Any]]
    traverse_required: bool
    raw_program: dict[str, Any]


@dataclass
class SkillWaypoints:
    pregrasp: np.ndarray
    grasp: np.ndarray
    rotate: np.ndarray
    lateral_dir: np.ndarray


def load_skill_program(path: str | Path) -> SkillProgram:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SkillProgram.from_obj(data)


def default_skill_program_from_args(args: Any, *, door_family: str = "") -> SkillProgram:
    dt = float(getattr(args, "sim_dt", 0.02) or 0.02)
    push_steps = int(getattr(args, "door_push_steps", 300))
    push_distance = float(getattr(args, "push_base_distance", 0.35))
    push_vx = push_distance / max(dt * float(push_steps), 1.0e-6)
    z_extra = 0.0
    if str(door_family) == "wc4":
        z_extra = float(getattr(args, "wc4_grasp_z_offset", 0.0))
    program = SkillProgram(
        metadata={
            "source": "current_a2w_float_ik_defaults",
            "execution_mode": "legacy_replay",
        },
        primitives=[
            SkillPrimitive(
                "MoveTo",
                {
                    "stage": "approach",
                    "vx": float(getattr(args, "walk_min_speed", 0.20)),
                    "vyaw": 0.0,
                    "stop_distance": float(getattr(args, "stop_distance", 0.15)),
                },
            ),
            SkillPrimitive("ApproachDoor", {"base_offset": float(getattr(args, "stop_distance", 0.15))}),
            SkillPrimitive(
                "MoveEEToHandle",
                {
                    "pregrasp_offset": [
                        float(getattr(args, "pregrasp_offset", 0.15)),
                        0.0,
                        float(getattr(args, "grasp_z_offset", -0.03)) + z_extra,
                    ],
                    "grasp_offset": [
                        float(getattr(args, "grasp_offset", 0.0)),
                        0.0,
                        float(getattr(args, "grasp_z_offset", -0.03)) + z_extra,
                    ],
                    "handle_goal_bias_world": [float(getattr(args, "grasp_x_offset", -0.015)), 0.0, 0.0],
                    "duration_steps": int(getattr(args, "grasp_steps", 50)),
                },
            ),
            SkillPrimitive("CloseGripper", {"force": float(getattr(args, "gripper_close_ratio", 0.8))}),
            SkillPrimitive(
                "RotateHandle",
                {
                    "angle": float(getattr(args, "handle_rotate_angle", 1.05)),
                    "duration_steps": int(getattr(args, "handle_rotate_steps", 100)),
                    "local_delta": [
                        0.0,
                        float(getattr(args, "handle_rotate_right_distance", 0.03)),
                        -float(getattr(args, "handle_rotate_down_distance", 0.03)),
                    ],
                },
            ),
            SkillPrimitive(
                "PushDoor",
                {
                    "distance": float(getattr(args, "door_push_distance", 1.10)),
                    "duration_steps": push_steps,
                    "contact_bias": float(getattr(args, "push_contact_bias", 0.025)),
                },
            ),
            SkillPrimitive(
                "MoveTo",
                {
                    "stage": "push",
                    "vx": push_vx,
                    "vyaw": float(getattr(args, "push_base_yaw_delta", 0.0))
                    / max(dt * float(push_steps), 1.0e-6),
                    "distance": push_distance,
                    "duration_steps": push_steps,
                },
            ),
            SkillPrimitive("ReleaseAndRetract", {"duration_steps": int(getattr(args, "return_home_steps", 150))}),
        ],
    )
    return program


def profile_from_program(program: SkillProgram, args: Any | None = None) -> SkillTrajectoryProfile:
    move = program.primitive("MoveEEToHandle")
    rotate = program.primitive("RotateHandle")
    pregrasp_default = [0.15, 0.0, -0.03]
    grasp_default = [0.0, 0.0, -0.03]
    if args is not None:
        pregrasp_default = [float(getattr(args, "pregrasp_offset", 0.15)), 0.0, float(getattr(args, "grasp_z_offset", -0.03))]
        grasp_default = [float(getattr(args, "grasp_offset", 0.0)), 0.0, float(getattr(args, "grasp_z_offset", -0.03))]
    pregrasp = _as_float_list(None if move is None else move.params.get("pregrasp_offset"), 3, pregrasp_default)
    grasp = _as_float_list(None if move is None else move.params.get("grasp_offset"), 3, grasp_default)
    bias = _as_float_list(None if move is None else move.params.get("handle_goal_bias_world"), 3, [0.0, 0.0, 0.0])
    rotate_delta = _as_float_list(
        None if rotate is None else rotate.params.get("local_delta"),
        3,
        [
            0.0,
            float(getattr(args, "handle_rotate_right_distance", 0.03)) if args is not None else 0.03,
            -float(getattr(args, "handle_rotate_down_distance", 0.03)) if args is not None else -0.03,
        ],
    )
    traverse_required = program.primitive("TraverseDoor") is not None
    base_moves = {}
    for move_to in program.primitives_named("MoveTo"):
        stage = str(move_to.params.get("stage", "approach")).strip().lower()
        if stage in ("approach", "push", "traverse"):
            base_moves[stage] = copy.deepcopy(move_to.params)
    return SkillTrajectoryProfile(
        pregrasp_local=pregrasp,
        grasp_local=grasp,
        handle_goal_bias_world=bias,
        rotate_local_delta=rotate_delta,
        base_moves=base_moves,
        traverse_required=traverse_required,
        raw_program=program.to_dict(),
    )


def apply_program_to_args(args: Any, program: SkillProgram) -> None:
    """Apply timing and scalar controller parameters before env creation."""

    dt = float(getattr(args, "sim_dt", 0.02) or 0.02)
    approach = program.primitive("ApproachDoor")
    if approach is not None and approach.params.get("base_offset") is not None:
        args.stop_distance = float(approach.params["base_offset"])

    move = program.primitive("MoveEEToHandle")
    if move is not None:
        args.grasp_steps = _seconds_or_steps(
            move.params.get("duration", move.params.get("duration_steps")),
            dt=dt,
            default_steps=int(getattr(args, "grasp_steps", 50)),
        )

    close = program.primitive("CloseGripper")
    if close is not None:
        if close.params.get("force") is not None:
            args.gripper_close_ratio = float(np.clip(float(close.params["force"]), 0.0, 1.2))
        args.gripper_close_steps = _seconds_or_steps(
            close.params.get("duration", close.params.get("duration_steps")),
            dt=dt,
            default_steps=int(getattr(args, "gripper_close_steps", 50)),
        )

    rotate = program.primitive("RotateHandle")
    if rotate is not None:
        if rotate.params.get("angle") is not None:
            angle = float(rotate.params["angle"])
            args.handle_rotate_angle = math.radians(angle) if abs(angle) > 2.0 * math.pi else angle
        if rotate.params.get("direction_sign") is not None:
            args.handle_rotate_direction_sign = float(rotate.params["direction_sign"])
        args.handle_rotate_steps = _seconds_or_steps(
            rotate.params.get("duration", rotate.params.get("duration_steps")),
            dt=dt,
            default_steps=int(getattr(args, "handle_rotate_steps", 100)),
        )

    push = program.primitive("PushDoor")
    if push is not None:
        if push.params.get("distance") is not None:
            args.door_push_distance = float(push.params["distance"])
        if push.params.get("base_distance") is not None:
            args.push_base_distance = float(push.params["base_distance"])
        if push.params.get("contact_bias") is not None:
            args.push_contact_bias = float(push.params["contact_bias"])
        args.door_push_steps = _seconds_or_steps(
            push.params.get("duration", push.params.get("duration_steps")),
            dt=dt,
            default_steps=int(getattr(args, "door_push_steps", 300)),
        )
        base_v = push.params.get("base_v")
        if base_v is not None:
            args.push_base_distance = max(0.0, float(base_v) * float(args.door_push_steps) * dt)

    traverse = program.primitive("TraverseDoor")
    legacy_replay = str(program.metadata.get("execution_mode", "")).strip().lower() == "legacy_replay"
    if not legacy_replay:
        args.pass_through_door = traverse is not None
    if traverse is not None:
        traverse_steps = _seconds_or_steps(
            traverse.params.get("duration", traverse.params.get("duration_steps")),
            dt=dt,
            default_steps=int(getattr(args, "traverse_steps", getattr(args, "door_push_steps", 300))),
        )
        args.traverse_steps = int(traverse_steps)
        if traverse.params.get("door_angle_target") is not None:
            args.pass_open_angle_deg = float(traverse.params["door_angle_target"])
        if traverse.params.get("distance") is not None:
            args.traverse_distance = max(0.0, float(traverse.params["distance"]))
        if traverse.params.get("base_v") is not None:
            args.traverse_distance = max(0.0, float(traverse.params["base_v"]) * float(traverse_steps) * dt)
        if traverse.params.get("yaw_delta") is not None:
            args.traverse_yaw_delta = float(traverse.params["yaw_delta"])

    release = program.primitive("ReleaseAndRetract")
    if release is not None:
        args.return_home_steps = _seconds_or_steps(
            release.params.get("duration", release.params.get("duration_steps")),
            dt=dt,
            default_steps=int(getattr(args, "return_home_steps", 150)),
        )

    for move_to in program.primitives_named("MoveTo"):
        params = move_to.params
        stage = str(params.get("stage", "approach")).strip().lower()
        if stage not in ("approach", "push", "traverse"):
            continue
        vx = float(params.get("vx", 0.0))
        vyaw = float(params.get("vyaw", 0.0))
        duration_value = params.get("duration", params.get("duration_steps"))
        if stage == "approach":
            if params.get("stop_distance") is not None:
                args.stop_distance = float(params["stop_distance"])
            if vx > 0.0:
                args.walk_min_speed = vx
            args.move_to_approach_vx = vx
            args.move_to_approach_vyaw = vyaw
            if duration_value is not None:
                args.walk_steps = _seconds_or_steps(
                    duration_value,
                    dt=dt,
                    default_steps=int(getattr(args, "walk_steps", 260)),
                )
                args.no_dynamic_walk_steps = True
                args.move_to_approach_yaw_delta = vyaw * float(args.walk_steps) * dt
            elif params.get("yaw_delta") is not None:
                args.move_to_approach_yaw_delta = float(params["yaw_delta"])
        elif stage == "push":
            distance = params.get("distance")
            if duration_value is not None:
                move_steps = _seconds_or_steps(
                    duration_value,
                    dt=dt,
                    default_steps=int(getattr(args, "door_push_steps", 300)),
                )
            elif distance is not None and abs(vx) > 1.0e-6:
                move_steps = max(1, int(round(abs(float(distance) / vx) / dt)))
            else:
                move_steps = int(getattr(args, "door_push_steps", 300))
            if distance is None and vx != 0.0:
                distance = abs(vx) * float(move_steps) * dt
            if distance is not None:
                args.push_base_distance = float(distance)
            yaw_delta = params.get("yaw_delta")
            if yaw_delta is None:
                yaw_delta = vyaw * float(move_steps) * dt
            args.push_base_yaw_delta = float(yaw_delta)
            args.move_to_push_vx = vx
            args.move_to_push_vyaw = vyaw
            args.move_to_push_steps = int(move_steps)
        else:
            args.pass_through_door = True
            distance = params.get("distance")
            if duration_value is not None:
                move_steps = _seconds_or_steps(
                    duration_value,
                    dt=dt,
                    default_steps=int(getattr(args, "traverse_steps", getattr(args, "door_push_steps", 300))),
                )
            elif distance is not None and abs(vx) > 1.0e-6:
                move_steps = max(1, int(round(abs(float(distance) / vx) / dt)))
            else:
                move_steps = int(getattr(args, "traverse_steps", getattr(args, "door_push_steps", 300)))
            if distance is None and vx != 0.0:
                distance = abs(vx) * float(move_steps) * dt
            if distance is not None:
                args.traverse_distance = float(distance)
            yaw_delta = params.get("yaw_delta")
            if yaw_delta is None:
                yaw_delta = vyaw * float(move_steps) * dt
            args.traverse_yaw_delta = float(yaw_delta)
            args.move_to_traverse_vx = vx
            args.move_to_traverse_vyaw = vyaw
            args.traverse_steps = int(move_steps)
            if stage == "traverse":
                args.pass_through_door = True


def _local_vector(local_xyz: list[float], approach_dir: np.ndarray, lateral_dir: np.ndarray) -> np.ndarray:
    local = np.asarray(local_xyz, dtype=np.float32)
    return (
        approach_dir * float(local[0])
        + lateral_dir * float(local[1])
        + np.asarray([0.0, 0.0, float(local[2])], dtype=np.float32)
    )


def compute_skill_waypoints(
    handle_goal: np.ndarray,
    approach_dir: np.ndarray,
    profile: SkillTrajectoryProfile,
) -> SkillWaypoints:
    approach = np.asarray(approach_dir, dtype=np.float32).copy()
    approach[2] = 0.0
    norm = float(np.linalg.norm(approach))
    if norm < 1.0e-6:
        approach = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        approach /= norm
    lateral = np.cross(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), approach)
    lateral_norm = float(np.linalg.norm(lateral))
    lateral = lateral / lateral_norm if lateral_norm >= 1.0e-6 else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    goal = np.asarray(handle_goal, dtype=np.float32) + np.asarray(profile.handle_goal_bias_world, dtype=np.float32)
    pregrasp = goal + _local_vector(profile.pregrasp_local, approach, lateral)
    grasp = goal + _local_vector(profile.grasp_local, approach, lateral)
    rotate = grasp + _local_vector(profile.rotate_local_delta, approach, lateral)
    return SkillWaypoints(pregrasp=pregrasp, grasp=grasp, rotate=rotate, lateral_dir=lateral)


PATCH_BOUNDS = {
    "MoveTo:approach.vx": (0.02, 0.80),
    "MoveTo:approach.vyaw": (-1.20, 1.20),
    "MoveTo:approach.stop_distance": (0.05, 0.80),
    "MoveTo:push.vx": (0.0, 0.60),
    "MoveTo:push.vyaw": (-1.20, 1.20),
    "MoveTo:push.distance": (0.0, 1.50),
    "MoveTo:traverse.vx": (0.0, 0.80),
    "MoveTo:traverse.vyaw": (-1.20, 1.20),
    "MoveTo:traverse.distance": (0.0, 2.50),
    "ApproachDoor.base_offset": (0.05, 0.80),
    "MoveEEToHandle.pregrasp_offset": ([-0.30, -0.25, -0.20], [0.45, 0.25, 0.20]),
    "MoveEEToHandle.grasp_offset": ([-0.20, -0.20, -0.20], [0.25, 0.20, 0.20]),
    "MoveEEToHandle.handle_goal_bias_world": ([-0.08, -0.08, -0.08], [0.08, 0.08, 0.08]),
    "RotateHandle.angle": (0.05, 1.50),
    "RotateHandle.duration_steps": (20, 250),
    "PushDoor.distance": (0.10, 2.00),
    "PushDoor.contact_bias": (0.0, 0.12),
    "PushDoor.duration_steps": (50, 600),
    "TraverseDoor.door_angle_target": (30.0, 100.0),
}


def _clamp_value(value: Any, bounds: tuple[Any, Any]) -> Any:
    lo, hi = bounds
    if isinstance(lo, list):
        values = _as_float_list(value, len(lo), [0.0] * len(lo))
        return [float(np.clip(v, float(lo_i), float(hi_i))) for v, lo_i, hi_i in zip(values, lo, hi)]
    if isinstance(lo, int) and isinstance(hi, int):
        return int(np.clip(int(round(float(value))), lo, hi))
    return float(np.clip(float(value), float(lo), float(hi)))


@dataclass
class ProgramPatch:
    patch: dict[str, Any]
    diagnostics: str = ""

    @classmethod
    def from_obj(cls, obj: Any) -> "ProgramPatch":
        if not isinstance(obj, dict):
            raise ValueError("ProgramPatch must be a JSON object")
        patch = obj.get("patch", obj)
        if not isinstance(patch, dict):
            raise ValueError("ProgramPatch.patch must be an object")
        return cls(patch=dict(patch), diagnostics=str(obj.get("diagnostics", "")))

    def apply(self, program: SkillProgram) -> SkillProgram:
        result = SkillProgram.from_obj(program.to_dict())
        by_name = {primitive.name: primitive for primitive in result.primitives}
        applied = False
        for path, value in self.patch.items():
            path = str(path)
            if path not in PATCH_BOUNDS:
                continue
            applied = True
            primitive_name, param_name = path.rsplit(".", 1)
            stage = None
            if ":" in primitive_name:
                primitive_name, stage = primitive_name.split(":", 1)
            primitive = None
            if stage is not None:
                primitive = next(
                    (
                        candidate
                        for candidate in result.primitives
                        if candidate.name == primitive_name
                        and str(candidate.params.get("stage", "")).strip().lower() == stage
                    ),
                    None,
                )
            else:
                primitive = by_name.get(primitive_name)
            if primitive is None:
                params = {} if stage is None else {"stage": stage}
                primitive = SkillPrimitive(primitive_name, params)
                result.primitives.append(primitive)
                if stage is None:
                    by_name[primitive_name] = primitive
            primitive.params[param_name] = _clamp_value(value, PATCH_BOUNDS[path])
        if applied:
            result.metadata["execution_mode"] = "skill_interpreter"
        result.metadata["last_patch_diagnostics"] = self.diagnostics
        return result

    def clamped_patch(self) -> dict[str, Any]:
        clamped = {}
        for path, value in self.patch.items():
            path = str(path)
            if path in PATCH_BOUNDS:
                clamped[path] = _clamp_value(value, PATCH_BOUNDS[path])
        return clamped


def heuristic_patches_for_failure(failure_stage: str) -> list[ProgramPatch]:
    """Small deterministic fallback when no VLM observer is connected."""

    if failure_stage == "grasp_miss":
        return [
            ProgramPatch({"MoveEEToHandle.pregrasp_offset": [0.18, 0.0, -0.02]}, "move pregrasp slightly farther out"),
            ProgramPatch({"MoveEEToHandle.handle_goal_bias_world": [-0.02, 0.0, 0.02]}, "bias handle goal toward current legacy successful grasp"),
        ]
    if failure_stage == "handle_not_unlocked":
        return [
            ProgramPatch({"RotateHandle.angle": 1.20, "RotateHandle.duration_steps": 140}, "increase handle rotation"),
        ]
    if failure_stage in ("door_push_insufficient", "contact_lost"):
        return [
            ProgramPatch(
                {
                    "PushDoor.distance": 1.35,
                    "PushDoor.duration_steps": 380,
                    "MoveTo:push.distance": 0.40,
                    "MoveTo:push.vx": 0.06,
                },
                "push longer with coordinated base motion",
            ),
            ProgramPatch({"PushDoor.contact_bias": 0.04}, "hold firmer contact with handle"),
        ]
    if failure_stage == "base_collision":
        return [
            ProgramPatch(
                {
                    "ApproachDoor.base_offset": 0.25,
                    "MoveTo:approach.stop_distance": 0.25,
                    "MoveTo:push.distance": 0.20,
                    "MoveTo:push.vx": 0.03,
                },
                "stand farther from the door and reduce coordinated base push",
            ),
        ]
    if failure_stage == "body_blocked":
        return [
            ProgramPatch(
                {
                    "TraverseDoor.door_angle_target": 75.0,
                    "MoveTo:traverse.distance": 0.80,
                    "MoveTo:traverse.vx": 0.18,
                },
                "open farther before traversal",
            ),
        ]
    return []
