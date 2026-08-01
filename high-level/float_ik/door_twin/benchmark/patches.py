"""Bounded asset/config and skill patches returned by the DoorTwin observer."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..skill import ProgramPatch
from .schema import Candidate


ASSET_NUMERIC_BOUNDS: dict[str, tuple[Any, Any]] = {
    "actor_scale": (0.25, 3.0),
    "actor_yaw_offset": (-math.pi, math.pi),
    "actor_position_offset": ([-2.0, -2.0, -1.0], [2.0, 2.0, 1.0]),
    "robot_y_offset": (-1.5, 1.5),
    "robot_alignment_y_offset": (-1.5, 1.5),
    "bounds_yaw_offset_override": (-math.pi, math.pi),
    "handle_goal_pos": ([-2.0, -2.0, -1.0], [2.0, 2.0, 2.5]),
}
ASSET_NAME_FIELDS = {"door_body_name", "handle_body_name", "door_dof_name", "handle_dof_name"}
ASSET_ALLOWED_FIELDS = set(ASSET_NUMERIC_BOUNDS) | ASSET_NAME_FIELDS | {"door_motion_sign_multiplier"}


def _clamp(value: Any, bounds: tuple[Any, Any]) -> Any:
    lo, hi = bounds
    if isinstance(lo, list):
        if not isinstance(value, (list, tuple)) or len(value) != len(lo):
            raise ValueError(f"Expected {len(lo)} values, got {value!r}")
        return [float(np.clip(float(v), float(a), float(b))) for v, a, b in zip(value, lo, hi)]
    return float(np.clip(float(value), float(lo), float(hi)))


@dataclass
class AssetConfigPatch:
    patch: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any) -> "AssetConfigPatch":
        if obj is None:
            return cls()
        if not isinstance(obj, dict):
            raise ValueError("asset_patch must be a JSON object")
        return cls(dict(obj))

    def validated(self, candidate: Candidate) -> tuple[dict[str, Any], list[str]]:
        structure = candidate.urdf_structure()
        links = set(structure["links"])
        joints = {joint["name"]: joint for joint in structure["joints"]}
        movable = {name for name, joint in joints.items() if joint["type"] in ("revolute", "continuous", "prismatic")}
        clean: dict[str, Any] = {}
        rejected: list[str] = []
        for key, value in self.patch.items():
            key = str(key)
            if key not in ASSET_ALLOWED_FIELDS:
                rejected.append(key)
                continue
            if key in ASSET_NUMERIC_BOUNDS:
                clean[key] = _clamp(value, ASSET_NUMERIC_BOUNDS[key])
            elif key == "door_motion_sign_multiplier":
                clean[key] = 1.0 if float(value) >= 0.0 else -1.0
            elif key in ("door_body_name", "handle_body_name"):
                name = str(value)
                if name and name not in links:
                    rejected.append(key)
                else:
                    clean[key] = name
            elif key in ("door_dof_name", "handle_dof_name"):
                name = str(value)
                if name and name not in movable:
                    rejected.append(key)
                else:
                    clean[key] = name
        return clean, rejected

    def apply(self, candidate: Candidate) -> tuple[Candidate, dict[str, Any]]:
        result = candidate.clone()
        clean, rejected = self.validated(candidate)
        for key, value in clean.items():
            if key == "handle_goal_pos":
                result.handle_bounding["goal_pos"] = copy.deepcopy(value)
            else:
                result.runtime_spec[key] = copy.deepcopy(value)
        return result, {"accepted": clean, "rejected_fields": rejected}


@dataclass
class CombinedRepairPatch:
    asset_patch: AssetConfigPatch = field(default_factory=AssetConfigPatch)
    skill_patch: ProgramPatch = field(default_factory=lambda: ProgramPatch({}))
    diagnostics: str = ""
    failure_stage: str = ""

    @classmethod
    def from_obj(cls, obj: Any) -> "CombinedRepairPatch":
        if not isinstance(obj, dict):
            raise ValueError("VLM response must be a JSON object")
        skill_obj = obj.get("skill_patch", {}) or {}
        if not isinstance(skill_obj, dict):
            raise ValueError("skill_patch must be a JSON object")
        diagnostics = str(obj.get("diagnostics", ""))
        return cls(
            asset_patch=AssetConfigPatch.from_obj(obj.get("asset_patch", {})),
            skill_patch=ProgramPatch(skill_obj, diagnostics),
            diagnostics=diagnostics,
            failure_stage=str(obj.get("failure_stage", "")),
        )

    def apply(self, candidate: Candidate) -> tuple[Candidate, dict[str, Any]]:
        result, asset_info = self.asset_patch.apply(candidate)
        before_skill = result.skill_program.to_dict()
        accepted_skill = self.skill_patch.clamped_patch()
        if accepted_skill:
            result.skill_program = self.skill_patch.apply(result.skill_program)
        result.metadata["last_patch_diagnostics"] = self.diagnostics
        info = {
            "asset": asset_info,
            "accepted_skill": accepted_skill,
            "rejected_skill_fields": sorted(set(self.skill_patch.patch) - set(accepted_skill)),
            "skill_changed": result.skill_program.to_dict() != before_skill,
            "diagnostics": self.diagnostics,
            "failure_stage": self.failure_stage,
        }
        return result, info
