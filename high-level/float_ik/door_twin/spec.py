"""Door asset structure extraction for the door digital-twin loop."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


HIGH_LEVEL_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _joint_child(joint: ET.Element) -> str:
    child = joint.find("child")
    return "" if child is None else str(child.get("link", ""))


def _movable_joints(urdf_path: Path) -> list[ET.Element]:
    root = ET.parse(urdf_path).getroot()
    return [
        joint
        for joint in root.findall("joint")
        if str(joint.get("type", "")).lower() in ("revolute", "continuous", "prismatic")
    ]


@dataclass
class DoorTwinSpec:
    """Stable, JSON-serializable structure summary for one simulated door."""

    schema_version: str
    name: str
    asset_root: str
    asset_file_door: str
    asset_path: str
    bounding_box: dict[str, Any]
    handle_bounding: dict[str, Any]
    actor_scale: float
    actor_yaw: float
    actor_yaw_offset: float
    actor_position_offset: list[float]
    robot_y_offset: float
    door_body_name: str
    handle_body_name: str
    door_dof_name: str
    handle_dof_name: str
    handle_goal_pos: list[float]
    handle_side: str
    generated_variant: str
    door_motion_sign_multiplier: float
    robot_alignment_y_offset: float | None
    robot_alignment_handle: dict[str, Any] | None
    supported: bool
    unsupported_reason: str = ""

    @classmethod
    def from_config_entry(
        cls,
        asset_root: str | Path,
        asset_file_door: str,
        spec: dict[str, Any],
        bounding: dict[str, Any],
        handle_bounding: dict[str, Any],
    ) -> "DoorTwinSpec":
        path = str(spec.get("path", ""))
        door_dof_name = str(spec.get("door_dof_name", ""))
        handle_dof_name = str(spec.get("handle_dof_name", ""))
        door_body_name = str(spec.get("door_body_name", ""))
        handle_body_name = str(spec.get("handle_body_name", ""))
        actor_yaw_offset = float(spec.get("actor_yaw_offset", 0.0))
        actor_position_offset = [float(v) for v in spec.get("actor_position_offset", (0.0, 0.0, 0.0))]
        if len(actor_position_offset) != 3:
            actor_position_offset = [0.0, 0.0, 0.0]
        supported = bool(path and "goal_pos" in handle_bounding)
        reason = "" if supported else "missing asset path or handle_bounding.goal_pos"
        return cls(
            schema_version="door_twin_spec_v1",
            name=str(spec.get("name", Path(path).parent.name or "door")),
            asset_root=str(asset_root),
            asset_file_door=str(asset_file_door or ""),
            asset_path=path,
            bounding_box=dict(bounding),
            handle_bounding=dict(handle_bounding),
            actor_scale=float(spec.get("actor_scale", 1.0)),
            actor_yaw=math.pi + actor_yaw_offset,
            actor_yaw_offset=actor_yaw_offset,
            actor_position_offset=actor_position_offset,
            robot_y_offset=float(spec.get("robot_y_offset", 0.0)),
            door_body_name=door_body_name,
            handle_body_name=handle_body_name,
            door_dof_name=door_dof_name,
            handle_dof_name=handle_dof_name,
            handle_goal_pos=[float(v) for v in handle_bounding.get("goal_pos", (0.0, 0.0, 0.0))],
            handle_side=str(spec.get("handle_side", "")),
            generated_variant=str(spec.get("generated_variant", "")),
            door_motion_sign_multiplier=float(spec.get("door_motion_sign_multiplier", 1.0)),
            robot_alignment_y_offset=(
                None
                if spec.get("robot_alignment_y_offset") is None
                else float(spec.get("robot_alignment_y_offset"))
            ),
            robot_alignment_handle=spec.get("robot_alignment_handle"),
            supported=supported,
            unsupported_reason=reason,
        )

    @classmethod
    def from_runtime(cls, door: Any) -> "DoorTwinSpec":
        """Build a spec from the DoorRuntime object used by Isaac Gym."""

        spec = dict(getattr(door, "spec", {}) or {})
        result = cls.from_config_entry(
            getattr(door, "asset_root", ""),
            getattr(door, "asset_file_door", ""),
            spec,
            dict(getattr(door, "bounding", {}) or {}),
            dict(getattr(door, "handle_bounding", {}) or {}),
        )
        result.actor_scale = float(getattr(door, "actor_scale", result.actor_scale))
        result.actor_yaw = float(getattr(door, "actor_yaw", result.actor_yaw))
        result.actor_position_offset = [
            float(v) for v in getattr(door, "actor_position_offset", result.actor_position_offset)
        ]
        result.robot_y_offset = float(getattr(door, "robot_y_offset", result.robot_y_offset))
        return result

    @classmethod
    def from_aigc_export_dir(cls, export_dir: str | Path) -> "DoorTwinSpec":
        """Extract a spec directly from a single AIGC export directory."""

        root = Path(export_dir).expanduser().resolve()
        bounding = _read_json(root / "bounding_box.json")
        handle_bounding = _read_json(root / "handle_bounding.json")
        joints = _movable_joints(root / "mobility.urdf")
        door_joint = joints[0] if joints else None
        handle_joint = joints[1] if len(joints) > 1 else None
        spec = {
            "name": root.name,
            "path": "mobility.urdf",
            "bounding_box": "bounding_box.json",
            "handle_bounding": "handle_bounding.json",
            "door_dof_name": "" if door_joint is None else str(door_joint.get("name", "")),
            "handle_dof_name": "" if handle_joint is None else str(handle_joint.get("name", "")),
            "door_body_name": "" if door_joint is None else _joint_child(door_joint),
            "handle_body_name": "" if handle_joint is None else _joint_child(handle_joint),
        }
        result = cls.from_config_entry(root, "", spec, bounding, handle_bounding)
        if len(joints) < 2:
            result.supported = False
            result.unsupported_reason = "AIGC export must expose door and handle movable joints"
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_specs_from_config(
    cfg_path: str | Path,
    *,
    door_name: str = "",
    door_index: int = -1,
) -> list[DoorTwinSpec]:
    """Load DoorTwinSpec objects from a DoorGym YAML config without Isaac Gym."""

    cfg_path = Path(cfg_path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (HIGH_LEVEL_ROOT.parents[0] / cfg_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    asset_cfg = cfg["env"]["asset"]
    train_assets = asset_cfg["trainAssets"]
    load_block = asset_cfg.get("load_block") or next(iter(train_assets.keys()))
    entries = sorted(train_assets[load_block].items(), key=lambda item: int(item[0]))
    if door_name:
        entries = [(idx, spec) for idx, spec in entries if spec.get("name") == door_name]
    elif int(door_index) >= 0:
        entries = [entries[int(door_index)]]

    asset_root = HIGH_LEVEL_ROOT / asset_cfg["assetRoot"]
    asset_file_door = str(asset_cfg.get("assetFileDoor", "") or "")
    door_set_root = asset_root / asset_file_door
    specs: list[DoorTwinSpec] = []
    for _idx, spec in entries:
        bounding = _read_json(door_set_root / spec["bounding_box"])
        handle_bounding = _read_json(door_set_root / spec["handle_bounding"])
        specs.append(DoorTwinSpec.from_config_entry(asset_root, asset_file_door, spec, bounding, handle_bounding))
    return specs
