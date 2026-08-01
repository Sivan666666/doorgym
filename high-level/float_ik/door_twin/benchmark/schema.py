"""Manifest and immutable candidate artifacts for DoorTwin agent benchmarks."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..skill import MIN_DOOR_TWIN_FORWARD_DISTANCE_M, SkillPrimitive, SkillProgram


SCHEMA_VERSION = "door_twin_agent_benchmark_v1"
RULE_BASED_INITIALIZATION_PROTOCOL = "rule_based_public_geometry_v1"
OURS_INITIALIZATION_PROTOCOL = "rule_based_plus_retrieval_residual_v1"
ABLATION_METHODS = (
    "rule_based",
    "ours",
    "without_visual_feedback",
    "without_vlm_fix",
    "validation_only_log",
    "validation_only",
    "without_simulation_rollout",
)
METHOD_LABELS = {
    "rule_based": "Rule-based",
    "ours": "Ours",
    "without_visual_feedback": "Ours w/ VLM Fix - Log",
    "without_vlm_fix": "Ours w/o VLM Fix (Deterministic Heuristic)",
    "validation_only_log": "Ours w/o VLM Fix (Log Diagnosis Only)",
    "validation_only": "Ours w/o VLM Fix (Log + Visual Diagnosis Only)",
    "without_simulation_rollout": "Ours w/o Simulation Rollout",
}


def rule_based_candidate(manifest: "BenchmarkManifest", door: "DoorCase", candidate_index: int) -> "Candidate":
    """Build a deterministic no-retrieval candidate from public geometry only."""

    candidate = Candidate.from_base_config(manifest, door, candidate_index)
    structure = candidate.urdf_structure()
    joints = [
        joint
        for joint in structure.get("joints", [])
        if joint.get("type") in {"revolute", "continuous", "prismatic"}
    ]
    link_text = {
        str(link.get("name", "")): " ".join(
            [str(link.get("name", ""))]
            + [
                f"{visual.get('name', '')} {visual.get('mesh', '')}"
                for visual in link.get("visuals", [])
            ]
        ).lower()
        for link in structure.get("link_details", [])
    }

    def angular_range(joint: dict[str, Any]) -> float:
        if joint.get("type") == "continuous":
            return 2.0 * math.pi
        lower, upper = joint.get("lower"), joint.get("upper")
        if lower is None or upper is None:
            return 0.0
        return abs(float(upper) - float(lower))

    def name_score(joint: dict[str, Any], tokens: tuple[str, ...]) -> tuple[int, float, str]:
        child = str(joint.get("child", ""))
        text = f"{joint.get('name', '')} {child} {link_text.get(child, '')}".lower()
        return (sum(token in text for token in tokens), angular_range(joint), str(joint.get("name", "")))

    door_joint = max(joints, key=lambda item: name_score(item, ("door", "hinge", "panel", "board")), default=None)
    remaining = [joint for joint in joints if joint is not door_joint]
    handle_joint = max(remaining, key=lambda item: name_score(item, ("handle", "lever", "spindle")), default=None)
    if door_joint is not None:
        candidate.runtime_spec["door_dof_name"] = str(door_joint["name"])
        candidate.runtime_spec["door_body_name"] = str(door_joint.get("child", ""))
    if handle_joint is not None and (
        name_score(handle_joint, ("handle", "lever", "spindle"))[0] > 0 or len(joints) > 1
    ):
        candidate.runtime_spec["handle_dof_name"] = str(handle_joint["name"])
        candidate.runtime_spec["handle_body_name"] = str(handle_joint.get("child", ""))
    else:
        candidate.runtime_spec["handle_dof_name"] = ""
        handle_links = [name for name in structure.get("links", []) if any(token in name.lower() for token in ("handle", "lever"))]
        candidate.runtime_spec["handle_body_name"] = handle_links[0] if handle_links else candidate.runtime_spec.get("door_body_name", "")
    candidate.metadata["initial_generation"] = {
        "mode": RULE_BASED_INITIALIZATION_PROTOCOL,
        "protocol": RULE_BASED_INITIALIZATION_PROTOCOL,
        "retrieval": {},
    }
    return candidate


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def find_config_entry(cfg: dict[str, Any], door_name: str) -> tuple[str, str, dict[str, Any]]:
    asset = cfg["env"]["asset"]
    blocks = asset["trainAssets"]
    for block_name, entries in blocks.items():
        for entry_id, entry in entries.items():
            if str(entry.get("name", "")) == str(door_name):
                return str(block_name), str(entry_id), copy.deepcopy(entry)
    raise KeyError(f"Door {door_name!r} was not found in the benchmark base config")


def generic_push_traverse_skill() -> SkillProgram:
    """Metadata-only initial skill; it contains no door-specific tuned values."""

    return SkillProgram(
        metadata={"source": "door_twin_benchmark_generic_v1", "execution_mode": "skill_interpreter"},
        primitives=[
            SkillPrimitive("MoveTo", {"stage": "approach", "vx": 0.20, "vyaw": 0.0, "stop_distance": 0.15}),
            SkillPrimitive("ApproachDoor", {"base_offset": 0.15}),
            SkillPrimitive(
                "MoveEEToHandle",
                {
                    "pregrasp_offset": [0.15, 0.0, -0.03],
                    "grasp_offset": [0.0, 0.0, -0.03],
                    "handle_goal_bias_world": [0.0, 0.0, 0.0],
                    "duration_steps": 50,
                },
            ),
            SkillPrimitive("CloseGripper", {"force": 0.8, "duration_steps": 50}),
            SkillPrimitive("RotateHandle", {"angle": 1.05, "duration_steps": 100, "local_delta": [0.0, 0.03, -0.03]}),
            SkillPrimitive("PushDoor", {"distance": 1.10, "duration_steps": 300, "contact_bias": 0.025}),
            SkillPrimitive(
                "MoveTo",
                {"stage": "push", "vx": 0.20, "vyaw": 0.0, "distance": 1.20, "duration_steps": 300},
            ),
            SkillPrimitive("TraverseDoor", {"door_angle_target": 80.0, "duration_steps": 300}),
            SkillPrimitive(
                "MoveTo",
                {
                    "stage": "traverse",
                    "vx": 0.20,
                    "vyaw": 0.0,
                    "distance": MIN_DOOR_TWIN_FORWARD_DISTANCE_M - 1.20,
                    "duration_steps": 185,
                },
            ),
            SkillPrimitive("ReleaseAndRetract", {"duration_steps": 150}),
        ],
    )


@dataclass(frozen=True)
class DoorCase:
    id: str
    door_name: str
    hidden_ground_truth: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass
class BenchmarkManifest:
    path: Path
    name: str
    base_door_cfg: Path
    runner_script: Path
    output_root: Path
    doors: list[DoorCase]
    candidate_repeats: int = 3
    max_repair_rounds: int = 5
    development_seeds: list[int] = field(default_factory=lambda: list(range(41001, 41017)))
    heldout_seeds: list[int] = field(default_factory=lambda: list(range(42001, 42017)))
    generation_success_min: int = 12
    asset_probe_seed: int = 43001
    asset_probe_steps: int = 100
    asset_probe_force: float = 20.0
    steps: int = 2405
    python_executable: str = "python"
    common_runner_args: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=lambda: list(ABLATION_METHODS))
    vlm: dict[str, Any] = field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def validate(self) -> None:
        if self.candidate_repeats < 1:
            raise ValueError("candidate_repeats must be positive")
        if self.max_repair_rounds < 0:
            raise ValueError("max_repair_rounds must be non-negative")
        if len(self.doors) != 10:
            raise ValueError(f"The formal benchmark requires exactly 10 fresh doors, got {len(self.doors)}")
        if len(set(door.id for door in self.doors)) != len(self.doors):
            raise ValueError("Door ids must be unique")
        unknown = sorted(set(self.methods) - set(ABLATION_METHODS))
        if unknown:
            raise ValueError(f"Unknown ablation method(s): {unknown}")
        if len(self.development_seeds) != 16:
            raise ValueError("The formal benchmark requires sixteen development seeds")
        if len(self.heldout_seeds) != 16:
            raise ValueError("The formal benchmark requires sixteen held-out seeds")
        if not 1 <= self.generation_success_min <= len(self.heldout_seeds):
            raise ValueError("generation_success_min is outside the held-out seed count")
        if self.asset_probe_steps < 100 or self.asset_probe_force <= 0.0:
            raise ValueError("asset probe requires at least 100 steps and a positive force")
        if not self.base_door_cfg.is_file():
            raise FileNotFoundError(self.base_door_cfg)
        if not self.runner_script.is_file():
            raise FileNotFoundError(self.runner_script)


def load_benchmark_manifest(path: str | Path, *, formal: bool = True) -> BenchmarkManifest:
    manifest_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    base = manifest_path.parent
    doors = [
        DoorCase(
            id=str(item.get("id", item["door_name"])),
            door_name=str(item["door_name"]),
            hidden_ground_truth=dict(item.get("hidden_ground_truth", {}) or {}),
            tags=tuple(str(tag) for tag in item.get("tags", [])),
        )
        for item in raw.get("doors", [])
    ]
    manifest = BenchmarkManifest(
        path=manifest_path,
        name=str(raw.get("name", manifest_path.stem)),
        base_door_cfg=_resolve_path(raw["base_door_cfg"], base),
        runner_script=_resolve_path(raw["runner_script"], base),
        output_root=_resolve_path(raw.get("output_root", "runs/agent_ablation"), base),
        doors=doors,
        candidate_repeats=int(raw.get("candidate_repeats", 3)),
        max_repair_rounds=int(raw.get("max_repair_rounds", 5)),
        development_seeds=[int(v) for v in raw.get("development_seeds", range(41001, 41017))],
        heldout_seeds=[int(v) for v in raw.get("heldout_seeds", range(42001, 42017))],
        generation_success_min=int(raw.get("generation_success_min", 12)),
        asset_probe_seed=int(raw.get("asset_probe_seed", 43001)),
        asset_probe_steps=int(raw.get("asset_probe_steps", 100)),
        asset_probe_force=float(raw.get("asset_probe_force", 20.0)),
        steps=int(raw.get("steps", 2405)),
        python_executable=str(raw.get("python_executable", "python")),
        common_runner_args=[str(v) for v in raw.get("common_runner_args", [])],
        methods=[str(v) for v in raw.get("methods", ABLATION_METHODS)],
        vlm=dict(raw.get("vlm", {}) or {}),
    )
    if formal:
        manifest.validate()
    return manifest


@dataclass
class Candidate:
    door_id: str
    door_name: str
    candidate_index: int
    runtime_spec: dict[str, Any]
    bounding_box: dict[str, Any]
    handle_bounding: dict[str, Any]
    skill_program: SkillProgram
    source_config: Path
    parent_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "door_id": self.door_id,
            "door_name": self.door_name,
            "candidate_index": self.candidate_index,
            "runtime_spec": self.runtime_spec,
            "bounding_box": self.bounding_box,
            "handle_bounding": self.handle_bounding,
            "skill_program": self.skill_program.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.payload())

    def clone(self) -> "Candidate":
        return Candidate(
            door_id=self.door_id,
            door_name=self.door_name,
            candidate_index=self.candidate_index,
            runtime_spec=copy.deepcopy(self.runtime_spec),
            bounding_box=copy.deepcopy(self.bounding_box),
            handle_bounding=copy.deepcopy(self.handle_bounding),
            skill_program=SkillProgram.from_obj(self.skill_program.to_dict()),
            source_config=self.source_config,
            parent_hash=self.fingerprint,
            metadata=copy.deepcopy(self.metadata),
        )

    @classmethod
    def from_base_config(
        cls,
        manifest: BenchmarkManifest,
        door: DoorCase,
        candidate_index: int,
        *,
        sanitize_handle_goal: bool = True,
    ) -> "Candidate":
        cfg = yaml.safe_load(manifest.base_door_cfg.read_text(encoding="utf-8")) or {}
        _block, _entry_id, entry = find_config_entry(cfg, door.door_name)
        asset_cfg = cfg["env"]["asset"]
        asset_root = manifest.base_door_cfg.parents[2] / str(asset_cfg["assetRoot"])
        door_root = asset_root / str(asset_cfg.get("assetFileDoor", ""))
        bounding_path = door_root / str(entry["bounding_box"])
        handle_path = door_root / str(entry["handle_bounding"])
        handle_bounding = _read_json(handle_path)
        if sanitize_handle_goal:
            handle_min = handle_bounding.get("handle_min")
            handle_max = handle_bounding.get("handle_max")
            if isinstance(handle_min, list) and isinstance(handle_max, list) and len(handle_min) == len(handle_max) == 3:
                handle_bounding["goal_pos"] = [0.5 * (float(lo) + float(hi)) for lo, hi in zip(handle_min, handle_max)]
        return cls(
            door_id=door.id,
            door_name=door.door_name,
            candidate_index=candidate_index,
            runtime_spec=entry,
            bounding_box=_read_json(bounding_path),
            handle_bounding=handle_bounding,
            skill_program=generic_push_traverse_skill(),
            source_config=manifest.base_door_cfg,
            metadata={"base_entry_id": _entry_id, "base_block": _block},
        )

    def asset_path(self) -> Path:
        cfg = yaml.safe_load(self.source_config.read_text(encoding="utf-8")) or {}
        asset_cfg = cfg["env"]["asset"]
        root = self.source_config.parents[2] / str(asset_cfg["assetRoot"]) / str(asset_cfg.get("assetFileDoor", ""))
        path = Path(str(self.runtime_spec["path"])).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    def urdf_structure(self) -> dict[str, Any]:
        root = ET.parse(self.asset_path()).getroot()
        links = [str(link.get("name", "")) for link in root.findall("link")]
        link_details = []
        for link in root.findall("link"):
            visuals = []
            for visual in link.findall("visual"):
                mesh = visual.find("geometry/mesh")
                visuals.append(
                    {
                        "name": str(visual.get("name", "")),
                        "mesh": "" if mesh is None else str(mesh.get("filename", "")),
                    }
                )
            link_details.append({"name": str(link.get("name", "")), "visuals": visuals})
        joints = []
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            child = joint.find("child")
            axis = joint.find("axis")
            joints.append(
                {
                    "name": str(joint.get("name", "")),
                    "type": str(joint.get("type", "")),
                    "child": "" if child is None else str(child.get("link", "")),
                    "axis": "" if axis is None else str(axis.get("xyz", "")),
                    "lower": None if limit is None or limit.get("lower") is None else float(limit.get("lower")),
                    "upper": None if limit is None or limit.get("upper") is None else float(limit.get("upper")),
                }
            )
        return {"links": links, "link_details": link_details, "joints": joints}

    def public_context(self) -> dict[str, Any]:
        public_spec_keys = ("name", "path", "actor_scale", "actor_yaw_offset", "actor_position_offset")
        handle_public = {key: value for key, value in self.handle_bounding.items() if key != "goal_pos"}
        return {
            "door_id": self.door_id,
            "runtime_spec": {key: copy.deepcopy(self.runtime_spec[key]) for key in public_spec_keys if key in self.runtime_spec},
            "bounding_box": copy.deepcopy(self.bounding_box),
            "handle_bounding_without_goal": handle_public,
            "urdf_structure": self.urdf_structure(),
            "current_candidate_skill_program": self.skill_program.to_dict(),
        }

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        handle_path = (out / "handle_bounding.json").resolve()
        handle_path.write_text(json.dumps(self.handle_bounding, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        skill_path = (out / "skill_program.json").resolve()
        skill_path.write_text(json.dumps(self.skill_program.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        cfg = yaml.safe_load(self.source_config.read_text(encoding="utf-8")) or {}
        asset = cfg["env"]["asset"]
        block = str(asset.get("load_block") or next(iter(asset["trainAssets"])))
        entry = copy.deepcopy(self.runtime_spec)
        entry["name"] = self.door_name
        entry["handle_bounding"] = str(handle_path)
        asset["load_block"] = block
        asset["trainAssets"] = {block: {"0": entry}}
        cfg_path = (out / "door_cfg.yaml").resolve()
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

        record = {
            "schema_version": "door_twin_candidate_v1",
            "fingerprint": self.fingerprint,
            "parent_hash": self.parent_hash,
            "payload": self.payload(),
            "metadata": self.metadata,
            "paths": {"door_cfg": str(cfg_path), "skill_program": str(skill_path), "handle_bounding": str(handle_path)},
        }
        candidate_path = (out / "candidate.json").resolve()
        candidate_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"candidate": candidate_path, "door_cfg": cfg_path, "skill_program": skill_path, "handle_bounding": handle_path}

    @classmethod
    def read(cls, path: str | Path, source_config: str | Path) -> "Candidate":
        raw = _read_json(Path(path))
        payload = raw["payload"]
        candidate = cls(
            door_id=str(payload["door_id"]),
            door_name=str(payload["door_name"]),
            candidate_index=int(payload["candidate_index"]),
            runtime_spec=dict(payload["runtime_spec"]),
            bounding_box=dict(payload["bounding_box"]),
            handle_bounding=dict(payload["handle_bounding"]),
            skill_program=SkillProgram.from_obj(payload["skill_program"]),
            source_config=Path(source_config).resolve(),
            parent_hash=str(raw.get("parent_hash", "")),
            metadata=dict(raw.get("metadata", {}) or {}),
        )
        expected = str(raw.get("fingerprint", ""))
        if expected and candidate.fingerprint != expected:
            raise ValueError(f"Candidate fingerprint mismatch: {path}")
        return candidate


def hidden_ground_truth_from_candidate(candidate: Candidate) -> dict[str, Any]:
    """Create a hidden annotation for a selected fresh asset; never include this in VLM prompts."""

    structure = candidate.urdf_structure()
    movable = [joint for joint in structure["joints"] if joint["type"] in ("revolute", "continuous", "prismatic")]
    fixed_handle_body = ""
    if len(movable) < 2:
        for link in structure.get("link_details", []):
            searchable = " ".join(
                [str(link.get("name", ""))]
                + [str(visual.get("name", "")) + " " + str(visual.get("mesh", "")) for visual in link.get("visuals", [])]
            ).lower()
            if "handle" in searchable:
                fixed_handle_body = str(link.get("name", ""))
                break
    return {
        "door_body_name": candidate.runtime_spec.get("door_body_name", movable[0]["child"] if movable else ""),
        "handle_body_name": candidate.runtime_spec.get(
            "handle_body_name", movable[1]["child"] if len(movable) > 1 else fixed_handle_body
        ),
        "door_dof_name": candidate.runtime_spec.get("door_dof_name", movable[0]["name"] if movable else ""),
        "handle_dof_name": candidate.runtime_spec.get("handle_dof_name", movable[1]["name"] if len(movable) > 1 else ""),
        "handle_goal_pos": [float(v) for v in candidate.handle_bounding.get("goal_pos", (0.0, 0.0, 0.0))],
        "handle_goal_reference_type": "generated_initial_candidate",
        "handle_goal_verified": False,
    }


def hinge_range_degrees(candidate: Candidate) -> float:
    requested = str(candidate.runtime_spec.get("door_dof_name", ""))
    movable = [j for j in candidate.urdf_structure()["joints"] if j["type"] in ("revolute", "continuous")]
    joint = next((j for j in movable if j["name"] == requested), movable[0] if movable else None)
    if joint is None:
        return 0.0
    if joint["type"] == "continuous":
        return 360.0
    if joint["lower"] is None or joint["upper"] is None:
        return 0.0
    return math.degrees(abs(float(joint["upper"]) - float(joint["lower"])))
