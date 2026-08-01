#!/usr/bin/env python3
"""Select ten unreferenced AIGC doors and write a formal benchmark manifest."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from door_twin.benchmark.schema import (  # noqa: E402
    ABLATION_METHODS,
    SCHEMA_VERSION,
    BenchmarkManifest,
    Candidate,
    DoorCase,
    hidden_ground_truth_from_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CFG = REPO_ROOT / "high-level/data/cfg/b1z1_opendoor.yaml"
DEFAULT_RUNNER = REPO_ROOT / "high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py"
KNOWN_REGRESSION_DOORS = {
    "wc4",
    "button_door",
    "fire_door",
    "glass_door",
    "99692809960048",
    "99650089960001",
    "99650089960006",
    "99655039960001",
    "99655039960006",
}
UNTUNED_KEYS = {"name", "path", "bounding_box", "handle_bounding"}


def leaked_door_names() -> set[str]:
    door_twin = REPO_ROOT / "high-level/float_ik/door_twin"
    roots = [door_twin / "examples", door_twin / "doc", door_twin / "experiments"]
    chunks = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in (".md", ".json", ".yaml", ".yml", ".log", ".txt"):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    if "door_twin_agent_benchmark_v1" in text or path.name.endswith("_selection_audit.json"):
                        continue
                    chunks.append(text)
                except OSError:
                    continue
    text = "\n".join(chunks)
    numeric = set(re.findall(r"(?<!\d)\d{14}(?!\d)", text))
    named = {name for name in KNOWN_REGRESSION_DOORS if name in text}
    return numeric | named


def eligible_names(cfg_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    asset = cfg["env"]["asset"]
    leaked = leaked_door_names()
    asset_root = cfg_path.parents[2] / str(asset["assetRoot"]) / str(asset.get("assetFileDoor", ""))
    accepted: list[str] = []
    audit: list[dict[str, Any]] = []
    for block, entries in asset["trainAssets"].items():
        for entry_id, entry in entries.items():
            name = str(entry.get("name", ""))
            reasons = []
            if not name or name in KNOWN_REGRESSION_DOORS:
                reasons.append("known_regression_or_empty")
            if name and name in leaked:
                reasons.append("referenced_in_door_twin_history")
            if set(entry) - UNTUNED_KEYS:
                reasons.append("contains_hand_tuned_config_fields")
            try:
                urdf_path = asset_root / str(entry["path"])
                urdf = ET.parse(urdf_path).getroot()
                movable = [
                    joint
                    for joint in urdf.findall("joint")
                    if str(joint.get("type", "")) in ("revolute", "continuous", "prismatic")
                ]
                if len(movable) < 1:
                    reasons.append("missing_movable_door_joint")
                handle_path = asset_root / str(entry["handle_bounding"])
                handle = json.loads(handle_path.read_text(encoding="utf-8"))
                if "goal_pos" not in handle:
                    reasons.append("missing_handle_goal_annotation")
            except Exception as exc:
                reasons.append(f"asset_parse_failed:{type(exc).__name__}")
            audit.append({"name": name, "block": str(block), "entry_id": str(entry_id), "accepted": not reasons, "reasons": reasons})
            if not reasons:
                accepted.append(name)
    return sorted(set(accepted)), audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output manifest YAML")
    parser.add_argument("--base_door_cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--runner_script", default=str(DEFAULT_RUNNER))
    parser.add_argument("--benchmark_name", default="door_twin_fresh10_ablation")
    parser.add_argument("--run_root", default="", help="Defaults beside the manifest under runs/")
    parser.add_argument("--selection_seed", type=int, default=615455575)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    cfg_path = Path(args.base_door_cfg).expanduser().resolve()
    runner = Path(args.runner_script).expanduser().resolve()
    names, audit = eligible_names(cfg_path)
    if len(names) < 10:
        raise RuntimeError(f"Only {len(names)} leakage-free untuned doors are available")
    rng = random.Random(args.selection_seed)
    selected = sorted(rng.sample(names, 10))
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else (output.parent / "runs").resolve()
    doors = []
    selection_manifest = BenchmarkManifest(
        path=output,
        name=args.benchmark_name,
        base_door_cfg=cfg_path,
        runner_script=runner,
        output_root=run_root,
        doors=[],
    )
    for name in selected:
        door = DoorCase(id=name, door_name=name)
        candidate = Candidate.from_base_config(selection_manifest, door, 0, sanitize_handle_goal=False)
        doors.append(
            {
                "id": name,
                "door_name": name,
                "tags": ["fresh", "aigc", "hinged_lever_push"],
                "hidden_ground_truth": hidden_ground_truth_from_candidate(candidate),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": args.benchmark_name,
        "base_door_cfg": str(cfg_path),
        "runner_script": str(runner),
        "output_root": str(run_root),
        "python_executable": args.python_executable,
        "candidate_repeats": 3,
        "max_repair_rounds": 5,
        "development_seeds": list(range(41001, 41017)),
        "heldout_seeds": list(range(42001, 42017)),
        "generation_success_min": 12,
        "asset_probe_seed": 43001,
        "asset_probe_steps": 100,
        "asset_probe_force": 20.0,
        "steps": 2405,
        "methods": list(ABLATION_METHODS),
        "vlm": {
            "model": "gpt-5",
            "temperature": 0.0,
            "api_key_env": "OPENAI_API_KEY",
            "timeout_s": 180,
            "max_retries": 3,
        },
        "common_runner_args": [
            "--headless",
            "--rl_device",
            args.rl_device,
            "--sim_device",
            args.sim_device,
            "--graphics_device_id",
            str(args.graphics_device_id),
            "--no_enable_depth_noise",
            "--no_enable_depth_gaussian_blur",
            "--no_show_seg",
        ],
        "doors": doors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output.parent / f"{output.stem}_selection_audit.json").write_text(
        json.dumps({"selection_seed": args.selection_seed, "selected": selected, "audit": audit}, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = BenchmarkManifest(
        path=output,
        name=args.benchmark_name,
        base_door_cfg=cfg_path,
        runner_script=runner,
        output_root=run_root,
        doors=[DoorCase(id=item["id"], door_name=item["door_name"], hidden_ground_truth=item["hidden_ground_truth"]) for item in doors],
    )
    loaded.validate()
    print(f"Wrote formal 10-door manifest to {output}")
    print("Selected doors:", ", ".join(selected))


if __name__ == "__main__":
    main()
