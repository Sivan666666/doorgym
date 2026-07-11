#!/usr/bin/env python3
"""Generate scripted recovery branch candidates from failure diagnoses.

This is the "Agent proposes" half of the recovery pipeline.  The output is a
candidate manifest only; IsaacGym verification must run these candidates and
keep only branches that satisfy Local Recovery Success AND Final Task Success.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


DEFAULT_RETREAT_DISTANCES = (0.03, 0.05, 0.08)
DEFAULT_REAPPROACH_Z_OFFSETS = (-0.02, 0.0, 0.02)
DEFAULT_BASE_LATERAL_ADJUSTS = (0.0,)
DEFAULT_HANDLE_X_OFFSETS = (-0.02, -0.01, 0.0)
# Match the original A2W scripted expert: transition directly from grasp to
# gripper closing, without inserting an extra grasp-hold delay.
DEFAULT_CLOSE_TIMINGS = ("on_contact",)
DEFAULT_ROTATE_STEPS = (100,)
DEFAULT_PUSH_STEPS = (300,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate recovery candidate manifests for failed Door ACT rollouts.")
    parser.add_argument(
        "--failure_rollout",
        type=str,
        default="",
        help="Path to one failure_rollout.npz or containing directory.",
    )
    parser.add_argument(
        "--failure_root",
        type=str,
        default="",
        help="Directory containing failure rollout bundles.",
    )
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--max_candidates_per_branch", type=int, default=20)
    parser.add_argument("--branch_offsets", type=str, default="-5,0,5,fail")
    parser.add_argument("--local_contact_min_frames", type=int, default=5)
    parser.add_argument("--final_open_angle_deg", type=float, default=80.0)
    return parser.parse_args()


def bundle_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    if args.failure_rollout:
        p = Path(args.failure_rollout).expanduser()
        dirs.append(p if p.is_dir() else p.parent)
    if args.failure_root:
        root = Path(args.failure_root).expanduser()
        dirs.extend(sorted(p.parent for p in root.glob("**/failure_rollout.npz")))
    unique: list[Path] = []
    seen = set()
    for d in dirs:
        rd = d.resolve()
        if rd not in seen:
            unique.append(rd)
            seen.add(rd)
    return unique


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_branch_offsets(value: str, t_dev: int, t_fail: int) -> list[int]:
    out: list[int] = []
    for token in str(value).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "fail":
            out.append(int(t_fail))
        else:
            out.append(int(t_dev) + int(token))
    return sorted({max(0, int(x)) for x in out})


def candidate_grid(failure_type: str, family: str):
    if failure_type == "insufficient_interaction" or family == "maintain_grasp_then_push":
        yield {
            "controller": "maintain_grasp_then_push",
            "open_gripper_first": False,
            "retreat_distance_m": 0.0,
            "reapproach_z_offset_m": 0.0,
            "base_lateral_adjust_m": 0.0,
            "handle_regrasp_x_offset_m": 0.0,
            "close_timing": "keep_closed",
            "rotate_steps": 100,
            "push_steps": 300,
        }
        for retreat, z_offset, handle_x, close_timing in itertools.product(
            DEFAULT_RETREAT_DISTANCES,
            DEFAULT_REAPPROACH_Z_OFFSETS,
            DEFAULT_HANDLE_X_OFFSETS,
            DEFAULT_CLOSE_TIMINGS,
        ):
            yield {
                "controller": "reopen_retreat_reapproach",
                "open_gripper_first": True,
                "retreat_distance_m": retreat,
                "reapproach_z_offset_m": z_offset,
                "base_lateral_adjust_m": 0.0,
                "handle_regrasp_x_offset_m": handle_x,
                "close_timing": close_timing,
                "rotate_steps": 100,
                "push_steps": 300,
            }
    elif failure_type == "geometric_misalignment" or family == "retreat_realign_reapproach":
        for retreat, z_offset, base_lateral, close_timing in itertools.product(
            DEFAULT_RETREAT_DISTANCES,
            DEFAULT_REAPPROACH_Z_OFFSETS,
            DEFAULT_BASE_LATERAL_ADJUSTS,
            DEFAULT_CLOSE_TIMINGS,
        ):
            yield {
                "controller": "retreat_realign_reapproach",
                "open_gripper_first": True,
                "retreat_distance_m": retreat,
                "reapproach_z_offset_m": z_offset,
                "base_lateral_adjust_m": base_lateral,
                "handle_regrasp_x_offset_m": -0.01,
                "close_timing": close_timing,
                "rotate_steps": 100,
                "push_steps": 300,
            }
    else:
        for open_first, retreat, z_offset, base_lateral, handle_x, close_timing in itertools.product(
            (True, False),
            DEFAULT_RETREAT_DISTANCES,
            DEFAULT_REAPPROACH_Z_OFFSETS,
            DEFAULT_BASE_LATERAL_ADJUSTS,
            DEFAULT_HANDLE_X_OFFSETS,
            DEFAULT_CLOSE_TIMINGS,
        ):
            yield {
                "controller": "reopen_retreat_reapproach" if open_first else "keep_grasp_realign_reapproach",
                "open_gripper_first": open_first,
                "retreat_distance_m": retreat,
                "reapproach_z_offset_m": z_offset,
                "base_lateral_adjust_m": base_lateral,
                "handle_regrasp_x_offset_m": handle_x,
                "close_timing": close_timing if open_first else "keep_closed",
                "rotate_steps": 100,
                "push_steps": 300,
            }


def spread_sample(values: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select deterministic, evenly spread points instead of a biased grid prefix."""
    if count >= len(values):
        return values
    if count <= 1:
        return [values[len(values) // 2]]
    indices = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
    return [values[int(idx)] for idx in indices]


def main() -> None:
    args = parse_args()
    if args.max_candidates_per_branch <= 0:
        raise ValueError("--max_candidates_per_branch must be positive.")
    dirs = bundle_dirs_from_args(args)
    if not dirs:
        raise ValueError("Provide --failure_rollout or --failure_root.")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "description": "Recovery candidates proposed for simulator verification; not training ground truth.",
        "verification_rule": {
            "local_recovery_success": {
                "contact_min_frames": int(args.local_contact_min_frames),
                "requires_progress_restart": True,
            },
            "final_task_success": {
                "door_open_angle_deg": float(args.final_open_angle_deg),
            },
            "keep_branch_if": "local_recovery_success AND final_task_success",
        },
        "bundles": [],
    }

    global_candidate_id = 0
    for bundle_id, bundle_dir in enumerate(dirs):
        npz_path = bundle_dir / "failure_rollout.npz"
        diagnosis_path = bundle_dir / "diagnosis.json"
        metadata_path = bundle_dir / "metadata.json"
        if not npz_path.is_file() or not diagnosis_path.is_file():
            raise FileNotFoundError(f"Missing failure_rollout.npz or diagnosis.json in {bundle_dir}")
        diagnosis = load_json(diagnosis_path)
        metadata = load_json(metadata_path) if metadata_path.is_file() else {}
        t_dev = int(diagnosis.get("t_dev", 0))
        t_fail = int(diagnosis.get("t_fail", t_dev))
        branch_steps = parse_branch_offsets(args.branch_offsets, t_dev, t_fail)

        full_candidate_grid = list(
            candidate_grid(
                str(diagnosis.get("failure_type", "contact_establishment_failure")),
                str(diagnosis.get("candidate_recovery_family", "")),
            )
        )
        base_candidates = spread_sample(full_candidate_grid, int(args.max_candidates_per_branch))
        branch_entries = []
        for branch_step in branch_steps:
            candidates = []
            for params in base_candidates:
                candidates.append(
                    {
                        "candidate_id": global_candidate_id,
                        "branch_step": int(branch_step),
                        "parameters": params,
                        "status": "proposed_unverified",
                    }
                )
                global_candidate_id += 1
            branch_entries.append({"branch_step": int(branch_step), "candidates": candidates})

        manifest["bundles"].append(
            {
                "bundle_id": int(bundle_id),
                "failure_rollout_npz": str(npz_path),
                "metadata_path": str(metadata_path),
                "diagnosis_path": str(diagnosis_path),
                "metadata": metadata,
                "diagnosis": diagnosis,
                "branches": branch_entries,
            }
        )

    out = Path(args.out_json).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    total = sum(len(branch["candidates"]) for bundle in manifest["bundles"] for branch in bundle["branches"])
    print(f"Recovery candidate manifest written to {out} ({total} candidates)", flush=True)


if __name__ == "__main__":
    main()
