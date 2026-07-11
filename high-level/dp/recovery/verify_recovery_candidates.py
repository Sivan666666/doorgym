#!/usr/bin/env python3
"""Run proposed recovery candidates in Isaac Gym and retain verified branches.

The agent-produced manifest is only a search proposal. This runner flattens it,
executes candidates in parallel Isaac Gym batches, and aggregates simulator
results. The simulator writes a raw episode only when both local recovery and
final task success are true.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
FLOAT_SCRIPT = REPO_ROOT / "high-level" / "float_ik" / "isaacgym_float_ik_a2w_basearn_push_door_parallel.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulator-verify Door ACT recovery candidate branches.")
    parser.add_argument("--candidate_manifest", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--verified_raw_root", required=True)
    parser.add_argument("--door_cfg", default="high-level/data/cfg/b1z1_opendoor.yaml")
    parser.add_argument("--door_name", default="wc4")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_candidates", type=int, default=0, help="0 runs all candidates.")
    parser.add_argument("--candidate_ids", default="", help="Optional comma-separated candidate ids to verify.")
    parser.add_argument("--seed", type=int, default=615455575)
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--pass_open_angle_deg", type=float, default=80.0)
    parser.add_argument("--contact_min_frames", type=int, default=5)
    parser.add_argument("--camera_depth_clip_lower", type=float, default=0.2)
    parser.add_argument("--camera_depth_clip_far", type=float, default=1.5)
    parser.add_argument("--resume", action="store_true", help="Reuse completed batch result JSON files.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)


def flatten_candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for bundle in manifest.get("bundles", []):
        failure_npz = Path(bundle["failure_rollout_npz"]).expanduser().resolve()
        metadata_path = Path(bundle.get("metadata_path") or failure_npz.with_name("metadata.json")).resolve()
        diagnosis = dict(bundle.get("diagnosis") or {})
        for branch in bundle.get("branches", []):
            for candidate in branch.get("candidates", []):
                entries.append(
                    {
                        "candidate_id": int(candidate["candidate_id"]),
                        "bundle_id": int(bundle.get("bundle_id", -1)),
                        "failure_rollout_npz": str(failure_npz),
                        "metadata_path": str(metadata_path),
                        "diagnosis": diagnosis,
                        "branch_step": int(candidate["branch_step"]),
                        "parameters": dict(candidate["parameters"]),
                    }
                )
    return entries


def validate_snapshots(entries: list[dict[str, Any]]) -> None:
    required = (
        "replay_root_state",
        "replay_dof_pos",
        "replay_dof_vel",
        "replay_door_root_state",
        "replay_door_dof_pos",
        "replay_door_dof_vel",
    )
    checked: set[str] = set()
    for entry in entries:
        path = str(entry["failure_rollout_npz"])
        if path in checked:
            continue
        checked.add(path)
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Failure rollout cannot branch because {path} is missing snapshots: {missing}")


def command_for_batch(args: argparse.Namespace, batch_manifest: Path, result_json: Path, batch_index: int) -> list[str]:
    return [
        sys.executable,
        str(FLOAT_SCRIPT),
        "--recovery_batch_manifest",
        str(batch_manifest),
        "--recovery_result_json",
        str(result_json),
        "--recovery_raw_root",
        str(Path(args.verified_raw_root).expanduser().resolve()),
        "--recovery_contact_min_frames",
        str(args.contact_min_frames),
        "--pass_open_angle_deg",
        str(args.pass_open_angle_deg),
        "--steps",
        "1",
        "--seed",
        str(int(args.seed) + int(batch_index)),
        "--door_cfg",
        str(Path(args.door_cfg).expanduser().resolve()),
        "--door_name",
        str(args.door_name),
        "--rl_device",
        str(args.rl_device),
        "--sim_device",
        str(args.sim_device),
        "--graphics_device_id",
        str(args.graphics_device_id),
        "--enable_wrist_camera",
        "--enable_front_camera",
        "--camera_depth",
        "--depth_only",
        "--camera_depth_clip_lower",
        str(args.camera_depth_clip_lower),
        "--camera_depth_clip_far",
        str(args.camera_depth_clip_far),
        "--headless",
        "--no_show_camera_images",
        "--no_enable_depth_noise",
        "--no_enable_depth_gaussian_blur",
        "--no_enable_depth_camera_randomization",
        "--no_preview_trajectory_at_spawn",
        "--no_draw_ik_target",
        "--no_draw_camera_axes",
        "--no_show_seg",
        "--ee_pose_frame",
        "robot_base_full",
    ]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    manifest_path = Path(args.candidate_manifest).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    entries = flatten_candidates(manifest)
    if str(args.candidate_ids).strip():
        selected_ids = {int(token.strip()) for token in str(args.candidate_ids).split(",") if token.strip()}
        entries = [entry for entry in entries if int(entry["candidate_id"]) in selected_ids]
    if args.max_candidates > 0:
        entries = entries[: int(args.max_candidates)]
    if not entries:
        raise ValueError(f"No candidates found in {manifest_path}")
    validate_snapshots(entries)

    all_results: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(entries), int(args.batch_size))):
        batch_entries = entries[start : start + int(args.batch_size)]
        batch_manifest = run_root / f"batch_{batch_index:04d}_manifest.json"
        result_json = run_root / f"batch_{batch_index:04d}_result.json"
        stdout_path = run_root / f"batch_{batch_index:04d}.out"
        write_json(
            batch_manifest,
            {
                "schema_version": 1,
                "source_candidate_manifest": str(manifest_path),
                "entries": batch_entries,
            },
        )
        cmd = command_for_batch(args, batch_manifest, result_json, batch_index)
        if args.resume and result_json.is_file():
            print(f"[recovery batch {batch_index}] reuse {result_json}", flush=True)
        elif args.dry_run:
            print(" ".join(cmd), flush=True)
            continue
        else:
            print(
                f"[recovery batch {batch_index}] candidates={len(batch_entries)} "
                f"ids={batch_entries[0]['candidate_id']}..{batch_entries[-1]['candidate_id']}",
                flush=True,
            )
            with stdout_path.open("w", encoding="utf-8") as stdout_file:
                completed = subprocess.run(cmd, cwd=REPO_ROOT, stdout=stdout_file, stderr=subprocess.STDOUT)
            if completed.returncode != 0:
                raise RuntimeError(f"Recovery batch {batch_index} failed; inspect {stdout_path}")
        if result_json.is_file():
            all_results.extend(load_json(result_json).get("results", []))

    if args.dry_run:
        return
    verified = [item for item in all_results if bool(item.get("recovery_success"))]
    result_by_id = {int(item["candidate_id"]): item for item in all_results}
    verified_manifest = copy.deepcopy(manifest)
    for bundle in verified_manifest.get("bundles", []):
        for branch in bundle.get("branches", []):
            for candidate in branch.get("candidates", []):
                result = result_by_id.get(int(candidate["candidate_id"]))
                if result is None:
                    continue
                candidate["status"] = "verified_success" if result.get("recovery_success") else "verified_failed"
                candidate["simulator_verification"] = result
    verified_manifest_path = run_root / "recovery_candidates_verified.json"
    write_json(verified_manifest_path, verified_manifest)
    summary = {
        "schema_version": 1,
        "source_candidate_manifest": str(manifest_path),
        "verification_rule": (
            "local_recovery_success AND final_task_success AND "
            "NOT unsafe_or_collision AND NOT premature_unlock"
        ),
        "candidate_count": len(entries),
        "evaluated_count": len(all_results),
        "local_success_count": sum(bool(x.get("local_recovery_success")) for x in all_results),
        "final_success_count": sum(bool(x.get("final_task_success")) for x in all_results),
        "verified_recovery_count": len(verified),
        "verified_raw_root": str(Path(args.verified_raw_root).expanduser().resolve()),
        "verified_candidate_manifest": str(verified_manifest_path),
        "results": all_results,
    }
    summary_path = run_root / "verification_summary.json"
    write_json(summary_path, summary)
    print(
        f"Simulator verification complete: verified={len(verified)}/{len(all_results)} -> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
