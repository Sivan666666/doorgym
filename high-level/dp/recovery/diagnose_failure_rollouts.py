#!/usr/bin/env python3
"""Diagnose exported ACT failure rollouts.

The evaluator can export one bundle per failed env:

    failure_rollout.npz
    metadata.json
    diagnosis.json

This script reruns the deterministic rule-based diagnosis over one bundle or a
directory tree of bundles.  It intentionally treats the diagnosis as search
guidance, not as training ground truth; simulator verification should decide
which recovery branches are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


DP_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = DP_ROOT / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from eval_door_policy_success import diagnose_failure_arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose exported Door ACT failure rollout bundles.")
    parser.add_argument(
        "--failure_rollout",
        type=str,
        default="",
        help="Path to one failure_rollout.npz or its containing directory.",
    )
    parser.add_argument(
        "--failure_root",
        type=str,
        default="",
        help="Directory containing many failure_rollout.npz bundles.",
    )
    parser.add_argument("--success_threshold_deg", type=float, default=80.0)
    parser.add_argument("--write", action="store_true", help="Write diagnosis.json beside each npz.")
    parser.add_argument("--summary_json", type=str, default="", help="Optional summary JSON output path.")
    return parser.parse_args()


def npz_paths_from_args(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.failure_rollout:
        p = Path(args.failure_rollout).expanduser()
        if p.is_dir():
            p = p / "failure_rollout.npz"
        paths.append(p)
    if args.failure_root:
        root = Path(args.failure_root).expanduser()
        paths.extend(sorted(root.glob("**/failure_rollout.npz")))
    unique: list[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            unique.append(rp)
            seen.add(rp)
    return unique


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    args = parse_args()
    npz_paths = npz_paths_from_args(args)
    if not npz_paths:
        raise ValueError("Provide --failure_rollout or --failure_root.")

    results = []
    for npz_path in npz_paths:
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        arrays = load_npz_arrays(npz_path)
        diagnosis = diagnose_failure_arrays(arrays, float(args.success_threshold_deg))
        item = {
            "npz_path": str(npz_path),
            "diagnosis_path": str(npz_path.with_name("diagnosis.json")),
            "diagnosis": diagnosis,
        }
        results.append(item)
        print(
            f"{npz_path}: type={diagnosis['failure_type']} "
            f"t_dev={diagnosis['t_dev']} t_fail={diagnosis['t_fail']} "
            f"recoverability={diagnosis['recoverability']}",
            flush=True,
        )
        if args.write:
            with npz_path.with_name("diagnosis.json").open("w", encoding="utf-8") as f:
                json.dump(diagnosis, f, indent=2, sort_keys=True)

    if args.summary_json:
        out = Path(args.summary_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "count": len(results),
                    "results": results,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"Summary written to {out}", flush=True)


if __name__ == "__main__":
    main()
