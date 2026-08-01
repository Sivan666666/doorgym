#!/usr/bin/env python3
"""Evaluate the deterministic public-geometry DoorTwin baseline only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from door_twin.benchmark.evaluation import evaluate_asset, stage_metrics  # noqa: E402
from door_twin.benchmark.run_agent_ablation import run_asset_probe, run_seed_set  # noqa: E402
from door_twin.benchmark.schema import load_benchmark_manifest, rule_based_candidate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--doors", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stream_output", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    manifest = load_benchmark_manifest(args.manifest, formal=False)
    selected = set(filter(None, args.doors.split(",")))
    root = Path(args.output_root).expanduser().resolve()
    options = SimpleNamespace(dry_run=False, stream_output=args.stream_output, force=args.force)
    results = []
    for door in manifest.doors:
        if selected and door.id not in selected and door.door_name not in selected:
            continue
        candidate = rule_based_candidate(manifest, door, 0)
        out = root / door.id
        paths = candidate.write(out / "candidate")
        heldout, elapsed = run_seed_set(
            manifest,
            candidate,
            paths,
            manifest.heldout_seeds,
            out / "heldout",
            capture_images=False,
            args=options,
        )
        probe, probe_elapsed = run_asset_probe(manifest, candidate, paths, out / "asset_probe", options)
        result = {
            "schema_version": "door_twin_rule_based_result_v2",
            "door_id": door.id,
            "door_name": door.door_name,
            "method": "rule_based",
            "candidate_hash": candidate.fingerprint,
            "heldout_seeds": manifest.heldout_seeds,
            "heldout": stage_metrics(heldout),
            "asset": evaluate_asset(candidate, door, probe, motion_fallback_summary=heldout),
            "elapsed_s": elapsed + probe_elapsed,
            "vlm_calls": 0,
            "development_seeds": [],
        }
        result["generation_success"] = bool(
            result["asset"]["structure_success"]
            and result["heldout"]["success_count"] >= manifest.generation_success_min
        )
        write_json(out / "result.json", result)
        results.append(result)
        print(
            f"{door.id}: {result['heldout']['success_count']}/{result['heldout']['trials']} "
            f"structure={result['asset']['structure_success']}",
            flush=True,
        )
    write_json(root / "results.json", {"results": results})


if __name__ == "__main__":
    main()
