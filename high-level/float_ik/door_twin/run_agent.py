#!/usr/bin/env python3
"""Run the tool-driven DoorTwin Agent using the current Codex file bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    _FLOAT_IK = Path(__file__).resolve().parents[1]
    if str(_FLOAT_IK) not in sys.path:
        sys.path.insert(0, str(_FLOAT_IK))
    from door_twin.agent_runtime import DoorTwinAgentSession, ManualCodexBackend
    from door_twin.benchmark.schema import (
        OURS_INITIALIZATION_PROTOCOL,
        Candidate,
        load_benchmark_manifest,
        rule_based_candidate,
    )
    from door_twin.experience import ExperienceCatalog
else:
    from .agent_runtime import DoorTwinAgentSession, ManualCodexBackend
    from .benchmark.schema import (
        OURS_INITIALIZATION_PROTOCOL,
        Candidate,
        load_benchmark_manifest,
        rule_based_candidate,
    )
    from .experience import ExperienceCatalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--door", required=True, help="DoorCase id or door_name in the manifest")
    parser.add_argument("--candidate_index", type=int, default=0)
    parser.add_argument(
        "--initial_candidate",
        default="",
        help="Optional frozen candidate.json shared by paired benchmark branches.",
    )
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--manual_exchange_dir", required=True)
    parser.add_argument(
        "--experience_catalog",
        default=str(Path(__file__).with_name("experience") / "catalog.yaml"),
    )
    parser.add_argument("--max_tool_turns", type=int, default=30)
    parser.add_argument("--max_repairs", type=int, default=5)
    parser.add_argument("--benchmark_mode", action="store_true")
    parser.add_argument("--no_visual_feedback", action="store_true")
    parser.add_argument("--stream_output", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_benchmark_manifest(args.manifest, formal=False)
    door = next((item for item in manifest.doors if item.id == args.door or item.door_name == args.door), None)
    if door is None:
        raise KeyError(f"Door {args.door!r} is not in {args.manifest}")
    if args.initial_candidate:
        candidate = Candidate.read(args.initial_candidate, manifest.base_door_cfg)
        if candidate.door_name != door.door_name or candidate.door_id != door.id:
            raise ValueError(
                "Frozen initial candidate does not match the selected door: "
                f"candidate=({candidate.door_id}, {candidate.door_name}) "
                f"door=({door.id}, {door.door_name})"
            )
        if candidate.candidate_index != args.candidate_index:
            raise ValueError(
                f"Frozen candidate index {candidate.candidate_index} != --candidate_index {args.candidate_index}"
            )
    else:
        candidate = rule_based_candidate(manifest, door, args.candidate_index)
    if args.benchmark_mode:
        protocol = str((candidate.metadata.get("initial_generation", {}) or {}).get("protocol", ""))
        if not args.initial_candidate or protocol != OURS_INITIALIZATION_PROTOCOL:
            raise ValueError(
                "Benchmark Log/Full agents must receive the shared Rule-based + retrieval-residual candidate via "
                f"--initial_candidate (expected protocol={OURS_INITIALIZATION_PROTOCOL!r}, got {protocol!r})."
            )
    catalog = ExperienceCatalog.load(args.experience_catalog)
    backend = ManualCodexBackend(args.manual_exchange_dir)
    session = DoorTwinAgentSession(
        run_root=args.run_root,
        manifest=manifest,
        door=door,
        initial_candidate=candidate,
        catalog=catalog,
        backend=backend,
        visual_feedback=not args.no_visual_feedback,
        benchmark_mode=args.benchmark_mode,
        max_tool_turns=args.max_tool_turns,
        max_repairs=args.max_repairs,
        force=args.force,
        stream_output=args.stream_output,
    )
    result = session.run()
    print(f"DoorTwin Agent finished: {result}")


if __name__ == "__main__":
    main()
