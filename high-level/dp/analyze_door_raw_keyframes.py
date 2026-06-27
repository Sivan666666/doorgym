#!/usr/bin/env python
"""Inspect dynamic keyframes extracted from Door DP raw .npz episodes."""

import argparse
from pathlib import Path

import numpy as np

try:
    from .door_dp_common import (
        DEFAULT_A2W_PHASE_NAMES,
        extract_door_keyframes_from_phase_ids,
        extract_motion_keyframes_from_raw_arrays,
    )
except ImportError:
    from door_dp_common import (
        DEFAULT_A2W_PHASE_NAMES,
        extract_door_keyframes_from_phase_ids,
        extract_motion_keyframes_from_raw_arrays,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", required=True, help="Directory containing episode_*.npz raw files.")
    parser.add_argument("--episode", type=int, default=None, help="Only inspect one episode index.")
    parser.add_argument("--max_episodes", type=int, default=5, help="How many episodes to print when --episode is omitted.")
    parser.add_argument("--summary_all", action="store_true", help="Also summarize all episodes in raw_root.")
    return parser.parse_args()


def load_episode_paths(raw_root, episode=None):
    raw_root = Path(raw_root).expanduser()
    if episode is not None:
        path = raw_root / f"episode_{episode:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]
    return sorted(raw_root.glob("episode_*.npz"))


def scalar_phase_names(data):
    if "phase_names" in data.files:
        return [str(x) for x in data["phase_names"].tolist()]
    return list(DEFAULT_A2W_PHASE_NAMES)


def inspect_one(path):
    data = np.load(path, allow_pickle=True)
    phase_ids = data["subtask_index"].reshape(-1) if "subtask_index" in data.files else None
    phase_names = scalar_phase_names(data)
    manual = ([], [], [])
    if phase_ids is not None:
        manual = extract_door_keyframes_from_phase_ids(phase_ids, phase_names)
    dynamic = extract_motion_keyframes_from_raw_arrays(data, phase_names=phase_names)
    return manual, dynamic


def print_episode(path):
    manual, dynamic = inspect_one(path)
    manual_indices, manual_names, _ = manual
    dyn_indices, dyn_names, dyn_rules = dynamic
    print(f"\n{path.name}")
    print(f"  manual keyframes ({len(manual_indices)}):")
    print("   ", list(zip(manual_indices, manual_names)))
    print(f"  dynamic+manual keyframes ({len(dyn_indices)}):")
    for idx, name, rule in zip(dyn_indices, dyn_names, dyn_rules):
        print(f"    {idx:4d}  {name:<55s}  {rule}")
    return len(dyn_indices)


def main():
    args = parse_args()
    paths = load_episode_paths(args.raw_root, args.episode)
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz found under {args.raw_root}")

    printed = paths if args.episode is not None else paths[: max(1, int(args.max_episodes))]
    for path in printed:
        print_episode(path)

    if args.summary_all or args.episode is None:
        if args.summary_all:
            paths = load_episode_paths(args.raw_root, episode=None)
        counts = []
        for path in paths:
            _, dynamic = inspect_one(path)
            counts.append(len(dynamic[0]))
        counts = np.asarray(counts, dtype=np.int64)
        print("\nSummary")
        print(f"  episodes: {len(paths)}")
        print(
            "  keyframes per episode: "
            f"min={int(counts.min())} max={int(counts.max())} "
            f"mean={float(counts.mean()):.2f} median={float(np.median(counts)):.1f}"
        )


if __name__ == "__main__":
    main()
