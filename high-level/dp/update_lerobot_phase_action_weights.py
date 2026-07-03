#!/usr/bin/env python3
"""Update ``loss.action_weight`` by Door phase in an existing LeRobot dataset.

This is useful for ACT experiments where an entire scripted phase should get a
larger per-timestep action loss weight, instead of sparse keyframe windows.

The script reads raw ``episode_*.npz`` files for ``subtask_index`` and
``phase_names``, then rewrites the scalar per-frame ``loss.action_weight`` column
in the already-converted LeRobot dataset. Observations, actions, and videos are
left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


WEIGHT_FEATURE = "loss.action_weight"
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--weight", type=float, default=3.0)
    parser.add_argument(
        "--phase",
        dest="phase_list",
        action="append",
        default=[],
        help="Phase name to up-weight. Can be passed multiple times.",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="grasp,close_gripper,rotate_handle",
        help="Comma-separated phase names to up-weight.",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not retain one .before_phase_weight_update backup per modified file.",
    )
    return parser.parse_args()


def phase_action_loss_weights(path: Path, phases: list[str], weight: float) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "subtask_index" not in data.files:
            raise KeyError(f"{path} does not contain subtask_index")
        if "phase_names" not in data.files:
            raise KeyError(f"{path} does not contain phase_names")
        phase_names = [str(x) for x in data["phase_names"].reshape(-1)]
        phase_to_id = {name: idx for idx, name in enumerate(phase_names)}
        missing = [name for name in phases if name not in phase_to_id]
        if missing:
            raise KeyError(f"{path} missing phases {missing}; available={phase_names}")
        target_ids = {phase_to_id[name] for name in phases}
        phase_ids = data["subtask_index"].reshape(-1).astype(np.int64)
    values = np.ones(phase_ids.shape[0], dtype=np.float32)
    values[np.isin(phase_ids, list(target_ids))] = float(weight)
    return values


def load_episode_weights(raw_root: Path, phases: list[str], weight: float) -> dict[int, np.ndarray]:
    files = sorted(raw_root.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found under {raw_root}")
    episode_weights = {}
    for episode_index, path in enumerate(files):
        episode_weights[episode_index] = phase_action_loss_weights(path, phases, weight)
    return episode_weights


def weight_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "count": int(values.size),
        "q01": float(np.quantile(values, 0.01)),
        "q10": float(np.quantile(values, 0.10)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
    }


def backup_file(path: Path, enabled: bool) -> None:
    if not enabled:
        return
    backup = path.with_name(path.name + ".before_phase_weight_update")
    if not backup.exists():
        shutil.copy2(path, backup)


def write_parquet_atomic(path: Path, table: pa.Table, backup: bool) -> None:
    backup_file(path, backup)
    tmp = path.with_name(path.name + ".phase_weight_tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: dict, backup: bool) -> None:
    backup_file(path, backup)
    tmp = path.with_name(path.name + ".phase_weight_tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def replace_data_weights(dataset_root: Path, episode_weights: dict[int, np.ndarray], backup: bool) -> int:
    paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files found under {dataset_root / 'data'}")
    total_rows = 0
    for path in paths:
        table = pq.read_table(path)
        if WEIGHT_FEATURE not in table.column_names:
            raise KeyError(f"{path} does not contain {WEIGHT_FEATURE}")
        episode_indices = table["episode_index"].to_numpy(zero_copy_only=False)
        frame_indices = table["frame_index"].to_numpy(zero_copy_only=False)
        values = np.empty(table.num_rows, dtype=np.float32)
        for row, (episode_index, frame_index) in enumerate(zip(episode_indices, frame_indices)):
            episode_index = int(episode_index)
            frame_index = int(frame_index)
            if episode_index not in episode_weights:
                raise KeyError(f"Dataset references episode {episode_index}, absent from raw_root")
            weights = episode_weights[episode_index]
            if not 0 <= frame_index < weights.shape[0]:
                raise IndexError(f"Invalid frame_index={frame_index} for episode {episode_index}")
            values[row] = weights[frame_index]
        column_index = table.column_names.index(WEIGHT_FEATURE)
        field_type = table.schema.field(WEIGHT_FEATURE).type
        table = table.set_column(column_index, WEIGHT_FEATURE, pa.array(values, type=field_type))
        write_parquet_atomic(path, table, backup)
        total_rows += table.num_rows
    return total_rows


def replace_episode_stats(dataset_root: Path, episode_weights: dict[int, np.ndarray], backup: bool) -> None:
    paths = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet found under {dataset_root / 'meta/episodes'}")
    seen = set()
    for path in paths:
        table = pq.read_table(path)
        episode_indices = table["episode_index"].to_numpy(zero_copy_only=False)
        per_episode_stats = []
        for episode_index in episode_indices:
            episode_index = int(episode_index)
            if episode_index not in episode_weights:
                raise KeyError(f"Episode metadata references episode {episode_index}, absent from raw_root")
            seen.add(episode_index)
            per_episode_stats.append(weight_stats(episode_weights[episode_index]))
        for stat_name in STAT_NAMES:
            column_name = f"stats/{WEIGHT_FEATURE}/{stat_name}"
            if column_name not in table.column_names:
                raise KeyError(f"{path} does not contain {column_name}")
            values = [[stats[stat_name]] for stats in per_episode_stats]
            column_index = table.column_names.index(column_name)
            field_type = table.schema.field(column_name).type
            table = table.set_column(column_index, column_name, pa.array(values, type=field_type))
        write_parquet_atomic(path, table, backup)
    missing = set(episode_weights) - seen
    if missing:
        raise ValueError(f"Raw episodes absent from dataset episode metadata: {sorted(missing)}")


def replace_global_stats(dataset_root: Path, episode_weights: dict[int, np.ndarray], backup: bool) -> dict:
    path = dataset_root / "meta" / "stats.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    all_weights = np.concatenate([episode_weights[index] for index in sorted(episode_weights)])
    stats = weight_stats(all_weights)
    payload[WEIGHT_FEATURE] = {name: [stats[name]] for name in STAT_NAMES}
    write_json_atomic(path, payload, backup)
    return stats


def replace_feature_sidecar(dataset_root: Path, phases: list[str], weight: float, backup: bool) -> None:
    path = dataset_root / "door_dp_feature_names.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phase_action_loss_weight_enabled"] = True
    payload["phase_action_loss_weight"] = float(weight)
    payload["phase_action_loss_weight_phases"] = list(phases)
    payload["action_loss_weight_feature"] = WEIGHT_FEATURE
    write_json_atomic(path, payload, backup)


def main() -> None:
    args = parse_args()
    if args.weight <= 1.0:
        raise ValueError("--weight should be > 1 for an up-weighting experiment")
    phases = []
    for value in str(args.phases or "").split(","):
        value = value.strip()
        if value:
            phases.append(value)
    phases.extend([str(x).strip() for x in args.phase_list if str(x).strip()])
    phases = list(dict.fromkeys(phases))
    if not phases:
        raise ValueError("No phases specified; use --phases or repeated --phase.")
    raw_root = args.raw_root.resolve()
    dataset_root = args.dataset_root.resolve()
    episode_weights = load_episode_weights(raw_root, phases, args.weight)
    backup = not args.no_backup
    total_rows = replace_data_weights(dataset_root, episode_weights, backup)
    replace_episode_stats(dataset_root, episode_weights, backup)
    stats = replace_global_stats(dataset_root, episode_weights, backup)
    replace_feature_sidecar(dataset_root, phases, args.weight, backup)
    weighted_frames = sum(int(np.count_nonzero(values > 1.0)) for values in episode_weights.values())
    print(
        f"Updated {len(episode_weights)} episodes / {total_rows} frames: "
        f"phases={phases}, weight={args.weight:g}, weighted_frames={weighted_frames}, "
        f"mean_weight={stats['mean']:.6f}"
    )
    if backup:
        print("Backups retained with suffix .before_phase_weight_update")


if __name__ == "__main__":
    main()
