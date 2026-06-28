#!/usr/bin/env python3
"""Update only keyframe loss weights in an existing LeRobot door dataset.

This keeps observations, actions, videos, and keyframe indices unchanged. It
rewrites the scalar ``loss.action_weight`` parquet column and its dataset- and
episode-level statistics from the keyframe indices stored in raw episodes.
"""

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--weight", type=float, default=8.0)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not retain one .before_keyframe_weight_update backup per modified file.",
    )
    return parser.parse_args()


def action_loss_weights(num_frames, keyframe_indices, weight, radius):
    values = np.ones(int(num_frames), dtype=np.float32)
    for index in np.asarray(keyframe_indices, dtype=np.int64).reshape(-1):
        lo = max(0, int(index) - int(radius))
        hi = min(int(num_frames), int(index) + int(radius) + 1)
        values[lo:hi] = float(weight)
    return values


def load_episode_weights(raw_root, weight, radius):
    files = sorted(raw_root.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found under {raw_root}")
    episode_weights = {}
    # The converter assigns LeRobot episode_index by sorted raw-file order,
    # independently of any numeric gaps in raw filenames.
    for episode_index, path in enumerate(files):
        with np.load(path, allow_pickle=True) as data:
            if "keyframe_indices" not in data.files:
                raise KeyError(f"{path} does not contain keyframe_indices")
            num_frames = int(data["state"].shape[0])
            episode_weights[episode_index] = action_loss_weights(
                num_frames,
                data["keyframe_indices"],
                weight,
                radius,
            )
    return episode_weights


def weight_stats(values):
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


def backup_file(path, enabled):
    if not enabled:
        return
    backup = path.with_name(path.name + ".before_keyframe_weight_update")
    if not backup.exists():
        shutil.copy2(path, backup)


def write_parquet_atomic(path, table, backup):
    backup_file(path, backup)
    tmp = path.with_name(path.name + ".keyframe_weight_tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def write_json_atomic(path, payload, backup):
    backup_file(path, backup)
    tmp = path.with_name(path.name + ".keyframe_weight_tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def replace_data_weights(dataset_root, episode_weights, backup):
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
            if not 0 <= frame_index < episode_weights[episode_index].shape[0]:
                raise IndexError(f"Invalid frame_index={frame_index} for episode {episode_index}")
            values[row] = episode_weights[episode_index][frame_index]
        column_index = table.column_names.index(WEIGHT_FEATURE)
        field_type = table.schema.field(WEIGHT_FEATURE).type
        table = table.set_column(column_index, WEIGHT_FEATURE, pa.array(values, type=field_type))
        write_parquet_atomic(path, table, backup)
        total_rows += table.num_rows
    return total_rows


def replace_episode_stats(dataset_root, episode_weights, backup):
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


def replace_global_stats(dataset_root, episode_weights, backup):
    path = dataset_root / "meta" / "stats.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    all_weights = np.concatenate([episode_weights[index] for index in sorted(episode_weights)])
    stats = weight_stats(all_weights)
    payload[WEIGHT_FEATURE] = {name: [stats[name]] for name in STAT_NAMES}
    write_json_atomic(path, payload, backup)
    return stats


def replace_feature_sidecar(dataset_root, weight, radius, backup):
    path = dataset_root / "door_dp_feature_names.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["keyframe_loss_weight"] = float(weight)
    payload["keyframe_loss_radius"] = int(radius)
    payload["keyframe_loss_enabled"] = True
    payload["action_loss_weight_feature"] = WEIGHT_FEATURE
    write_json_atomic(path, payload, backup)


def main():
    args = parse_args()
    if args.weight <= 0.0:
        raise ValueError("--weight must be > 0")
    if args.radius < 0:
        raise ValueError("--radius must be >= 0")
    raw_root = args.raw_root.resolve()
    dataset_root = args.dataset_root.resolve()
    episode_weights = load_episode_weights(raw_root, args.weight, args.radius)
    backup = not args.no_backup
    total_rows = replace_data_weights(dataset_root, episode_weights, backup)
    replace_episode_stats(dataset_root, episode_weights, backup)
    stats = replace_global_stats(dataset_root, episode_weights, backup)
    replace_feature_sidecar(dataset_root, args.weight, args.radius, backup)
    weighted_frames = sum(int(np.count_nonzero(values == args.weight)) for values in episode_weights.values())
    print(
        f"Updated {len(episode_weights)} episodes / {total_rows} frames: "
        f"weight={args.weight:g}, radius=±{args.radius}, weighted_frames={weighted_frames}, "
        f"mean_weight={stats['mean']:.6f}"
    )
    if backup:
        print("Backups retained with suffix .before_keyframe_weight_update")


if __name__ == "__main__":
    main()
