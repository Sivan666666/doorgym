#!/usr/bin/env python3
"""Create exact nested prefix subsets of a LeRobot dataset.

The subset parquet files, episode metadata, and normalization statistics are
materialized independently. Video files are shared through read-only symlinks,
which preserves the exact decoded pixels and avoids lossy re-encoding.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import (
    _copy_and_reindex_data,
    _copy_and_reindex_episodes_metadata,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


SIDECAR_NAME = "door_dp_feature_names.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True, type=Path)
    parser.add_argument("--source_repo_id", required=True)
    parser.add_argument("--output_parent", required=True, type=Path)
    parser.add_argument("--output_name_prefix", required=True)
    parser.add_argument("--sizes", default="50,100,150")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_sizes(text: str, total_episodes: int) -> list[int]:
    sizes = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not sizes or sizes[0] <= 0 or sizes[-1] > total_episodes:
        raise ValueError(f"Expected sizes in [1, {total_episodes}], got {sizes}")
    return sizes


def shared_video_metadata(
    source: LeRobotDataset,
    destination_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> dict[int, dict]:
    metadata: dict[int, dict] = {new_idx: {} for new_idx in episode_mapping.values()}
    linked_paths: set[Path] = set()
    if destination_meta.video_path is None:
        return metadata

    for video_key in source.meta.video_keys:
        for old_idx, new_idx in episode_mapping.items():
            src_episode = source.meta.episodes[old_idx]
            chunk_idx = int(src_episode[f"videos/{video_key}/chunk_index"])
            file_idx = int(src_episode[f"videos/{video_key}/file_index"])
            src_video = source.root / source.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            dst_video = destination_meta.root / destination_meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            if dst_video not in linked_paths:
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                dst_video.symlink_to(src_video.resolve())
                linked_paths.add(dst_video)

            item = metadata[new_idx]
            item[f"videos/{video_key}/chunk_index"] = chunk_idx
            item[f"videos/{video_key}/file_index"] = file_idx
            item[f"videos/{video_key}/from_timestamp"] = float(
                src_episode[f"videos/{video_key}/from_timestamp"]
            )
            item[f"videos/{video_key}/to_timestamp"] = float(
                src_episode[f"videos/{video_key}/to_timestamp"]
            )
    return metadata


def write_sidecar(source_root: Path, output_root: Path, size: int, source_repo_id: str) -> None:
    source_path = source_root / SIDECAR_NAME
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing Door sidecar: {source_path}")
    sidecar = json.loads(source_path.read_text(encoding="utf-8"))
    sidecar.update(
        {
            "subset_source_repo_id": source_repo_id,
            "subset_episode_indices": [0, size],
            "subset_episode_range_semantics": "half_open",
            "subset_total_episodes": size,
            "subset_video_storage": "symlink_to_source_exact_pixels",
        }
    )
    (output_root / SIDECAR_NAME).write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def create_subset(
    source: LeRobotDataset,
    output_root: Path,
    output_repo_id: str,
    size: int,
    overwrite: bool,
) -> LeRobotDataset:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)

    destination_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=source.meta.fps,
        features=source.meta.features,
        robot_type=source.meta.robot_type,
        root=output_root,
        use_videos=bool(source.meta.video_keys),
        chunks_size=source.meta.chunks_size,
        data_files_size_in_mb=source.meta.data_files_size_in_mb,
        video_files_size_in_mb=source.meta.video_files_size_in_mb,
    )
    episode_mapping = {episode_idx: episode_idx for episode_idx in range(size)}
    video_metadata = shared_video_metadata(source, destination_meta, episode_mapping)
    data_metadata = _copy_and_reindex_data(source, destination_meta, episode_mapping)
    _copy_and_reindex_episodes_metadata(
        source,
        destination_meta,
        episode_mapping,
        data_metadata,
        video_metadata,
    )
    write_sidecar(source.root, output_root, size, source.repo_id)
    return LeRobotDataset(repo_id=output_repo_id, root=output_root)


def main() -> None:
    args = parse_args()
    source = LeRobotDataset(repo_id=args.source_repo_id, root=args.source_root.resolve())
    sizes = parse_sizes(args.sizes, source.meta.total_episodes)
    args.output_parent.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        output_name = f"{args.output_name_prefix}_{size}"
        output_repo_id = f"local/{output_name}"
        output_root = (args.output_parent / output_name).resolve()
        subset = create_subset(source, output_root, output_repo_id, size, args.overwrite)
        print(
            f"Created {output_repo_id}: episodes={subset.num_episodes} "
            f"frames={subset.num_frames} root={output_root}",
            flush=True,
        )


if __name__ == "__main__":
    main()
