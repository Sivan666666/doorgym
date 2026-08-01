#!/usr/bin/env python3
"""Convert A2W front/wrist depth into separate robot-base DP3 point clouds."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_door_raw_to_dp3 import (  # noqa: E402
    save_depth_png,
    scalar,
    write_ply,
    write_pointcloud_html,
)
from pointcloud import (  # noqa: E402
    FrontDepthPointCloudConfig,
    front_depth_to_point_cloud_batch,
    intrinsics_dict,
)


VIEWS = ("front", "wrist")


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=workspace / "high-level/data/door_dp_raw/a2w_wc4_robotbasefull_contactcheck_200",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            workspace.parent
            / "3D-Diffusion-Policy/3D-Diffusion-Policy/data/a2w_wc4_dual_xyz_200.zarr"
        ),
    )
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--candidate_rows", type=int, default=64)
    parser.add_argument("--candidate_cols", type=int, default=64)
    parser.add_argument("--workspace_min", type=float, nargs=3, default=(0.20, -1.00, 0.00))
    parser.add_argument("--workspace_max", type=float, nargs=3, default=(2.00, 1.00, 1.80))
    parser.add_argument("--point_frame", choices=("robot_base",), default="robot_base")
    parser.add_argument("--empty_depth_policy", choices=("error", "previous"), default="error")
    parser.add_argument("--batch_frames", type=int, default=16)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--debug_dir", type=Path, default=None)
    parser.add_argument("--debug_per_door", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def episode_metadata(data: np.lib.npyio.NpzFile) -> dict:
    intrinsics = scalar(data, "camera_intrinsics")
    noise = scalar(data, "depth_noise_config", {}) or {}
    return {
        "camera_intrinsics": {view: intrinsics_dict(intrinsics, camera=view) for view in VIEWS},
        "near_clip_m": float(noise.get("near_clip_m", 0.2)),
        "far_clip_m": float(noise.get("far_clip_m", 1.5)),
        "state_feature_names": list(scalar(data, "state_feature_names", [])),
        "action_names": list(scalar(data, "action_names", [])),
        "action_frame": str(scalar(data, "action_frame", "robot_base_full")),
        "state_format": str(scalar(data, "state_format", "")),
        "action_format": str(scalar(data, "action_format", "")),
        "ikpush_state_version": str(scalar(data, "ikpush_state_version", "legacy")),
        "door_dp_mode": str(scalar(data, "door_dp_mode", "ikpush")),
        "fps": float(scalar(data, "fps", 25.0)),
    }


def convert_view(
    *,
    depth: np.ndarray,
    poses: np.ndarray,
    intrinsics: dict,
    cfg: FrontDepthPointCloudConfig,
    target,
    target_offset: int,
    batch_frames: int,
    empty_depth_policy: str,
    episode_index: int,
    view: str,
) -> list[dict]:
    """Convert one episode/view and write it directly to its Zarr array."""
    fallbacks: list[dict] = []
    last_valid_points = None
    last_valid_frame = None
    for start in range(0, len(depth), max(1, int(batch_frames))):
        end = min(len(depth), start + max(1, int(batch_frames)))
        try:
            points = front_depth_to_point_cloud_batch(depth[start:end], poses[start:end], intrinsics, cfg)
        except ValueError as exc:
            if empty_depth_policy != "previous" or "No valid front-depth points" not in str(exc):
                raise ValueError(
                    f"episode={episode_index} view={view} frames=[{start},{end}) failed: {exc}"
                ) from exc
            frame_points = []
            for frame_index in range(start, end):
                try:
                    current = front_depth_to_point_cloud_batch(
                        depth[frame_index], poses[frame_index], intrinsics, cfg
                    )
                    last_valid_points = current
                    last_valid_frame = frame_index
                except ValueError as frame_exc:
                    if (
                        "No valid front-depth points" not in str(frame_exc)
                        or last_valid_points is None
                    ):
                        raise ValueError(
                            f"episode={episode_index} view={view} frame={frame_index} has no "
                            "valid points and no previous cloud."
                        ) from frame_exc
                    current = last_valid_points.copy()
                    fallbacks.append(
                        {
                            "episode": episode_index,
                            "view": view,
                            "frame": frame_index,
                            "source_frame": last_valid_frame,
                        }
                    )
                frame_points.append(current)
            points = np.stack(frame_points, axis=0)
        target[target_offset + start : target_offset + end] = points
        last_valid_points = points[-1]
        if not any(start <= item["frame"] < end for item in fallbacks):
            last_valid_frame = end - 1
    return fallbacks


def main() -> None:
    args = parse_args()
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise RuntimeError("Conversion requires zarr and numcodecs; use the dp3 environment.") from exc

    raw_root = args.raw_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    files = sorted(raw_root.glob("episode_*.npz"))
    if args.max_episodes is not None:
        files = files[: max(0, int(args.max_episodes))]
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found under {raw_root}.")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    required = ["state", "action", "camera_intrinsics"]
    for view in VIEWS:
        required.extend([f"{view}_masked_depth", f"{view}_camera_pose_base"])

    lengths: list[int] = []
    meta = None
    state_dim = action_dim = 0
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in required if key not in data.files]
            if missing:
                raise ValueError(f"{path.name} is missing required fields: {missing}.")
            length = int(data["action"].shape[0])
            current = episode_metadata(data)
            if meta is None:
                meta = current
                state_dim = int(data["state"].shape[-1])
                action_dim = int(data["action"].shape[-1])
            elif current != meta:
                raise ValueError(f"Inconsistent camera/task metadata in {path.name}.")
            if data["state"].shape != (length, state_dim):
                raise ValueError(f"Unexpected state shape in {path.name}: {data['state'].shape}.")
            if data["action"].shape != (length, action_dim):
                raise ValueError(f"Unexpected action shape in {path.name}: {data['action'].shape}.")
            # Do not access the compressed depth members here merely to read
            # their shape: np.load would decompress both 480x640 movies once
            # during validation and again during conversion. Shapes are
            # checked immediately after each episode is materialized below.
            lengths.append(length)
    assert meta is not None

    cfg = FrontDepthPointCloudConfig(
        num_points=int(args.num_points),
        candidate_rows=int(args.candidate_rows),
        candidate_cols=int(args.candidate_cols),
        workspace_min=tuple(float(x) for x in args.workspace_min),
        workspace_max=tuple(float(x) for x in args.workspace_max),
        near_clip_m=float(meta["near_clip_m"]),
        far_clip_m=float(meta["far_clip_m"]),
        point_frame=args.point_frame,
    )
    cfg.validate()

    total_frames = int(sum(lengths))
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output), mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    clouds = {
        view: data_group.create_dataset(
            f"{view}_point_cloud",
            shape=(total_frames, cfg.num_points, 3),
            chunks=(min(64, total_frames), cfg.num_points, 3),
            dtype="f4",
            compressor=compressor,
        )
        for view in VIEWS
    }
    state_array = data_group.create_dataset(
        "state",
        shape=(total_frames, state_dim),
        chunks=(min(512, total_frames), state_dim),
        dtype="f4",
        compressor=compressor,
    )
    action_array = data_group.create_dataset(
        "action",
        shape=(total_frames, action_dim),
        chunks=(min(512, total_frames), action_dim),
        dtype="f4",
        compressor=compressor,
    )
    episode_ends = np.cumsum(lengths, dtype=np.int64)
    meta_group.create_dataset("episode_ends", data=episode_ends, dtype="i8", compressor=compressor)

    conversion = {
        "format": "a2w_dual_depth_dp3_v1",
        "source_raw_root": str(raw_root),
        "episode_count": len(files),
        "total_frames": total_frames,
        "views": list(VIEWS),
        "pointcloud_keys": {view: f"{view}_point_cloud" for view in VIEWS},
        "depth_keys": {view: f"{view}_masked_depth" for view in VIEWS},
        "camera_pose_keys": {view: f"{view}_camera_pose_base" for view in VIEWS},
        "camera_frame": "+X forward, +Y left, +Z up",
        "pixel_center": True,
        "pointcloud": cfg.to_dict(),
        "empty_depth_policy": str(args.empty_depth_policy),
        **meta,
    }
    root.attrs["conversion_json"] = json.dumps(conversion, ensure_ascii=False, sort_keys=True)

    debug_dir = args.debug_dir or output.parent / f"{output.stem}_debug"
    debug_dir = debug_dir.expanduser().resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_counts: dict[str, int] = {}
    debug_records: list[dict] = []
    fallbacks: list[dict] = []
    offset = 0
    started = time.time()
    for episode_index, (path, length) in enumerate(zip(files, lengths)):
        with np.load(path, allow_pickle=True) as episode:
            state_array[offset : offset + length] = np.asarray(episode["state"], dtype=np.float32)
            action_array[offset : offset + length] = np.asarray(episode["action"], dtype=np.float32)
            door_name = str(scalar(episode, "door_asset_name", "unknown"))
            remaining = max(0, int(args.debug_per_door) - debug_counts.get(door_name, 0))
            debug_frames = (
                np.rint(np.linspace(0, length - 1, remaining)).astype(int).tolist()
                if remaining
                else []
            )
            for view in VIEWS:
                depth = np.asarray(episode[f"{view}_masked_depth"], dtype=np.uint8)
                poses = np.asarray(episode[f"{view}_camera_pose_base"], dtype=np.float32)
                expected_hw = (
                    int(meta["camera_intrinsics"][view]["height"]),
                    int(meta["camera_intrinsics"][view]["width"]),
                )
                if depth.shape != (length, *expected_hw):
                    raise ValueError(
                        f"Unexpected {view} depth shape in {path.name}: {depth.shape}.")
                if poses.shape != (length, 7):
                    raise ValueError(
                        f"Unexpected {view} pose shape in {path.name}: {poses.shape}.")
                fallbacks.extend(
                    convert_view(
                        depth=depth,
                        poses=poses,
                        intrinsics=meta["camera_intrinsics"][view],
                        cfg=cfg,
                        target=clouds[view],
                        target_offset=offset,
                        batch_frames=args.batch_frames,
                        empty_depth_policy=str(args.empty_depth_policy),
                        episode_index=episode_index,
                        view=view,
                    )
                )
                for frame in debug_frames:
                    points = np.asarray(clouds[view][offset + frame], dtype=np.float32)
                    stem = f"door_{door_name}_ep{episode_index:06d}_frame{frame:04d}_{view}"
                    write_ply(debug_dir / f"{stem}.ply", points)
                    write_pointcloud_html(debug_dir / f"{stem}.html", points)
                    save_depth_png(debug_dir / f"{stem}_depth.png", depth[frame])
                    debug_records.append(
                        {
                            "door": door_name,
                            "episode": episode_index,
                            "frame": frame,
                            "view": view,
                            "point_min": points.min(axis=0).tolist(),
                            "point_max": points.max(axis=0).tolist(),
                        }
                    )
            debug_counts[door_name] = debug_counts.get(door_name, 0) + len(debug_frames)
            offset += length
        print(
            f"[{episode_index + 1:03d}/{len(files):03d}] {path.name}: {length} frames; "
            f"written={offset}/{total_frames}; elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    conversion["empty_depth_fallback_count"] = len(fallbacks)
    root.attrs["conversion_json"] = json.dumps(conversion, ensure_ascii=False, sort_keys=True)
    (debug_dir / "manifest.json").write_text(
        json.dumps(
            {"conversion": conversion, "samples": debug_records, "empty_depth_fallbacks": fallbacks},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done: {output}")
    for view in VIEWS:
        print(f"{view}_point_cloud={clouds[view].shape}")
    print(f"state={state_array.shape} action={action_array.shape}")
    print(f"episode_ends[-1]={int(episode_ends[-1])}; debug={debug_dir}")
    print(f"empty_depth_fallbacks={len(fallbacks)} policy={args.empty_depth_policy}")


if __name__ == "__main__":
    main()
