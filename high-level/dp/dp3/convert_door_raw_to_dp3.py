#!/usr/bin/env python3
"""Convert A2W raw door episodes into the Zarr layout consumed by DP3."""

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

from pointcloud import (  # noqa: E402
    FrontDepthPointCloudConfig,
    front_depth_to_point_cloud_batch,
    intrinsics_dict,
)


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[3]
    default_raw = workspace / "high-level/data/door_dp_raw/a2w_5door_robotbasefull_contactcheck_250"
    default_output = (
        workspace.parent / "3D-Diffusion-Policy/3D-Diffusion-Policy/data/a2w_5door_front_xyz_250.zarr"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", type=Path, default=default_raw)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--candidate_rows", type=int, default=64)
    parser.add_argument("--candidate_cols", type=int, default=64)
    parser.add_argument("--workspace_min", type=float, nargs=3, default=(0.20, -1.00, 0.00))
    parser.add_argument("--workspace_max", type=float, nargs=3, default=(2.00, 1.00, 1.80))
    parser.add_argument("--point_frame", choices=("robot_base",), default="robot_base")
    parser.add_argument(
        "--empty_depth_policy",
        choices=("error", "previous"),
        default="error",
        help="How to handle an all-invalid frame. 'previous' reuses the last valid cloud and records every fallback.",
    )
    parser.add_argument("--batch_frames", type=int, default=16)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--debug_dir", type=Path, default=None)
    parser.add_argument("--debug_per_door", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if value.ndim == 0 else value


def write_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(stream, points, fmt="%.7f %.7f %.7f")


def write_pointcloud_html(path: Path, points: np.ndarray) -> None:
    """Write a lightweight interactive point-cloud viewer with embedded XYZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    xyz_json = json.dumps(points.round(6).tolist(), separators=(",", ":"))
    path.write_text(
        """<!doctype html><meta charset=\"utf-8\"><title>A2W DP3 point cloud</title>
<style>html,body,#plot{width:100%;height:100%;margin:0;background:#111}</style>
<div id=\"plot\"></div><script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<script>
const p=__POINTS__; const x=p.map(v=>v[0]), y=p.map(v=>v[1]), z=p.map(v=>v[2]);
Plotly.newPlot('plot',[{type:'scatter3d',mode:'markers',x,y,z,
 marker:{size:2,color:z,colorscale:'Viridis',opacity:0.9}}],{
 paper_bgcolor:'#111',plot_bgcolor:'#111',font:{color:'#ddd'},
 scene:{aspectmode:'data',xaxis:{title:'base X (m)'},yaxis:{title:'base Y (m)'},zaxis:{title:'base Z (m)'}},
 margin:{l:0,r:0,b:0,t:25},title:'Front-depth XYZ in robot-base frame'});
</script>""".replace("__POINTS__", xyz_json),
        encoding="utf-8",
    )


def save_depth_png(path: Path, depth: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(depth, dtype=np.uint8), mode="L").save(path)


def episode_metadata(data: np.lib.npyio.NpzFile) -> dict:
    noise = scalar(data, "depth_noise_config", {}) or {}
    return {
        "intrinsics": intrinsics_dict(scalar(data, "camera_intrinsics")),
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


def main() -> None:
    args = parse_args()
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise RuntimeError("Conversion requires zarr and numcodecs; run it in b1z1_lerobot or dp3.") from exc

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

    with np.load(files[0], allow_pickle=True) as first:
        meta = episode_metadata(first)
        first_n = int(first["action"].shape[0])
        state_dim = int(first["state"].shape[-1])
        action_dim = int(first["action"].shape[-1])
    lengths = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            required = ("state", "action", "front_masked_depth", "front_camera_pose_base", "camera_intrinsics")
            missing = [key for key in required if key not in data.files]
            if missing:
                raise ValueError(f"{path.name} is missing required fields: {missing}.")
            lengths.append(int(data["action"].shape[0]))
            current = episode_metadata(data)
            for key in ("intrinsics", "near_clip_m", "far_clip_m", "state_feature_names", "action_names", "action_frame", "ikpush_state_version", "door_dp_mode"):
                if current[key] != meta[key]:
                    raise ValueError(f"Inconsistent {key} in {path.name}: {current[key]!r} != {meta[key]!r}.")
            if data["state"].shape != (lengths[-1], state_dim):
                raise ValueError(f"Unexpected state shape in {path.name}: {data['state'].shape}.")
            if data["action"].shape != (lengths[-1], action_dim):
                raise ValueError(f"Unexpected action shape in {path.name}: {data['action'].shape}.")

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
    pc_array = data_group.create_dataset(
        "point_cloud",
        shape=(total_frames, cfg.num_points, 3),
        chunks=(min(64, total_frames), cfg.num_points, 3),
        dtype="f4",
        compressor=compressor,
    )
    state_array = data_group.create_dataset(
        "state", shape=(total_frames, state_dim), chunks=(min(512, total_frames), state_dim), dtype="f4", compressor=compressor
    )
    action_array = data_group.create_dataset(
        "action", shape=(total_frames, action_dim), chunks=(min(512, total_frames), action_dim), dtype="f4", compressor=compressor
    )
    episode_ends = np.cumsum(lengths, dtype=np.int64)
    meta_group.create_dataset("episode_ends", data=episode_ends, dtype="i8", compressor=compressor)

    conversion = {
        "format": "a2w_front_depth_dp3_v1",
        "source_raw_root": str(raw_root),
        "episode_count": len(files),
        "total_frames": total_frames,
        "front_depth_key": "front_masked_depth",
        "front_camera_pose_key": "front_camera_pose_base",
        "camera_frame": "+X forward, +Y left, +Z up",
        "pixel_center": True,
        "pointcloud": cfg.to_dict(),
        "empty_depth_policy": str(args.empty_depth_policy),
        **meta,
    }
    root.attrs["conversion_json"] = json.dumps(conversion, ensure_ascii=False, sort_keys=True)

    debug_dir = args.debug_dir
    if debug_dir is None:
        debug_dir = output.parent / f"{output.stem}_debug"
    debug_dir = debug_dir.expanduser().resolve()
    debug_counts: dict[str, int] = {}
    debug_records = []
    empty_depth_fallbacks = []
    offset = 0
    start_time = time.time()
    for episode_index, (path, length) in enumerate(zip(files, lengths)):
        with np.load(path, allow_pickle=True) as episode:
            state_array[offset : offset + length] = np.asarray(episode["state"], dtype=np.float32)
            action_array[offset : offset + length] = np.asarray(episode["action"], dtype=np.float32)
            # NPZ members are deflate-compressed. Indexing the NpzFile member
            # in every mini-batch would decompress the same 480x640 movie over
            # and over, so materialize each episode exactly once.
            episode_depth = np.asarray(episode["front_masked_depth"], dtype=np.uint8)
            episode_poses = np.asarray(episode["front_camera_pose_base"], dtype=np.float32)
            door_name = str(scalar(episode, "door_asset_name", "unknown"))
            debug_count = debug_counts.get(door_name, 0)
            debug_frames = set()
            last_valid_points = None
            last_valid_frame = None
            remaining = max(0, int(args.debug_per_door) - debug_count)
            if remaining:
                debug_frames = set(np.rint(np.linspace(0, length - 1, remaining)).astype(int).tolist())
            for start in range(0, length, max(1, int(args.batch_frames))):
                end = min(length, start + max(1, int(args.batch_frames)))
                depth = episode_depth[start:end]
                poses = episode_poses[start:end]
                try:
                    points = front_depth_to_point_cloud_batch(depth, poses, meta["intrinsics"], cfg)
                except ValueError as exc:
                    if str(args.empty_depth_policy) != "previous" or "No valid front-depth points" not in str(exc):
                        raise ValueError(
                            f"{path.name} frames [{start},{end}) failed point-cloud conversion: {exc}") from exc
                    frame_points = []
                    for local_index in range(end - start):
                        frame_index = start + local_index
                        try:
                            current = front_depth_to_point_cloud_batch(
                                depth[local_index], poses[local_index], meta["intrinsics"], cfg)
                            last_valid_points = current
                            last_valid_frame = frame_index
                        except ValueError as frame_exc:
                            if "No valid front-depth points" not in str(frame_exc) or last_valid_points is None:
                                raise ValueError(
                                    f"{path.name} frame {frame_index} has no valid points and no previous cloud.") from frame_exc
                            current = last_valid_points.copy()
                            empty_depth_fallbacks.append(
                                {"episode": episode_index, "frame": frame_index, "source_frame": last_valid_frame})
                        frame_points.append(current)
                    points = np.stack(frame_points, axis=0)
                last_valid_points = points[-1]
                if not any(
                    record["episode"] == episode_index and start <= record["frame"] < end
                    for record in empty_depth_fallbacks
                ):
                    last_valid_frame = end - 1
                pc_array[offset + start : offset + end] = points
                for frame in sorted(debug_frames.intersection(range(start, end))):
                    stem = f"door_{door_name}_ep{episode_index:06d}_frame{frame:04d}"
                    write_ply(debug_dir / f"{stem}.ply", points[frame - start])
                    write_pointcloud_html(debug_dir / f"{stem}.html", points[frame - start])
                    save_depth_png(debug_dir / f"{stem}_depth.png", depth[frame - start])
                    debug_records.append(
                        {
                            "door": door_name,
                            "episode": episode_index,
                            "frame": frame,
                            "depth_path": f"{stem}_depth.png",
                            "pointcloud_path": f"{stem}.ply",
                            "pointcloud_html_path": f"{stem}.html",
                            "point_min": points[frame - start].min(axis=0).tolist(),
                            "point_max": points[frame - start].max(axis=0).tolist(),
                        }
                    )
                    debug_counts[door_name] = debug_counts.get(door_name, 0) + 1
            offset += length
        elapsed = time.time() - start_time
        print(
            f"[{episode_index + 1:03d}/{len(files):03d}] {path.name}: {length} frames; "
            f"written={offset}/{total_frames}; elapsed={elapsed:.1f}s",
            flush=True,
        )

    debug_dir.mkdir(parents=True, exist_ok=True)
    conversion["empty_depth_fallback_count"] = len(empty_depth_fallbacks)
    root.attrs["conversion_json"] = json.dumps(conversion, ensure_ascii=False, sort_keys=True)
    (debug_dir / "manifest.json").write_text(
        json.dumps(
            {"conversion": conversion, "samples": debug_records, "empty_depth_fallbacks": empty_depth_fallbacks},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done: {output}")
    print(f"point_cloud={pc_array.shape} state={state_array.shape} action={action_array.shape}")
    print(f"episode_ends[-1]={int(episode_ends[-1])}; debug={debug_dir}")
    print(f"empty_depth_fallbacks={len(empty_depth_fallbacks)} policy={args.empty_depth_policy}")


if __name__ == "__main__":
    main()
