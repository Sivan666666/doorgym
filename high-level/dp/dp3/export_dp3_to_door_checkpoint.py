#!/usr/bin/env python3
"""Package an official DP3 checkpoint for the Door play/eval backend."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    import dill
    import zarr

    args = parse_args()
    source_checkpoint = args.checkpoint.expanduser().resolve()
    zarr_path = args.zarr.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    if not zarr_path.is_dir():
        raise FileNotFoundError(zarr_path)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(zarr_path), mode="r")
    conversion = json.loads(root.attrs["conversion_json"])
    pointcloud = dict(conversion["pointcloud"])
    dual_view = conversion.get("format") == "a2w_dual_depth_dp3_v1"
    if dual_view:
        camera_intrinsics = dict(conversion["camera_intrinsics"])
        front_intrinsics = dict(camera_intrinsics["front"])
        wrist_intrinsics = dict(camera_intrinsics["wrist"])
    else:
        front_intrinsics = dict(conversion["intrinsics"])
        wrist_intrinsics = None
    state_names = list(conversion["state_feature_names"])
    action_names = list(conversion["action_names"])
    state_dim = int(root["data/state"].shape[1])
    action_dim = int(root["data/action"].shape[1])
    if len(state_names) != state_dim:
        raise ValueError(
            f"state_feature_names has {len(state_names)} entries for state dim {state_dim}.")
    if len(action_names) != action_dim:
        raise ValueError(
            f"action_names has {len(action_names)} entries for action dim {action_dim}.")
    if dual_view:
        for key in ("front_point_cloud", "wrist_point_cloud"):
            if root[f"data/{key}"].shape[1:] != (1024, 3):
                raise ValueError(
                    f"Expected {key} [T,1024,3], got {root[f'data/{key}'].shape}.")
    elif root["data/point_cloud"].shape[1:] != (1024, 3):
        raise ValueError(
            f"Expected [T,1024,3] point clouds, got {root['data/point_cloud'].shape}.")
    payload = torch.load(source_checkpoint, map_location="cpu", pickle_module=dill)
    train_cfg = payload.get("cfg")
    if train_cfg is None:
        raise ValueError("Official DP3 checkpoint does not contain its Hydra cfg.")
    obs_horizon = int(train_cfg.n_obs_steps)
    prediction_horizon = int(train_cfg.horizon)
    trained_action_horizon = int(train_cfg.n_action_steps)
    inference_steps = int(train_cfg.policy.num_inference_steps)

    # Keep the official payload intact: it contains the EMA model, DP3
    # normalizer, complete Hydra config, optimizer, and resume counters.
    packaged_checkpoint = output_dir / "dp3.ckpt"
    shutil.copy2(source_checkpoint, packaged_checkpoint)
    policy_config = {
        "backend": "dp3",
        "device": "cuda:0",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "obs_horizon": obs_horizon,
        "horizon": prediction_horizon,
        "action_horizon": trained_action_horizon,
        "num_inference_steps": inference_steps,
        "vision_mode": "depth_only",
        "pointcloud_conditioning": True,
        "pointcloud_mode": "dual_view" if dual_view else "single_front",
        "pointcloud_key": "observation.point_cloud",
        "front_pointcloud_key": "observation.front_point_cloud",
        "wrist_pointcloud_key": "observation.wrist_point_cloud",
        "front_depth_key": "front_masked_depth",
        "front_camera_pose_key": "front_camera_pose_base",
        "front_camera_intrinsics": front_intrinsics,
        "pointcloud_config": pointcloud,
        # A randomized wrist camera can occasionally have no in-workspace
        # points on the very first online frame.  Dual-view inference uses a
        # deterministic blank cloud until a valid wrist cloud is available;
        # subsequent empty frames still reuse the previous cloud, matching
        # the offline conversion policy.  The single-front path is unchanged.
        "pointcloud_empty_depth_policy": (
            "previous_or_zero"
            if dual_view and conversion.get("empty_depth_policy") == "previous"
            else conversion.get("empty_depth_policy", "error")
        ),
        "state_feature_names": state_names,
        "action_names": action_names,
        "action_frame": conversion.get("action_frame", "robot_base_full"),
        "ikpush_state_version": conversion.get("ikpush_state_version", "legacy"),
        "door_dp_mode": conversion.get("door_dp_mode", "ikpush"),
    }
    if dual_view:
        policy_config.update(
            {
                "wrist_depth_key": "wrist_masked_depth",
                "wrist_camera_pose_key": "wrist_camera_pose_base",
                "wrist_camera_intrinsics": wrist_intrinsics,
            }
        )
    sidecar = {
        "state": state_names,
        "action": action_names,
        "state_format": conversion.get("state_format", ""),
        "action_format": conversion.get("action_format", ""),
        "point_frame": pointcloud.get("point_frame", "robot_base"),
        "source_zarr": str(zarr_path),
    }
    meta = {
        "backend": "dp3",
        "format_version": 1,
        "vision_mode": "depth_only",
        "action_frame": policy_config["action_frame"],
        "dp3_checkpoint": packaged_checkpoint.name,
        "policy_config": policy_config,
        "sidecar_config": sidecar,
        "source_official_checkpoint": str(source_checkpoint),
        "source_zarr": str(zarr_path),
    }
    (output_dir / "door_policy_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = args.manifest
    if manifest is None:
        manifest = output_dir.parent / "model_latest.pt"
    manifest = manifest.expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backend": "dp3",
            "format_version": 1,
            "checkpoint_dir": str(output_dir),
            "checkpoint_dir_name": output_dir.name,
            "config": policy_config,
            "action_names": action_names,
        },
        manifest,
    )
    print(f"Door DP3 checkpoint: {output_dir}")
    print(f"Door manifest: {manifest}")


if __name__ == "__main__":
    main()
