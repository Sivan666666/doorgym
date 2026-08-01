#!/usr/bin/env python3
"""Run FoundationPose's official BundleSDF neural reconstruction on WC4 refs.

The implementation follows ``FoundationPose/bundlesdf/run_nerf.py`` but keeps
the extracted mesh untextured because FoundationPose's pose scorer/refiner only
requires geometry and vertex normals.  Avoiding UV unwrapping also makes this
small-object smoke test substantially faster and more robust.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh
import yaml


DEFAULT_FP_ROOT = Path("/home/sivan/whole_body/FoundationPose")
DEFAULT_REFS = Path(
    "/home/sivan/whole_body/visual_whole_body/high-level/logs/"
    "foundationpose/model_free_wc4_refs_sam3_16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose_root", type=Path, default=DEFAULT_FP_ROOT)
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFS)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n_step", type=int, default=1000)
    parser.add_argument("--mesh_resolution", type=float, default=0.003)
    parser.add_argument("--output_mesh", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.foundationpose_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else root / "bundlesdf/config_ycbv.yml"
    )
    output_mesh = (
        args.output_mesh.expanduser().resolve()
        if args.output_mesh is not None
        else reference_dir / "model/model.obj"
    )
    for required in (root, reference_dir, config_path, reference_dir / "K.txt"):
        if not required.exists():
            raise FileNotFoundError(required)

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "bundlesdf"))
    sys.path.insert(0, str(root / "bundlesdf/mycuda"))
    os.chdir(root)

    from nerf_runner import NerfRunner
    from nerf_helpers import (
        get_optimized_poses_in_real_world,
        mesh_to_real_world,
        preprocess_data,
    )
    from tool import compute_scene_bounds
    from Utils import glcam_in_cvcam

    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["n_step"] = int(args.n_step)
    cfg["mesh_resolution"] = float(args.mesh_resolution)
    # Extract only the final model; intermediate marching-cubes passes add no
    # value for this one-shot smoke test.
    cfg["i_mesh"] = max(1_000_000, int(args.n_step) + 1)
    cfg["i_img"] = max(1_000_000, int(args.n_step) + 1)
    cfg["i_nerf_normals"] = max(1_000_000, int(args.n_step) + 1)
    cfg["i_save_ray"] = max(1_000_000, int(args.n_step) + 1)

    color_files = sorted((reference_dir / "rgb").glob("*.png"))
    if not color_files:
        raise RuntimeError(f"No RGB reference frames in {reference_dir / 'rgb'}")
    K = np.loadtxt(reference_dir / "K.txt").reshape(3, 3)
    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    camera_in_object: list[np.ndarray] = []
    for color_file in color_files:
        stem = color_file.stem
        rgb = imageio.imread(color_file)[..., :3]
        depth = cv2.imread(str(reference_dir / "depth_enhanced" / f"{stem}.png"), -1)
        mask = cv2.imread(str(reference_dir / "mask" / f"{stem}.png"), -1)
        pose = np.loadtxt(reference_dir / "cam_in_ob" / f"{stem}.txt").reshape(4, 4)
        if depth is None or mask is None:
            raise FileNotFoundError(f"Missing depth/mask for {color_file}")
        rgbs.append(np.asarray(rgb, dtype=np.uint8))
        depths.append(np.asarray(depth, dtype=np.float32) / 1000.0)
        masks.append(np.asarray(mask, dtype=np.uint8))
        camera_in_object.append(pose)

    rgbs_array = np.asarray(rgbs)
    depths_array = np.asarray(depths)
    masks_array = np.asarray(masks)
    camera_in_object_array = np.asarray(camera_in_object)
    gl_camera_in_object = camera_in_object_array @ glcam_in_cvcam
    save_dir = reference_dir / "nerf"
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg["save_dir"] = str(save_dir)
    started = time.time()

    sc_factor, translation, _, normalized_cloud = compute_scene_bounds(
        None,
        gl_camera_in_object,
        K,
        use_mask=True,
        base_dir=str(save_dir),
        rgbs=rgbs_array,
        depths=depths_array,
        masks=masks_array,
        eps=cfg["dbscan_eps"],
        min_samples=cfg["dbscan_eps_min_samples"],
    )
    cfg["sc_factor"] = float(sc_factor)
    cfg["translation"] = np.asarray(translation, dtype=np.float64).tolist()

    rgbs_processed, depths_processed, masks_processed, _, poses_processed = preprocess_data(
        rgbs_array,
        depths_array,
        masks_array,
        normal_maps=None,
        poses=gl_camera_in_object,
        sc_factor=cfg["sc_factor"],
        translation=cfg["translation"],
    )
    nerf = NerfRunner(
        cfg,
        rgbs_processed,
        depths_processed,
        masks_processed,
        normal_maps=None,
        poses=poses_processed,
        K=K,
        occ_masks=None,
        build_octree_pcd=normalized_cloud,
    )
    nerf.train()
    mesh = nerf.extract_mesh(isolevel=0, voxel_size=cfg["mesh_resolution"])
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("Neural Object Field produced no mesh")
    _, pose_offset = get_optimized_poses_in_real_world(
        poses_processed,
        nerf.models["pose_array"],
        cfg["sc_factor"],
        cfg["translation"],
    )
    mesh = mesh_to_real_world(
        mesh,
        pose_offset=pose_offset,
        translation=cfg["translation"],
        sc_factor=cfg["sc_factor"],
    )
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh)

    summary = {
        "status": "ok",
        "method": "FoundationPose model-free BundleSDF Neural Object Field",
        "reference_dir": str(reference_dir),
        "output_mesh": str(output_mesh),
        "num_reference_views": len(color_files),
        "mask_source": "SAM3",
        "cad_mesh_input": False,
        "relative_pose_source": "recorded simulator camera-to-handle transforms",
        "n_step": int(args.n_step),
        "mesh_resolution_m": float(args.mesh_resolution),
        "sc_factor": float(cfg["sc_factor"]),
        "translation": cfg["translation"],
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_m": np.asarray(mesh.bounds).tolist(),
        "extents_m": np.asarray(mesh.extents).tolist(),
        "watertight": bool(mesh.is_watertight),
        "elapsed_s": float(time.time() - started),
    }
    (reference_dir / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    # Kaolin's SPC object in this legacy BundleSDF path can call an
    # incompatible native destructor during Python 3.11 interpreter teardown
    # (after all outputs are already safely written), producing
    # ``free(): invalid pointer``.  A process exit releases CUDA/host memory at
    # the OS boundary without invoking that stale destructor and gives the
    # standalone reconstruction command the correct success status.
    os._exit(0)


if __name__ == "__main__":
    main()
