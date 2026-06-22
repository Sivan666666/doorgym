"""Depth-camera pose jitter and light sensor-noise helpers for Door DP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


DEPTH_IMAGE_WIDTH = 640
DEPTH_IMAGE_HEIGHT = 480
DEPTH_CAMERA_RESOLUTION = [DEPTH_IMAGE_WIDTH, DEPTH_IMAGE_HEIGHT]


@dataclass(frozen=True)
class DepthNoiseConfig:
    enabled: bool = False
    noise_prob: float = 0.50
    env_selected: bool | None = None
    gaussian_std_m: float = 0.005
    gaussian_distance_factor: float = 0.05
    edge_noise_prob: float = 0.10
    edge_gradient_threshold_m: float = 0.05
    edge_dilation_kernel_size: int = 3
    hole_noise_prob: float = 0.01
    hole_block_size: int = 8
    dropout_prob: float = 0.0
    salt_pepper_prob: float = 0.0
    near_clip_m: float = 0.02
    far_clip_m: float = 2.0


def add_depth_aug_args(parser: Any) -> None:
    parser.add_argument("--enable_depth_noise", dest="enable_depth_noise", action="store_true", default=False)
    parser.add_argument("--no_enable_depth_noise", dest="enable_depth_noise", action="store_false")
    parser.add_argument(
        "--depth_noise_prob",
        type=float,
        default=0.50,
        help="Fraction of envs assigned persistent depth noise; selected envs apply noise to every frame.",
    )
    parser.add_argument("--depth_gaussian_std_m", type=float, default=0.005)
    parser.add_argument("--depth_edge_noise_prob", type=float, default=0.10)
    parser.add_argument("--depth_hole_noise_prob", type=float, default=0.01)
    parser.add_argument("--depth_dropout_prob", type=float, default=0.0)
    parser.add_argument("--enable_depth_camera_randomization", dest="enable_depth_camera_randomization", action="store_true", default=False)
    parser.add_argument("--no_enable_depth_camera_randomization", dest="enable_depth_camera_randomization", action="store_false")
    parser.add_argument("--depth_camera_pos_rand_m", type=float, default=0.01)
    parser.add_argument("--depth_camera_rot_rand_deg", type=float, default=2.0)


def depth_aug_custom_parameters() -> list[dict[str, Any]]:
    return [
        {"name": "--enable_depth_noise", "dest": "enable_depth_noise", "action": "store_true", "default": False},
        {"name": "--no_enable_depth_noise", "dest": "enable_depth_noise", "action": "store_false"},
        {
            "name": "--depth_noise_prob",
            "type": float,
            "default": 0.50,
            "help": "Fraction of envs assigned persistent depth noise; selected envs apply noise to every frame.",
        },
        {"name": "--depth_gaussian_std_m", "type": float, "default": 0.005},
        {"name": "--depth_edge_noise_prob", "type": float, "default": 0.10},
        {"name": "--depth_hole_noise_prob", "type": float, "default": 0.01},
        {"name": "--depth_dropout_prob", "type": float, "default": 0.0},
        {"name": "--enable_depth_camera_randomization", "dest": "enable_depth_camera_randomization", "action": "store_true", "default": False},
        {"name": "--no_enable_depth_camera_randomization", "dest": "enable_depth_camera_randomization", "action": "store_false"},
        {"name": "--depth_camera_pos_rand_m", "type": float, "default": 0.01},
        {"name": "--depth_camera_rot_rand_deg", "type": float, "default": 2.0},
    ]


def add_depth_aug_command_args(cmd: list[str], args: Any) -> None:
    if bool(getattr(args, "enable_depth_noise", False)):
        cmd.append("--enable_depth_noise")
    else:
        cmd.append("--no_enable_depth_noise")
    cmd += [
        "--depth_noise_prob",
        str(float(getattr(args, "depth_noise_prob", 0.50))),
        "--depth_gaussian_std_m",
        str(float(getattr(args, "depth_gaussian_std_m", 0.005))),
        "--depth_edge_noise_prob",
        str(float(getattr(args, "depth_edge_noise_prob", 0.10))),
        "--depth_hole_noise_prob",
        str(float(getattr(args, "depth_hole_noise_prob", 0.01))),
        "--depth_dropout_prob",
        str(float(getattr(args, "depth_dropout_prob", 0.0))),
    ]
    if bool(getattr(args, "enable_depth_camera_randomization", False)):
        cmd.append("--enable_depth_camera_randomization")
    else:
        cmd.append("--no_enable_depth_camera_randomization")
    cmd += [
        "--depth_camera_pos_rand_m",
        str(float(getattr(args, "depth_camera_pos_rand_m", 0.01))),
        "--depth_camera_rot_rand_deg",
        str(float(getattr(args, "depth_camera_rot_rand_deg", 2.0))),
    ]


def configure_depth_noise_for_env(args: Any) -> bool:
    if hasattr(args, "depth_noise_env_selected"):
        return bool(getattr(args, "depth_noise_env_selected"))

    requested = bool(getattr(args, "enable_depth_noise", False))
    probability = min(1.0, max(0.0, float(getattr(args, "depth_noise_prob", 0.50))))
    num_envs = max(1, int(getattr(args, "num_envs", 1)))
    env_index = min(num_envs - 1, max(0, int(getattr(args, "parallel_env_id", 0))))

    selected_count = int(math.floor(probability * num_envs + 0.5))
    if requested and probability > 0.0 and selected_count == 0:
        selected_count = 1
    if not requested:
        selected_count = 0
    selected_count = min(num_envs, max(0, selected_count))

    selected = False
    selected_env_ids: list[int] = []
    if requested and selected_count > 0:
        seed = int(getattr(args, "seed", 0)) & 0xFFFFFFFF
        selection_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xD3E7A5]))
        selected_env_ids = sorted(
            int(value)
            for value in selection_rng.choice(num_envs, size=selected_count, replace=False).tolist()
        )
        selected = env_index in selected_env_ids

    args.depth_noise_env_selected = bool(selected)
    args.depth_noise_env_probability = float(probability)
    args.depth_noise_selected_env_count = int(selected_count)
    args.depth_noise_selection_mode = "fixed_env_subset"
    return bool(selected)


def depth_noise_config_from_args(args: Any) -> DepthNoiseConfig:
    env_selected = configure_depth_noise_for_env(args)
    return DepthNoiseConfig(
        enabled=bool(getattr(args, "enable_depth_noise", False)),
        noise_prob=float(getattr(args, "depth_noise_prob", 0.50)),
        env_selected=bool(env_selected),
        gaussian_std_m=float(getattr(args, "depth_gaussian_std_m", 0.005)),
        edge_noise_prob=float(getattr(args, "depth_edge_noise_prob", 0.10)),
        hole_noise_prob=float(getattr(args, "depth_hole_noise_prob", 0.01)),
        dropout_prob=float(getattr(args, "depth_dropout_prob", 0.0)),
        near_clip_m=float(getattr(args, "camera_depth_clip_lower", 0.02)),
        far_clip_m=float(getattr(args, "camera_depth_clip_far", 2.0)),
    )


def depth_aug_metadata_from_args(args: Any) -> dict[str, Any]:
    cfg = depth_noise_config_from_args(args)
    return {
        "image_width": DEPTH_IMAGE_WIDTH,
        "image_height": DEPTH_IMAGE_HEIGHT,
        "depth_noise_enabled": bool(cfg.enabled),
        "depth_noise_selection": {
            "mode": str(getattr(args, "depth_noise_selection_mode", "fixed_env_subset")),
            "env_probability": float(getattr(args, "depth_noise_env_probability", cfg.noise_prob)),
            "env_selected": bool(cfg.env_selected),
            "selected_env_count": int(getattr(args, "depth_noise_selected_env_count", 0)),
            "num_envs": int(getattr(args, "num_envs", 1)),
        },
        "depth_noise_config": {
            "noise_prob": float(cfg.noise_prob),
            "noise_prob_semantics": "fixed_env_fraction",
            "gaussian_std_m": float(cfg.gaussian_std_m),
            "gaussian_distance_factor": float(cfg.gaussian_distance_factor),
            "edge_noise_prob": float(cfg.edge_noise_prob),
            "edge_gradient_threshold_m": float(cfg.edge_gradient_threshold_m),
            "edge_dilation_kernel_size": int(cfg.edge_dilation_kernel_size),
            "hole_noise_prob": float(cfg.hole_noise_prob),
            "hole_block_size": int(cfg.hole_block_size),
            "dropout_prob": float(cfg.dropout_prob),
            "salt_pepper_prob": float(cfg.salt_pepper_prob),
            "near_clip_m": float(cfg.near_clip_m),
            "far_clip_m": float(cfg.far_clip_m),
        },
        "depth_camera_randomization_config": {
            "enabled": bool(getattr(args, "enable_depth_camera_randomization", False)),
            "pos_rand_m": float(getattr(args, "depth_camera_pos_rand_m", 0.01)),
            "rot_rand_deg": float(getattr(args, "depth_camera_rot_rand_deg", 2.0)),
        },
    }


def _rng_uniform(rng: Any, low: float, high: float, size: int | tuple[int, ...]):
    if rng is None:
        return np.random.uniform(low, high, size=size)
    if hasattr(rng, "uniform"):
        return rng.uniform(low, high, size=size)
    return np.random.uniform(low, high, size=size)


def jitter_camera_pose(
    pos: Any,
    rot: Any,
    rng: Any = None,
    *,
    enabled: bool = False,
    pos_range_m: float = 0.01,
    rot_range_deg: float = 2.0,
) -> tuple[list[float], list[float]]:
    pos_out = np.asarray(pos, dtype=np.float32).reshape(3).copy()
    rot_out = np.asarray(rot, dtype=np.float32).reshape(3).copy()
    if enabled:
        pos_out += np.asarray(_rng_uniform(rng, -float(pos_range_m), float(pos_range_m), 3), dtype=np.float32)
        rot_out += np.asarray(
            _rng_uniform(rng, -math.radians(float(rot_range_deg)), math.radians(float(rot_range_deg)), 3),
            dtype=np.float32,
        )
    return [float(x) for x in pos_out], [float(x) for x in rot_out]


def _config_from_mapping(cfg: DepthNoiseConfig | Mapping[str, Any]) -> DepthNoiseConfig:
    if isinstance(cfg, DepthNoiseConfig):
        return cfg
    return DepthNoiseConfig(**dict(cfg))


def _is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch"


def _torch_modules():
    import torch
    import torch.nn.functional as F

    return torch, F


def apply_depth_noise(depth_m: Any, rng: Any, cfg: DepthNoiseConfig | Mapping[str, Any], valid_mask: Any = None):
    cfg = _config_from_mapping(cfg)
    if not cfg.enabled or cfg.noise_prob <= 0.0:
        return depth_m
    if cfg.env_selected is False:
        return depth_m
    if _is_torch_tensor(depth_m):
        return _apply_depth_noise_torch(depth_m, cfg, valid_mask)
    return _apply_depth_noise_numpy(depth_m, rng, cfg, valid_mask)


def _apply_depth_noise_numpy(depth_m: Any, rng: Any, cfg: DepthNoiseConfig, valid_mask: Any = None):
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    if cfg.env_selected is None and (
        rng.random() if rng is not None and hasattr(rng, "random") else np.random.random()
    ) >= cfg.noise_prob:
        return depth
    valid = np.isfinite(depth) & (depth >= cfg.near_clip_m) & (depth <= cfg.far_clip_m)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return depth
    if cfg.gaussian_std_m > 0.0:
        scale = cfg.gaussian_std_m * (1.0 + cfg.gaussian_distance_factor * depth)
        noise = (rng.normal(0.0, 1.0, depth.shape) if rng is not None and hasattr(rng, "normal") else np.random.normal(0.0, 1.0, depth.shape))
        depth[valid] += (noise.astype(np.float32) * scale.astype(np.float32))[valid]
    if cfg.edge_noise_prob > 0.0:
        grad_y, grad_x = np.gradient(depth)
        edge = np.sqrt(grad_x * grad_x + grad_y * grad_y) > cfg.edge_gradient_threshold_m
        if cfg.edge_dilation_kernel_size > 1:
            edge = _dilate_bool_numpy(edge, int(cfg.edge_dilation_kernel_size))
        random_edge = (
            rng.random(depth.shape) if rng is not None and hasattr(rng, "random") else np.random.random(depth.shape)
        ) < cfg.edge_noise_prob
        depth[edge & random_edge & valid] = 0.0
    if cfg.hole_noise_prob > 0.0:
        block = max(1, int(cfg.hole_block_size))
        sparse_shape = (max(depth.shape[-2] // block, 1), max(depth.shape[-1] // block, 1))
        sparse = (
            rng.random(sparse_shape) if rng is not None and hasattr(rng, "random") else np.random.random(sparse_shape)
        ) < cfg.hole_noise_prob
        holes = np.repeat(np.repeat(sparse, block, axis=0), block, axis=1)[: depth.shape[-2], : depth.shape[-1]]
        depth[holes & valid] = 0.0
    if cfg.dropout_prob > 0.0:
        drop = (
            rng.random(depth.shape) if rng is not None and hasattr(rng, "random") else np.random.random(depth.shape)
        ) < cfg.dropout_prob
        depth[drop & valid] = 0.0
    if cfg.salt_pepper_prob > 0.0:
        rand = rng.random(depth.shape) if rng is not None and hasattr(rng, "random") else np.random.random(depth.shape)
        depth[(rand < cfg.salt_pepper_prob * 0.5) & valid] = cfg.far_clip_m
        depth[((rand >= cfg.salt_pepper_prob * 0.5) & (rand < cfg.salt_pepper_prob)) & valid] = 0.0
    return np.clip(depth, 0.0, cfg.far_clip_m).astype(np.float32)


def _dilate_bool_numpy(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = max(0, kernel_size // 2)
    padded = np.pad(mask.astype(bool), ((pad, pad), (pad, pad)), mode="edge")
    out = np.zeros_like(mask, dtype=bool)
    for y in range(kernel_size):
        for x in range(kernel_size):
            out |= padded[y : y + mask.shape[0], x : x + mask.shape[1]]
    return out


def _apply_depth_noise_torch(depth_m: Any, cfg: DepthNoiseConfig, valid_mask: Any = None):
    torch, F = _torch_modules()
    depth = depth_m.clone()
    if cfg.env_selected is None:
        if depth.ndim >= 3:
            apply_shape = (*depth.shape[:-2], 1, 1)
        else:
            apply_shape = (1,) * depth.ndim
        apply_sample = torch.rand(apply_shape, device=depth.device) < float(cfg.noise_prob)
        if not bool(torch.any(apply_sample).detach().cpu().item()):
            return depth
    else:
        apply_sample = torch.ones(
            (*depth.shape[:-2], 1, 1) if depth.ndim >= 3 else (1,) * depth.ndim,
            dtype=torch.bool,
            device=depth.device,
        )
    valid = torch.isfinite(depth) & (depth >= float(cfg.near_clip_m)) & (depth <= float(cfg.far_clip_m))
    valid = valid & apply_sample
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=depth.device, dtype=torch.bool)
    if not bool(torch.any(valid).detach().cpu().item()):
        return depth
    if cfg.gaussian_std_m > 0.0:
        adaptive_std = float(cfg.gaussian_std_m) * (1.0 + float(cfg.gaussian_distance_factor) * depth)
        noise = torch.randn_like(depth) * adaptive_std
        depth = torch.where(valid, depth + noise, depth)
    if cfg.edge_noise_prob > 0.0 and depth.ndim >= 2:
        grad_x = torch.zeros_like(depth)
        grad_y = torch.zeros_like(depth)
        grad_x[..., :, 1:-1] = (depth[..., :, 2:] - depth[..., :, :-2]) * 0.5
        grad_x[..., :, 0] = depth[..., :, 1] - depth[..., :, 0]
        grad_x[..., :, -1] = depth[..., :, -1] - depth[..., :, -2]
        grad_y[..., 1:-1, :] = (depth[..., 2:, :] - depth[..., :-2, :]) * 0.5
        grad_y[..., 0, :] = depth[..., 1, :] - depth[..., 0, :]
        grad_y[..., -1, :] = depth[..., -1, :] - depth[..., -2, :]
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y) > float(cfg.edge_gradient_threshold_m)
        if cfg.edge_dilation_kernel_size > 1 and F is not None:
            original_shape = edge.shape
            edge_4d = edge.reshape(-1, 1, edge.shape[-2], edge.shape[-1]).float()
            edge = F.max_pool2d(
                edge_4d,
                kernel_size=int(cfg.edge_dilation_kernel_size),
                stride=1,
                padding=int(cfg.edge_dilation_kernel_size) // 2,
            ).reshape(original_shape).bool()
        edge_drop = torch.rand_like(depth) < float(cfg.edge_noise_prob)
        depth = torch.where(edge & edge_drop & valid, torch.zeros_like(depth), depth)
    if cfg.hole_noise_prob > 0.0 and depth.ndim >= 2 and F is not None:
        block = max(1, int(cfg.hole_block_size))
        batch_shape = depth.shape[:-2]
        h, w = depth.shape[-2:]
        sparse_h, sparse_w = max(h // block, 1), max(w // block, 1)
        sparse = torch.rand((*batch_shape, sparse_h, sparse_w), device=depth.device) < float(cfg.hole_noise_prob)
        sparse_4d = sparse.reshape(-1, 1, sparse_h, sparse_w).float()
        holes = F.interpolate(sparse_4d, size=(h, w), mode="nearest").reshape(depth.shape).bool()
        depth = torch.where(holes & valid, torch.zeros_like(depth), depth)
    if cfg.dropout_prob > 0.0:
        dropout = torch.rand_like(depth) < float(cfg.dropout_prob)
        depth = torch.where(dropout & valid, torch.zeros_like(depth), depth)
    if cfg.salt_pepper_prob > 0.0:
        rand = torch.rand_like(depth)
        salt = rand < float(cfg.salt_pepper_prob) * 0.5
        pepper = (rand >= float(cfg.salt_pepper_prob) * 0.5) & (rand < float(cfg.salt_pepper_prob))
        depth = torch.where(salt & valid, torch.full_like(depth, float(cfg.far_clip_m)), depth)
        depth = torch.where(pepper & valid, torch.zeros_like(depth), depth)
    return torch.clamp(depth, 0.0, float(cfg.far_clip_m))
