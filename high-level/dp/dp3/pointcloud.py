"""Deterministic front-depth point clouds shared by conversion and play.

Isaac Gym camera sensors look along local +X. Local +Y is image-left and
local +Z is image-up, so pixels right/down have negative Y/Z components.
Depth values are forward-axis distances rather than Euclidean ray lengths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FrontDepthPointCloudConfig:
    num_points: int = 1024
    candidate_rows: int = 64
    candidate_cols: int = 64
    workspace_min: tuple[float, float, float] = (0.20, -1.00, 0.00)
    workspace_max: tuple[float, float, float] = (2.00, 1.00, 1.80)
    near_clip_m: float = 0.20
    far_clip_m: float = 1.50
    point_frame: str = "robot_base"
    sampling: str = "deterministic_image_grid"

    def validate(self) -> None:
        if self.num_points <= 0:
            raise ValueError(f"num_points must be positive, got {self.num_points}.")
        if self.candidate_rows <= 0 or self.candidate_cols <= 0:
            raise ValueError("candidate_rows/candidate_cols must be positive.")
        if self.candidate_rows * self.candidate_cols < self.num_points:
            raise ValueError(
                "The candidate grid must contain at least num_points pixels; "
                f"got {self.candidate_rows}x{self.candidate_cols} < {self.num_points}."
            )
        if self.near_clip_m < 0 or self.far_clip_m <= self.near_clip_m:
            raise ValueError(
                f"Invalid depth range [{self.near_clip_m}, {self.far_clip_m}]."
            )
        lo = np.asarray(self.workspace_min, dtype=np.float32)
        hi = np.asarray(self.workspace_max, dtype=np.float32)
        if lo.shape != (3,) or hi.shape != (3,) or np.any(hi <= lo):
            raise ValueError(f"Invalid workspace bounds: min={lo}, max={hi}.")
        if self.point_frame != "robot_base":
            raise ValueError(f"Only point_frame='robot_base' is supported, got {self.point_frame!r}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def intrinsics_dict(value: Any, camera: str = "front") -> dict[str, float]:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, Mapping) and camera in value:
        value = value[camera]
    if not isinstance(value, Mapping):
        raise ValueError(f"Camera intrinsics must be a mapping, got {type(value).__name__}.")
    required = ("fx", "fy", "cx", "cy", "width", "height")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Camera intrinsics are missing keys: {missing}.")
    result = {key: float(value[key]) for key in required}
    result["width"] = int(result["width"])
    result["height"] = int(result["height"])
    return result


@lru_cache(maxsize=32)
def _candidate_pixels(height: int, width: int, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.rint(np.linspace(0, height - 1, rows)).astype(np.int64)
    xs = np.rint(np.linspace(0, width - 1, cols)).astype(np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy.reshape(-1), xx.reshape(-1)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected xyzw quaternion(s), got shape {quat.shape}.")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("Camera pose contains a zero-length quaternion.")
    x, y, z, w = np.moveaxis(quat / norm, -1, 0)
    row0 = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), axis=-1)
    row1 = np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), axis=-1)
    row2 = np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), axis=-1)
    return np.stack((row0, row1, row2), axis=-2).astype(np.float32)


def decode_depth_u8(depth_u8: np.ndarray, near_clip_m: float, far_clip_m: float) -> tuple[np.ndarray, np.ndarray]:
    depth_u8 = np.asarray(depth_u8)
    if depth_u8.ndim not in (2, 3):
        raise ValueError(f"Expected depth shaped HxW or BxHxW, got {depth_u8.shape}.")
    if depth_u8.dtype != np.uint8:
        depth_u8 = np.clip(depth_u8, 0, 255).astype(np.uint8)
    valid = depth_u8 > 0
    depth_m = float(near_clip_m) + depth_u8.astype(np.float32) * (
        (float(far_clip_m) - float(near_clip_m)) / 255.0
    )
    depth_m[~valid] = 0.0
    return depth_m, valid


def _select_fixed_count(points: np.ndarray, num_points: int) -> np.ndarray:
    count = int(points.shape[0])
    if count == 0:
        raise ValueError("No valid depth points remain after workspace filtering.")
    if count >= num_points:
        indices = np.floor(np.linspace(0, count, num_points, endpoint=False)).astype(np.int64)
        return points[indices]
    indices = np.arange(num_points, dtype=np.int64) % count
    return points[indices]


def front_depth_to_point_cloud_batch(
    depth_u8: np.ndarray,
    camera_pose_base: np.ndarray,
    intrinsics: Mapping[str, Any],
    config: FrontDepthPointCloudConfig | None = None,
) -> np.ndarray:
    """Convert BxHxW uint8 front depth to BxNx3 base-frame XYZ."""
    cfg = config or FrontDepthPointCloudConfig()
    cfg.validate()
    depth_u8 = np.asarray(depth_u8)
    # Runtime camera images are commonly HxWx3 (the same quantized depth
    # repeated into RGB channels), whereas raw episodes store HxW. Accept
    # both without ever treating H as a batch dimension.
    single_hwc = (
        depth_u8.ndim == 3
        and depth_u8.shape[-1] in (1, 3, 4)
        and depth_u8.shape[0] == int(intrinsics_dict(intrinsics)["height"])
        and depth_u8.shape[1] == int(intrinsics_dict(intrinsics)["width"])
    )
    if single_hwc:
        depth_u8 = depth_u8[..., 0]
    single = depth_u8.ndim == 2
    if single:
        depth_u8 = depth_u8[None]
    if depth_u8.ndim == 4 and depth_u8.shape[-1] >= 1:
        depth_u8 = depth_u8[..., 0]
    if depth_u8.ndim != 3:
        raise ValueError(f"Expected BxHxW front depth, got {depth_u8.shape}.")

    poses = np.asarray(camera_pose_base, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses[None]
    if poses.shape != (depth_u8.shape[0], 7):
        raise ValueError(f"Expected camera poses {(depth_u8.shape[0], 7)}, got {poses.shape}.")

    k = intrinsics_dict(intrinsics)
    height, width = depth_u8.shape[1:]
    if (height, width) != (int(k["height"]), int(k["width"])):
        raise ValueError(
            f"Depth shape {height}x{width} disagrees with intrinsics {int(k['height'])}x{int(k['width'])}."
        )
    yy, xx = _candidate_pixels(height, width, cfg.candidate_rows, cfg.candidate_cols)
    sampled_u8 = depth_u8[:, yy, xx]
    depth_m, valid = decode_depth_u8(sampled_u8, cfg.near_clip_m, cfg.far_clip_m)

    u_factor = -((xx.astype(np.float32) + 0.5) - float(k["cx"])) / float(k["fx"])
    v_factor = -((yy.astype(np.float32) + 0.5) - float(k["cy"])) / float(k["fy"])
    camera_points = np.stack(
        (depth_m, depth_m * u_factor[None], depth_m * v_factor[None]),
        axis=-1,
    )
    rotations = quat_xyzw_to_matrix(poses[:, 3:7])
    base_points = np.einsum("bij,bnj->bni", rotations, camera_points) + poses[:, None, :3]
    lo = np.asarray(cfg.workspace_min, dtype=np.float32)
    hi = np.asarray(cfg.workspace_max, dtype=np.float32)
    valid &= np.all(base_points >= lo[None, None], axis=-1)
    valid &= np.all(base_points <= hi[None, None], axis=-1)
    valid &= np.all(np.isfinite(base_points), axis=-1)

    result = np.empty((depth_u8.shape[0], cfg.num_points, 3), dtype=np.float32)
    for batch_index in range(depth_u8.shape[0]):
        result[batch_index] = _select_fixed_count(base_points[batch_index, valid[batch_index]], cfg.num_points)
    return result[0] if single else result


def front_depth_to_point_cloud(
    depth_u8: np.ndarray,
    camera_pose_base: Sequence[float] | np.ndarray,
    intrinsics: Mapping[str, Any],
    config: FrontDepthPointCloudConfig | None = None,
) -> np.ndarray:
    return front_depth_to_point_cloud_batch(depth_u8, camera_pose_base, intrinsics, config)


# View-neutral aliases. Keeping the historical front_* names preserves DP3 checkpoints
# and scripts while allowing wrist and fused ACT data to use the exact same geometry code.
depth_to_point_cloud_batch = front_depth_to_point_cloud_batch
depth_to_point_cloud = front_depth_to_point_cloud


def fuse_depth_views_to_point_cloud(
    *,
    front_depth_u8: np.ndarray,
    front_camera_pose_base: Sequence[float] | np.ndarray,
    front_intrinsics: Mapping[str, Any],
    wrist_depth_u8: np.ndarray,
    wrist_camera_pose_base: Sequence[float] | np.ndarray,
    wrist_intrinsics: Mapping[str, Any],
    config: FrontDepthPointCloudConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse equally sized front/wrist clouds in the shared robot-base frame.

    Returns ``(fused, source_id, per_view_valid_clouds)`` where source_id is 0
    for front and 1 for wrist. The final element is an object-free stacked
    ``[2, N/2, 3]`` array useful for visualization/tests.
    """
    if int(config.num_points) % 2 != 0:
        raise ValueError("Dual-view point_cloud_num_points must be even.")
    per_view_cfg = FrontDepthPointCloudConfig(
        **{**config.to_dict(), "num_points": int(config.num_points) // 2}
    )
    front = depth_to_point_cloud(
        front_depth_u8, front_camera_pose_base, front_intrinsics, per_view_cfg
    )
    wrist = depth_to_point_cloud(
        wrist_depth_u8, wrist_camera_pose_base, wrist_intrinsics, per_view_cfg
    )
    views = np.stack((front, wrist), axis=0).astype(np.float32, copy=False)
    fused = np.concatenate((front, wrist), axis=0).astype(np.float32, copy=False)
    source_id = np.concatenate(
        (np.zeros(front.shape[0], dtype=np.uint8), np.ones(wrist.shape[0], dtype=np.uint8))
    )
    return fused, source_id, views
