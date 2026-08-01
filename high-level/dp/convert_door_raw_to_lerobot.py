import argparse
import json
import math
import shutil
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .dp3.pointcloud import FrontDepthPointCloudConfig, depth_to_point_cloud, intrinsics_dict
except ImportError:
    from dp3.pointcloud import FrontDepthPointCloudConfig, depth_to_point_cloud, intrinsics_dict

try:
    from .door_dp_common import (
        ACTION_NAMES,
        ACTION_LOSS_WEIGHT_FEATURE,
        FRONT_CAMERA_POSE_FEATURE,
        FRONT_HANDLE_LATENT_FEATURE,
        FRONT_HANDLE_LATENT_VALID_FEATURE,
        RAW_FRONT_CAMERA_POSE_KEY,
        RAW_FRONT_HANDLE_BBOX_KEY,
        RAW_FRONT_HANDLE_BBOX_VALID_KEY,
        RAW_WRIST_CAMERA_POSE_KEY,
        RAW_WRIST_HANDLE_BBOX_KEY,
        RAW_WRIST_HANDLE_BBOX_VALID_KEY,
        RECOVERY_INDICATOR_FEATURE,
        DATASET_METADATA_KEYS,
        DEFAULT_KEYFRAME_LOSS_RADIUS,
        DEFAULT_KEYFRAME_LOSS_WEIGHT,
        DEFAULT_NEAR_ZERO_RATE_EPS,
        DEFAULT_END_SIGNAL_POSITIVE_PHASES,
        DoorDPLeRobotRecorder,
        END_SIGNAL_FEATURE,
        INTERACTION_CONTACT_FEATURE,
        INTERACTION_DOOR_PROGRESS_FEATURE,
        INTERACTION_HANDLE_PROGRESS_FEATURE,
        INTERACTION_STATE_FEATURES,
        POINT_CLOUD_FEATURE,
        RAW_END_SIGNAL_KEY,
        RAW_ACTION_LOSS_WEIGHT_KEY,
        WRIST_HANDLE_LATENT_FEATURE,
        WRIST_HANDLE_LATENT_VALID_FEATURE,
        WRIST_CAMERA_POSE_FEATURE,
        apply_door_dp_action_preprocess,
        apply_door_dp_state_preprocess,
        extract_motion_keyframes_from_raw_arrays,
        fit_door_dp_action_preprocess,
        fit_door_dp_state_preprocess,
        image_to_three_channel_uint8,
        lerobot_image_keys_for_vision_mode,
        make_keyframe_action_loss_weight,
        make_end_signal_from_phase_ids,
        make_interaction_state_targets,
        make_door_dp_sanitize_config,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
        sanitize_door_dp_action,
        sanitize_door_dp_state,
    )
except ImportError:
    from door_dp_common import (
        ACTION_NAMES,
        ACTION_LOSS_WEIGHT_FEATURE,
        FRONT_CAMERA_POSE_FEATURE,
        FRONT_HANDLE_LATENT_FEATURE,
        FRONT_HANDLE_LATENT_VALID_FEATURE,
        RAW_FRONT_CAMERA_POSE_KEY,
        RAW_FRONT_HANDLE_BBOX_KEY,
        RAW_FRONT_HANDLE_BBOX_VALID_KEY,
        RAW_WRIST_CAMERA_POSE_KEY,
        RAW_WRIST_HANDLE_BBOX_KEY,
        RAW_WRIST_HANDLE_BBOX_VALID_KEY,
        RECOVERY_INDICATOR_FEATURE,
        DATASET_METADATA_KEYS,
        DEFAULT_KEYFRAME_LOSS_RADIUS,
        DEFAULT_KEYFRAME_LOSS_WEIGHT,
        DEFAULT_NEAR_ZERO_RATE_EPS,
        DEFAULT_END_SIGNAL_POSITIVE_PHASES,
        DoorDPLeRobotRecorder,
        END_SIGNAL_FEATURE,
        INTERACTION_CONTACT_FEATURE,
        INTERACTION_DOOR_PROGRESS_FEATURE,
        INTERACTION_HANDLE_PROGRESS_FEATURE,
        INTERACTION_STATE_FEATURES,
        POINT_CLOUD_FEATURE,
        RAW_END_SIGNAL_KEY,
        RAW_ACTION_LOSS_WEIGHT_KEY,
        WRIST_HANDLE_LATENT_FEATURE,
        WRIST_HANDLE_LATENT_VALID_FEATURE,
        WRIST_CAMERA_POSE_FEATURE,
        apply_door_dp_action_preprocess,
        apply_door_dp_state_preprocess,
        extract_motion_keyframes_from_raw_arrays,
        fit_door_dp_action_preprocess,
        fit_door_dp_state_preprocess,
        image_to_three_channel_uint8,
        lerobot_image_keys_for_vision_mode,
        make_keyframe_action_loss_weight,
        make_end_signal_from_phase_ids,
        make_interaction_state_targets,
        make_door_dp_sanitize_config,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
        sanitize_door_dp_action,
        sanitize_door_dp_state,
    )


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent

RAW_STATE_ACTION_MODE = "raw"
DIRECT_JOINT_STATE9_MODE = "joint_state9"
TRACIK_JOINT_STATE9_MODE = "tracik_joint_state9"
JOINT_STATE9_MODES = (DIRECT_JOINT_STATE9_MODE, TRACIK_JOINT_STATE9_MODE)
TRACIK_JOINT_STATE9_NAMES = [
    "last_command_vx",
    "last_command_vyaw",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "jointGripper",
]
TRACIK_JOINT_ACTION9_NAMES = [
    "vx",
    "yaw",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "jointGripper",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw Door DP .npz episodes into a local LeRobotDataset.")
    parser.add_argument("--raw_root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp"))
    parser.add_argument(
        "--additional_raw_root",
        action="append",
        default=[],
        help=(
            "Additional compatible raw episode root to append to --raw_root. Repeat this option to mix "
            "multiple roots without copying depth data. Episodes are written in root order."
        ),
    )
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--state_action_mode",
        choices=[RAW_STATE_ACTION_MODE, DIRECT_JOINT_STATE9_MODE, TRACIK_JOINT_STATE9_MODE],
        default=RAW_STATE_ACTION_MODE,
        help=(
            "raw keeps the recorded state/action arrays and metadata. joint_state9 reads an "
            "already-recorded 9D joint state/action dataset and marks actions as joint commands. "
            "tracik_joint_state9 derives "
            "state=[last base command,current arm q,current gripper] and "
            "action=[base command,TRAC-IK q_command,gripper command] from an EE-state raw dataset."
        ),
    )
    parser.add_argument("--rgb", action="store_true", help="Convert raw RGB+mask Door DP data. Required for RGB raw data.")
    parser.add_argument(
        "--depth_only",
        dest="depth_only",
        action="store_true",
        default=True,
        help="Convert raw depth-only Door DP data with no mask image fields (default).",
    )
    parser.add_argument(
        "--no_depth_only",
        dest="depth_only",
        action="store_false",
        help="Convert legacy depth+mask Door DP data.",
    )
    parser.add_argument(
        "--image_storage",
        choices=["video", "image"],
        default="video",
        help="Store visual observations as LeRobot v3 videos by default; use 'image' for embedded parquet images.",
    )
    parser.add_argument("--video_codec", type=str, default="h264", help="Video codec used when --image_storage video.")
    parser.add_argument(
        "--point_cloud_views",
        choices=["front", "wrist", "front,wrist"],
        default=None,
        help="Convert depth to one fused robot-base point cloud from front, wrist, or both views.",
    )
    parser.add_argument("--point_cloud_num_points", type=int, default=1024)
    parser.add_argument("--point_cloud_candidate_rows", type=int, default=64)
    parser.add_argument("--point_cloud_candidate_cols", type=int, default=64)
    parser.add_argument("--point_cloud_workspace_min", type=str, default="0.20,-1.00,0.00")
    parser.add_argument("--point_cloud_workspace_max", type=str, default="2.00,1.00,1.80")
    parser.add_argument(
        "--point_cloud_empty_depth_policy",
        choices=["previous", "error"],
        default="previous",
    )
    parser.add_argument(
        "--point_cloud_storage",
        choices=["point_cloud_only"],
        default="point_cloud_only",
        help="Point-cloud datasets omit depth videos to avoid lossy H264 depth storage.",
    )
    parser.add_argument(
        "--include_recovery_indicator",
        action="store_true",
        help=(
            "Write per-frame aux.is_recovery. Raw episodes with scalar/sequence aux.is_recovery use that "
            "value; episodes without it default to 0. Required for --recovery_sampling_ratio training."
        ),
    )
    parser.add_argument(
        "--add_end_signal",
        action="store_true",
        help=(
            "Write aux.end_signal. Prefer an explicit raw end_signal field; otherwise backfill it "
            "from subtask_index and phase_names."
        ),
    )
    parser.add_argument(
        "--end_signal_positive_phases",
        type=str,
        default=",".join(DEFAULT_END_SIGNAL_POSITIVE_PHASES),
        help="Comma-separated phases labeled 1; every other phase is labeled 0.",
    )
    parser.add_argument(
        "--add_interaction_state",
        action="store_true",
        help="Derive current-frame contact/handle/door interaction targets from privileged raw arrays.",
    )
    parser.add_argument("--interaction_contact_min_consecutive_frames", type=int, default=3)
    parser.add_argument("--interaction_handle_unlock_angle_deg", type=float, default=40.0)
    parser.add_argument("--interaction_door_goal_angle_deg", type=float, default=90.0)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help=(
            "Number of worker threads used to preload/validate raw npz episodes. "
            "LeRobot writing stays ordered in the main process. Defaults to 4."
        ),
    )
    parser.add_argument(
        "--keep_phase_state",
        action="store_true",
        help="Keep legacy phase_* one-hot columns in observation.state. By default they are removed.",
    )
    parser.add_argument(
        "--state_preprocess",
        choices=["robust_quantile", "none"],
        default="none",
        help=(
            "Preprocess observation.state while converting raw npz to LeRobot. "
            "Default none keeps physical values; robust_quantile clips to dataset quantiles and stores "
            "normalized state in [-1, 1]."
        ),
    )
    parser.add_argument("--state_quantile_low", type=float, default=0.01)
    parser.add_argument("--state_quantile_high", type=float, default=0.99)
    parser.add_argument("--state_preprocess_eps", type=float, default=1.0e-6)
    parser.add_argument(
        "--near_zero_rate_eps",
        type=float,
        default=DEFAULT_NEAR_ZERO_RATE_EPS,
        help="Set tiny yaw/yaw_rate/angular velocity values with abs(value) <= eps to 0 before writing LeRobot.",
    )
    parser.add_argument(
        "--action_preprocess",
        choices=["robust_quantile", "none"],
        default="none",
        help=(
            "Preprocess action while converting raw npz to LeRobot. Default none keeps physical commands; "
            "robust_quantile stores normalized action in [-1, 1]."
        ),
    )
    parser.add_argument("--action_quantile_low", type=float, default=0.01)
    parser.add_argument("--action_quantile_high", type=float, default=0.99)
    parser.add_argument("--action_preprocess_eps", type=float, default=1.0e-6)
    parser.add_argument(
        "--keyframe_loss_weight",
        type=float,
        default=None,
        help=(
            "Override the keyframe action-loss weight stored in raw episodes. "
            "When set, loss.action_weight is recomputed from keyframe_indices."
        ),
    )
    parser.add_argument(
        "--keyframe_loss_radius",
        type=int,
        default=None,
        help=(
            "Override the keyframe window radius stored in raw episodes. "
            "When set, loss.action_weight is recomputed from keyframe_indices."
        ),
    )
    parser.add_argument(
        "--add_handle_latent",
        action="store_true",
        help="Precompute frozen DINOv2 handle depth-crop latent targets and store them as aux.* features.",
    )
    parser.add_argument("--handle_latent_teacher_model", type=str, default="facebook/dinov2-small")
    parser.add_argument("--handle_latent_crop_size", type=int, default=224)
    parser.add_argument("--handle_latent_bbox_margin", type=float, default=0.2)
    parser.add_argument("--handle_bbox_min_area", type=float, default=20.0)
    parser.add_argument("--handle_bbox_min_size", type=float, default=3.0)
    parser.add_argument("--handle_latent_batch_size", type=int, default=64)
    return parser.parse_args()


def load_sidecar(raw_root):
    sidecar = Path(raw_root) / "door_dp_feature_names.json"
    if not sidecar.exists():
        return None
    with open(sidecar, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_calibrated_camera_metadata(sidecar, first_episode, episode_path):
    if not sidecar or sidecar.get("camera_intrinsics_mode") != "real_k_remap":
        return
    required = [
        "camera_intrinsics",
        "camera_render_intrinsics",
        "camera_intrinsics_remap_version",
        "render_resolution",
        "output_resolution",
    ]
    missing = [key for key in required if key not in sidecar]
    if missing:
        raise ValueError(f"real_k_remap raw sidecar is missing metadata: {missing}.")
    camera_intrinsics = sidecar["camera_intrinsics"]
    for camera_name in ("front", "wrist"):
        values = camera_intrinsics.get(camera_name) or {}
        missing_k = [key for key in ("fx", "fy", "cx", "cy", "width", "height") if key not in values]
        if missing_k:
            raise ValueError(f"real_k_remap metadata for {camera_name} is missing {missing_k}.")
    expected_width, expected_height = [int(value) for value in sidecar["output_resolution"]]
    for key in ("wrist_masked_depth", "front_masked_depth", "wrist_rgb", "front_rgb"):
        if key not in first_episode.files:
            continue
        shape = np.asarray(first_episode[key]).shape
        if len(shape) < 3 or tuple(shape[1:3]) != (expected_height, expected_width):
            raise ValueError(
                f"{episode_path} field {key!r} has shape {shape}, expected "
                f"(T, {expected_height}, {expected_width}, ...)."
            )


def episode_files(raw_root):
    files = sorted(Path(raw_root).glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found under {raw_root}")
    return files


def scalar_str(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def parse_csv_names(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_xyz_csv(value, name):
    try:
        result = tuple(float(item.strip()) for item in str(value).split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must contain three comma-separated floats, got {value!r}.") from exc
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values, got {value!r}.")
    return result


def make_episode_point_clouds(
    *,
    path,
    views,
    wrist_depth,
    front_depth,
    wrist_camera_pose_base,
    front_camera_pose_base,
    camera_intrinsics,
    config,
    empty_depth_policy,
):
    """Use the same deterministic geometry path as online Door inference."""
    requested_views = tuple(str(views).split(","))
    if requested_views == ("front", "wrist") and config.num_points % 2:
        raise ValueError("Dual-view point_cloud_num_points must be even.")
    per_view_count = config.num_points if len(requested_views) == 1 else config.num_points // 2
    per_view_config = FrontDepthPointCloudConfig(
        **{**config.to_dict(), "num_points": int(per_view_count)}
    )
    depth_by_view = {"front": front_depth, "wrist": wrist_depth}
    pose_by_view = {"front": front_camera_pose_base, "wrist": wrist_camera_pose_base}
    previous = {view: None for view in requested_views}
    result = np.empty((front_depth.shape[0], config.num_points, 3), dtype=np.float32)
    for frame_index in range(result.shape[0]):
        clouds = []
        for view in requested_views:
            try:
                cloud = depth_to_point_cloud(
                    depth_by_view[view][frame_index],
                    pose_by_view[view][frame_index],
                    intrinsics_dict(camera_intrinsics, camera=view),
                    per_view_config,
                )
                previous[view] = cloud
            except ValueError as exc:
                if empty_depth_policy != "previous" or previous[view] is None:
                    raise ValueError(
                        f"Point-cloud conversion failed for episode={path}, camera={view}, "
                        f"frame={frame_index}: {exc}"
                    ) from exc
                cloud = previous[view]
            clouds.append(cloud)
        result[frame_index] = np.concatenate(clouds, axis=0)
    return result


def _normalize_handle_bbox_array(value, frame_count, key, path):
    array = np.asarray(value, dtype=np.float32).reshape(-1, 4)
    if array.shape[0] != frame_count:
        raise ValueError(f"Episode {path} has {key} length {array.shape[0]}, expected {frame_count}.")
    return array


def _normalize_handle_valid_array(value, frame_count, key, path):
    array = np.asarray(value, dtype=np.float32).reshape(-1, 1)
    if array.shape[0] != frame_count:
        raise ValueError(f"Episode {path} has {key} length {array.shape[0]}, expected {frame_count}.")
    return array


def _square_depth_crop_with_padding(depth_u8, bbox_xyxy, crop_size, margin, min_area, min_size, valid=True):
    if not valid:
        return None
    image = np.asarray(depth_u8)
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        return None
    height, width = image.shape
    x0, y0, x1, y1 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)]
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    if bw < float(min_size) or bh < float(min_size) or bw * bh < float(min_area):
        return None
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(bw, bh) * (1.0 + 2.0 * max(0.0, float(margin)))
    if side < float(min_size):
        return None
    left = int(np.floor(cx - 0.5 * side))
    top = int(np.floor(cy - 0.5 * side))
    right = int(np.ceil(cx + 0.5 * side))
    bottom = int(np.ceil(cy + 0.5 * side))
    if right <= left or bottom <= top:
        return None

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    crop = padded[top + pad_top : bottom + pad_top, left + pad_left : right + pad_left]
    if crop.size <= 0:
        return None
    crop_t = torch.from_numpy(crop.astype(np.float32, copy=False)).view(1, 1, crop.shape[0], crop.shape[1]) / 255.0
    crop_t = F.interpolate(crop_t, size=(int(crop_size), int(crop_size)), mode="bilinear", align_corners=False)
    crop_t = crop_t.repeat(1, 3, 1, 1).squeeze(0)
    return crop_t


class DINOv2HandleLatentTeacher:
    def __init__(self, model_name, device=None):
        try:
            from transformers import AutoModel
        except Exception as exc:
            raise RuntimeError(
                "DINOv2 handle-latent conversion requires the `transformers` package. "
                "Run this converter in the b1z1_lerobot environment."
            ) from exc
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModel.from_pretrained(str(model_name)).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        hidden_size = int(getattr(self.model.config, "hidden_size"))
        if hidden_size != 384:
            raise ValueError(
                f"Handle latent v1 expects DINOv2-small hidden size 384, got {hidden_size} from {model_name!r}."
            )
        self.registered_model_name = str(model_name)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_crops(self, crops, batch_size=64):
        if not crops:
            return np.zeros((0, 384), dtype=np.float32)
        latents = []
        batch_size = max(1, int(batch_size))
        for start in range(0, len(crops), batch_size):
            batch = torch.stack(crops[start : start + batch_size], dim=0).to(self.device, dtype=torch.float32)
            batch = (batch - self.mean) / self.std
            outputs = self.model(pixel_values=batch)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            pooled = F.normalize(patch_tokens.mean(dim=1), p=2, dim=-1)
            latents.append(pooled.detach().cpu().to(torch.float32).numpy())
        return np.concatenate(latents, axis=0).astype(np.float32, copy=False)


def compute_episode_handle_latents(
    payload,
    teacher,
    *,
    crop_size,
    margin,
    min_area,
    min_size,
    batch_size,
):
    n = int(payload["n"])
    outputs = {
        "front_handle_latent": np.zeros((n, 384), dtype=np.float32),
        "front_handle_latent_valid": np.zeros((n, 1), dtype=np.float32),
        "wrist_handle_latent": np.zeros((n, 384), dtype=np.float32),
        "wrist_handle_latent_valid": np.zeros((n, 1), dtype=np.float32),
    }
    for view, depth_key, bbox_key, valid_key in (
        ("front", "front_second", "front_handle_bbox_xyxy", "front_handle_bbox_valid"),
        ("wrist", "wrist_second", "wrist_handle_bbox_xyxy", "wrist_handle_bbox_valid"),
    ):
        crops = []
        crop_indices = []
        depths = payload[depth_key]
        bboxes = payload[bbox_key]
        valids = payload[valid_key].reshape(-1)
        for idx in range(n):
            crop = _square_depth_crop_with_padding(
                depths[idx],
                bboxes[idx],
                crop_size=crop_size,
                margin=margin,
                min_area=min_area,
                min_size=min_size,
                valid=bool(valids[idx] > 0.5),
            )
            if crop is None:
                continue
            crops.append(crop)
            crop_indices.append(idx)
        latents = teacher.encode_crops(crops, batch_size=batch_size)
        if crop_indices:
            outputs[f"{view}_handle_latent"][np.asarray(crop_indices, dtype=np.int64)] = latents
            outputs[f"{view}_handle_latent_valid"][np.asarray(crop_indices, dtype=np.int64), 0] = 1.0
    return outputs


def array_to_str_list(value):
    return [str(item) for item in np.asarray(value, dtype=object).reshape(-1).tolist()]


def state_action_arrays(data, mode, path=None):
    mode = str(mode or RAW_STATE_ACTION_MODE)
    if mode == RAW_STATE_ACTION_MODE:
        return data["state"].astype(np.float32), data["action"].astype(np.float32)
    if mode == DIRECT_JOINT_STATE9_MODE:
        states = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["action"], dtype=np.float32)
        label = str(path or "raw episode")
        if states.ndim != 2 or states.shape[1] != 9:
            raise ValueError(f"{label} joint_state9 requires state shape [T, 9], got {states.shape}")
        if actions.ndim != 2 or actions.shape != states.shape:
            raise ValueError(
                f"{label} joint_state9 requires action shape {states.shape}, got {actions.shape}"
            )
        return states, actions
    if mode != TRACIK_JOINT_STATE9_MODE:
        raise ValueError(f"Unsupported state_action_mode={mode!r}")

    label = str(path or "raw episode")
    required = ("state", "action", "replay_dof_pos", "tracik_command_q")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{label} cannot derive {mode}; missing fields: {missing}")
    raw_state = np.asarray(data["state"], dtype=np.float32)
    raw_action = np.asarray(data["action"], dtype=np.float32)
    replay_dof_pos = np.asarray(data["replay_dof_pos"], dtype=np.float32)
    tracik_command_q = np.asarray(data["tracik_command_q"], dtype=np.float32)
    if raw_state.ndim != 2 or raw_state.shape[1] < 2:
        raise ValueError(f"{label} has invalid state shape {raw_state.shape}; expected T x >=2")
    if raw_action.ndim != 2 or raw_action.shape[1] < 10:
        raise ValueError(f"{label} has invalid action shape {raw_action.shape}; expected T x >=10")
    if replay_dof_pos.ndim != 2 or replay_dof_pos.shape[1] < 19:
        raise ValueError(
            f"{label} has invalid replay_dof_pos shape {replay_dof_pos.shape}; "
            "expected the mapped A2W arm joints at columns 12:19"
        )
    if tracik_command_q.ndim != 2 or tracik_command_q.shape[1] != 6:
        raise ValueError(f"{label} has invalid tracik_command_q shape {tracik_command_q.shape}; expected T x 6")
    lengths = {raw_state.shape[0], raw_action.shape[0], replay_dof_pos.shape[0], tracik_command_q.shape[0]}
    if len(lengths) != 1:
        raise ValueError(
            f"{label} has misaligned frame counts: state={raw_state.shape[0]} action={raw_action.shape[0]} "
            f"replay_dof_pos={replay_dof_pos.shape[0]} tracik_command_q={tracik_command_q.shape[0]}"
        )
    states = np.concatenate([raw_state[:, :2], replay_dof_pos[:, 12:19]], axis=1).astype(np.float32)
    actions = np.concatenate(
        [raw_action[:, :2], tracik_command_q, raw_action[:, 9:10]],
        axis=1,
    ).astype(np.float32)
    if states.shape[1] != 9 or actions.shape[1] != 9:
        raise AssertionError(f"Derived joint state/action dimensions are {states.shape}/{actions.shape}, expected T x 9")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError(f"{label} produced non-finite joint state/action values")
    return states, actions


def detect_action_names(data, sidecar):
    if sidecar and sidecar.get("action") is not None:
        names = [str(item) for item in sidecar.get("action", [])]
        if names:
            return names
    if "action_names" in data.files:
        names = array_to_str_list(data["action_names"])
        if names:
            return names
    action_dim = int(data["action"].shape[-1])
    if action_dim == len(ACTION_NAMES):
        return list(ACTION_NAMES)
    return [f"action_{i}" for i in range(action_dim)]


def detect_action_frame(data, sidecar):
    for source in (data, sidecar or {}):
        for key in ("action_frame", "action_pose_frame", "target_pose_frame"):
            if isinstance(source, dict):
                if key in source:
                    return str(source[key]).lower()
            elif key in source.files:
                return scalar_str(source[key]).lower()
    return "world"


def detect_ikpush_state_version(data, sidecar):
    for source in (data, sidecar or {}):
        if isinstance(source, dict):
            if "ikpush_state_version" in source:
                return str(source["ikpush_state_version"])
        elif "ikpush_state_version" in source.files:
            return scalar_str(source["ikpush_state_version"])
    return "legacy"


def detect_controller_mode(data, sidecar):
    for source in (data, sidecar or {}):
        for key in ("door_dp_mode", "controller_mode"):
            if isinstance(source, dict):
                if key in source:
                    return str(source[key])
            elif key in source.files:
                return scalar_str(source[key])
    return "legacy"


def detect_raw_vision_mode(data, sidecar):
    if sidecar and sidecar.get("vision_mode") is not None:
        return normalize_vision_mode(sidecar["vision_mode"])
    if sidecar and sidecar.get("image_features") is not None:
        image_features = {str(key) for key in sidecar.get("image_features", [])}
        if image_features == {"wrist_masked_depth", "front_masked_depth"}:
            return "depth_only"
    if "vision_mode" in data.files:
        return normalize_vision_mode(scalar_str(data["vision_mode"]))
    if "wrist_rgb" in data.files or "front_rgb" in data.files:
        return "rgb"
    if (
        "wrist_masked_depth" in data.files
        and "front_masked_depth" in data.files
        and "wrist_handle_mask" not in data.files
        and "front_handle_mask" not in data.files
    ):
        return "depth_only"
    return "depth"


def require_fields(data, keys, path):
    missing = [key for key in keys if key not in data.files]
    if missing:
        raise KeyError(f"{path} is missing required fields for this vision mode: {missing}")


def validate_episode_metadata(path, data, sidecar, action_frame, ikpush_state_version, controller_mode):
    episode_action_frame = detect_action_frame(data, sidecar)
    if episode_action_frame != action_frame:
        raise ValueError(
            f"Episode {path} action_frame={episode_action_frame!r}, expected {action_frame!r}; "
            "do not mix world-frame and base-frame action datasets."
        )
    episode_state_version = detect_ikpush_state_version(data, sidecar)
    if episode_state_version != ikpush_state_version:
        raise ValueError(
            f"Episode {path} ikpush_state_version={episode_state_version!r}, expected {ikpush_state_version!r}; "
            "do not mix old and new ikpush state semantics."
        )
    episode_controller_mode = detect_controller_mode(data, sidecar)
    if episode_controller_mode != controller_mode:
        raise ValueError(
            f"Episode {path} door_dp_mode={episode_controller_mode!r}, expected {controller_mode!r}; "
            "do not mix ikpush and ikpull datasets."
        )


def fit_state_preprocess_from_episodes(
    files,
    sidecar,
    state_action_mode,
    keep_state_indices,
    state_names,
    action_frame,
    ikpush_state_version,
    controller_mode,
    lower_quantile,
    upper_quantile,
    eps,
):
    chunks = []
    total_frames = 0
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            validate_episode_metadata(path, data, sidecar, action_frame, ikpush_state_version, controller_mode)
            states, _ = state_action_arrays(data, state_action_mode, path)
            if keep_state_indices and states.shape[-1] <= max(keep_state_indices):
                raise ValueError(f"Episode {path} has state_dim={states.shape[-1]}, cannot apply selected state columns.")
            states = states[:, keep_state_indices]
            chunks.append(states)
            total_frames += int(states.shape[0])
    if not chunks:
        raise ValueError("Cannot fit state preprocessing without raw states.")
    config = fit_door_dp_state_preprocess(
        np.concatenate(chunks, axis=0),
        state_names,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        eps=eps,
    )
    constant_count = int(np.asarray(config.get("constant_mask", []), dtype=bool).sum())
    print(
        f"Fitted state_preprocess={config['version']} frames={total_frames} "
        f"state_dim={len(state_names)} constant_dims={constant_count} "
        f"quantiles=({lower_quantile}, {upper_quantile})",
        flush=True,
    )
    if config.get("angle_wrapped_features"):
        print(f"Angle-wrapped state features: {config['angle_wrapped_features']}", flush=True)
    if config.get("quaternion_groups"):
        print(f"Quaternion-normalized state groups: {config['quaternion_groups']}", flush=True)
    return config


def fit_action_preprocess_from_episodes(
    files,
    sidecar,
    state_action_mode,
    action_names,
    action_frame,
    ikpush_state_version,
    controller_mode,
    lower_quantile,
    upper_quantile,
    eps,
):
    chunks = []
    total_frames = 0
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            validate_episode_metadata(path, data, sidecar, action_frame, ikpush_state_version, controller_mode)
            _, actions = state_action_arrays(data, state_action_mode, path)
            chunks.append(actions)
            total_frames += int(actions.shape[0])
    if not chunks:
        raise ValueError("Cannot fit action preprocessing without raw actions.")
    config = fit_door_dp_action_preprocess(
        np.concatenate(chunks, axis=0),
        action_names,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        eps=eps,
    )
    constant_count = int(np.asarray(config.get("constant_mask", []), dtype=bool).sum())
    print(
        f"Fitted action_preprocess={config['version']} frames={total_frames} "
        f"action_dim={len(action_names)} constant_dims={constant_count} "
        f"quantiles=({lower_quantile}, {upper_quantile})",
        flush=True,
    )
    if config.get("quaternion_groups"):
        print(f"Quaternion-normalized action groups: {config['quaternion_groups']}", flush=True)
    return config


def load_episode_payload(
    path,
    sidecar,
    image_keys,
    state_action_mode,
    keep_state_indices,
    action_frame,
    ikpush_state_version,
    controller_mode,
    vision_mode,
    initial_task,
    keyframe_loss_weight_override=None,
    keyframe_loss_radius_override=None,
    load_handle_bbox=False,
    add_end_signal=False,
    end_signal_positive_phases=DEFAULT_END_SIGNAL_POSITIVE_PHASES,
    add_interaction_state=False,
    interaction_contact_min_consecutive_frames=3,
    interaction_handle_unlock_angle_deg=40.0,
    interaction_door_goal_angle_deg=90.0,
    point_cloud_views=None,
    point_cloud_config=None,
    point_cloud_camera_intrinsics=None,
    point_cloud_empty_depth_policy="previous",
):
    with np.load(path, allow_pickle=True) as data:
        validate_episode_metadata(path, data, sidecar, action_frame, ikpush_state_version, controller_mode)
        task = scalar_str(data["task"]) if "task" in data else initial_task
        states, actions = state_action_arrays(data, state_action_mode, path)
        if keep_state_indices and states.shape[-1] <= max(keep_state_indices):
            raise ValueError(f"Episode {path} has state_dim={states.shape[-1]}, cannot apply selected state columns.")
        states = states[:, keep_state_indices]
        if RECOVERY_INDICATOR_FEATURE in data.files:
            raw_is_recovery = np.asarray(data[RECOVERY_INDICATOR_FEATURE], dtype=np.float32)
            if raw_is_recovery.size == 1:
                is_recovery = np.full((states.shape[0], 1), float(raw_is_recovery.reshape(-1)[0]), dtype=np.float32)
            else:
                is_recovery = raw_is_recovery.reshape(-1, 1)
        else:
            is_recovery = np.zeros((states.shape[0], 1), dtype=np.float32)
        if not np.all(np.isfinite(is_recovery)) or np.any(is_recovery < 0.0) or np.any(is_recovery > 1.0):
            raise ValueError(
                f"Episode {path} has invalid {RECOVERY_INDICATOR_FEATURE}; expected finite values in [0, 1]."
            )
        if point_cloud_views is not None:
            requested_views = tuple(str(point_cloud_views).split(","))
            depth_key_by_view = (
                {"wrist": image_keys[0], "front": image_keys[1]}
                if vision_mode == "depth_only"
                else {"wrist": image_keys[1], "front": image_keys[3]}
            )
            require_fields(data, [depth_key_by_view[view] for view in requested_views], path)
            reference_depth = data[depth_key_by_view[requested_views[0]]].astype(np.uint8)
            empty_depth = np.zeros((reference_depth.shape[0], 1, 1), dtype=np.uint8)
            wrist_second = (
                data[depth_key_by_view["wrist"]].astype(np.uint8)
                if "wrist" in requested_views
                else empty_depth
            )
            front_second = (
                data[depth_key_by_view["front"]].astype(np.uint8)
                if "front" in requested_views
                else empty_depth
            )
            wrist_first = np.zeros_like(wrist_second)
            front_first = np.zeros_like(front_second)
        elif vision_mode == "depth_only":
            require_fields(data, image_keys, path)
            wrist_second = data[image_keys[0]].astype(np.uint8)
            front_second = data[image_keys[1]].astype(np.uint8)
            wrist_first = np.zeros_like(wrist_second)
            front_first = np.zeros_like(front_second)
        else:
            require_fields(data, image_keys[:2], path)
            wrist_first = data[image_keys[0]].astype(np.uint8)
            wrist_second = data[image_keys[1]].astype(np.uint8)
        if point_cloud_views is not None:
            pass
        elif vision_mode == "rgb":
            require_fields(data, image_keys[2:], path)
            front_first = data[image_keys[2]].astype(np.uint8)
            front_second = data[image_keys[3]].astype(np.uint8)
        elif vision_mode == "depth":
            front_first = data[image_keys[2]].astype(np.uint8) if image_keys[2] in data else np.zeros_like(wrist_first)
            front_second = data[image_keys[3]].astype(np.uint8) if image_keys[3] in data else np.zeros_like(wrist_second)
        subtasks = data["subtask_index"].astype(np.int64).reshape(-1)
        end_signal = None
        if add_end_signal:
            if RAW_END_SIGNAL_KEY in data.files:
                end_signal = np.asarray(data[RAW_END_SIGNAL_KEY], dtype=np.float32).reshape(-1, 1)
            else:
                raw_phase_names = (
                    array_to_str_list(data["phase_names"])
                    if "phase_names" in data.files
                    else list((sidecar or {}).get("phase_names", []))
                )
                end_signal = make_end_signal_from_phase_ids(
                    subtasks,
                    raw_phase_names,
                    positive_phases=end_signal_positive_phases,
                )
            if end_signal.shape[0] != states.shape[0]:
                raise ValueError(
                    f"Episode {path} has end_signal length {end_signal.shape[0]}, expected {states.shape[0]}."
                )
            if not np.all(np.isfinite(end_signal)) or np.any(end_signal < 0.0) or np.any(end_signal > 1.0):
                raise ValueError(f"Episode {path} has invalid end_signal; expected finite values in [0, 1].")
        interaction_state = None
        door_asset_name = scalar_str(data["door_asset_name"]) if "door_asset_name" in data.files else "unknown"
        if add_interaction_state:
            required_interaction = ["gripper_handle_contact_both", "replay_door_dof_pos"]
            missing_interaction = [key for key in required_interaction if key not in data.files]
            if missing_interaction:
                raise ValueError(
                    f"Episode {path} cannot generate interaction-state targets; missing {missing_interaction}. "
                    "Re-record with gripper-handle contact and replay snapshots enabled."
                )
            door_dof_pos = np.asarray(data["replay_door_dof_pos"], dtype=np.float32)
            if door_dof_pos.ndim != 2 or door_dof_pos.shape[1] < 2:
                raise ValueError(
                    f"Episode {path} replay_door_dof_pos must have shape (T, >=2), got {door_dof_pos.shape}."
                )

            def source_scalar(key, fallback=None):
                if key in data.files:
                    return float(np.asarray(data[key]).reshape(-1)[0])
                if sidecar and key in sidecar:
                    return float(sidecar[key])
                return fallback

            handle_closed = source_scalar("handle_closed_angle")
            door_closed = source_scalar("door_closed_angle")
            if handle_closed is None or door_closed is None:
                initial_handle = float(door_dof_pos[0, 1])
                initial_door = float(door_dof_pos[0, 0])
                recovery_episode = bool(np.any(is_recovery > 0.5)) or "recovery_source_failure_rollout" in data.files
                if recovery_episode and max(abs(initial_handle), abs(initial_door)) > math.radians(5.0):
                    raise ValueError(
                        f"Recovery episode {path} starts away from the closed state but has no closed-angle metadata. "
                        "Refusing to use its branch state as the progress origin."
                    )
                handle_closed = initial_handle if handle_closed is None else handle_closed
                door_closed = initial_door if door_closed is None else door_closed
            handle_unlock_delta = source_scalar(
                "handle_unlock_threshold",
                math.radians(float(interaction_handle_unlock_angle_deg)),
            )
            door_goal_delta = source_scalar(
                "door_progress_goal_angle",
                math.radians(float(interaction_door_goal_angle_deg)),
            )
            interaction_state = make_interaction_state_targets(
                data["gripper_handle_contact_both"],
                door_dof_pos,
                contact_min_consecutive_frames=interaction_contact_min_consecutive_frames,
                handle_closed_angle=handle_closed,
                door_closed_angle=door_closed,
                handle_unlock_delta_rad=handle_unlock_delta,
                door_goal_delta_rad=door_goal_delta,
            )
        has_front_pose = RAW_FRONT_CAMERA_POSE_KEY in data.files
        has_wrist_pose = RAW_WRIST_CAMERA_POSE_KEY in data.files
        if has_front_pose != has_wrist_pose and point_cloud_views is None:
            raise ValueError(
                f"Episode {path} has incomplete camera pose fields: "
                f"{RAW_FRONT_CAMERA_POSE_KEY}={has_front_pose}, {RAW_WRIST_CAMERA_POSE_KEY}={has_wrist_pose}."
            )
        if has_front_pose:
            front_camera_pose_base = data[RAW_FRONT_CAMERA_POSE_KEY].astype(np.float32).reshape(-1, 7)
        else:
            front_camera_pose_base = None
        if has_wrist_pose:
            wrist_camera_pose_base = data[RAW_WRIST_CAMERA_POSE_KEY].astype(np.float32).reshape(-1, 7)
        else:
            wrist_camera_pose_base = None
        point_cloud = None
        if point_cloud_views is not None:
            required_poses = {
                "front": front_camera_pose_base,
                "wrist": wrist_camera_pose_base,
            }
            missing_pose_views = [
                view for view in str(point_cloud_views).split(",") if required_poses[view] is None
            ]
            if missing_pose_views:
                raise ValueError(
                    f"Episode {path} cannot generate {point_cloud_views} point clouds; "
                    f"missing per-frame poses for {missing_pose_views}."
                )
            if point_cloud_camera_intrinsics is None:
                raise ValueError(f"Episode {path} has no front/wrist camera intrinsics metadata.")
            point_cloud = make_episode_point_clouds(
                path=path,
                views=point_cloud_views,
                wrist_depth=wrist_second,
                front_depth=front_second,
                wrist_camera_pose_base=wrist_camera_pose_base,
                front_camera_pose_base=front_camera_pose_base,
                camera_intrinsics=point_cloud_camera_intrinsics,
                config=point_cloud_config,
                empty_depth_policy=point_cloud_empty_depth_policy,
            )
        has_handle_bbox = bool(load_handle_bbox) and (
            RAW_FRONT_HANDLE_BBOX_KEY in data.files or RAW_WRIST_HANDLE_BBOX_KEY in data.files
        )
        if has_handle_bbox:
            required_bbox = [
                RAW_FRONT_HANDLE_BBOX_KEY,
                RAW_FRONT_HANDLE_BBOX_VALID_KEY,
                RAW_WRIST_HANDLE_BBOX_KEY,
                RAW_WRIST_HANDLE_BBOX_VALID_KEY,
            ]
            missing_bbox = [key for key in required_bbox if key not in data.files]
            if missing_bbox:
                raise ValueError(f"Episode {path} has incomplete handle bbox fields; missing {missing_bbox}.")
            front_handle_bbox_xyxy = _normalize_handle_bbox_array(
                data[RAW_FRONT_HANDLE_BBOX_KEY],
                states.shape[0],
                RAW_FRONT_HANDLE_BBOX_KEY,
                path,
            )
            front_handle_bbox_valid = _normalize_handle_valid_array(
                data[RAW_FRONT_HANDLE_BBOX_VALID_KEY],
                states.shape[0],
                RAW_FRONT_HANDLE_BBOX_VALID_KEY,
                path,
            )
            wrist_handle_bbox_xyxy = _normalize_handle_bbox_array(
                data[RAW_WRIST_HANDLE_BBOX_KEY],
                states.shape[0],
                RAW_WRIST_HANDLE_BBOX_KEY,
                path,
            )
            wrist_handle_bbox_valid = _normalize_handle_valid_array(
                data[RAW_WRIST_HANDLE_BBOX_VALID_KEY],
                states.shape[0],
                RAW_WRIST_HANDLE_BBOX_VALID_KEY,
                path,
            )
        else:
            front_handle_bbox_xyxy = None
            front_handle_bbox_valid = None
            wrist_handle_bbox_xyxy = None
            wrist_handle_bbox_valid = None
        override_keyframe_weight = (
            keyframe_loss_weight_override is not None or keyframe_loss_radius_override is not None
        )
        if RAW_ACTION_LOSS_WEIGHT_KEY in data.files and not override_keyframe_weight:
            action_loss_weight = data[RAW_ACTION_LOSS_WEIGHT_KEY].astype(np.float32).reshape(-1, 1)
        else:
            if "keyframe_indices" in data.files:
                keyframe_indices = data["keyframe_indices"].astype(np.int64).reshape(-1)
            else:
                keyframe_indices, _, _ = extract_motion_keyframes_from_raw_arrays(data)
            stored_keyframe_loss_weight = (
                float(data["keyframe_loss_weight"])
                if "keyframe_loss_weight" in data.files
                else DEFAULT_KEYFRAME_LOSS_WEIGHT
            )
            stored_keyframe_loss_radius = (
                int(data["keyframe_loss_radius"])
                if "keyframe_loss_radius" in data.files
                else DEFAULT_KEYFRAME_LOSS_RADIUS
            )
            keyframe_loss_weight = (
                stored_keyframe_loss_weight
                if keyframe_loss_weight_override is None
                else float(keyframe_loss_weight_override)
            )
            keyframe_loss_radius = (
                stored_keyframe_loss_radius
                if keyframe_loss_radius_override is None
                else int(keyframe_loss_radius_override)
            )
            keyframe_loss_enabled = (
                bool(data["keyframe_loss_enabled"]) if "keyframe_loss_enabled" in data.files else True
            )
            action_loss_weight = make_keyframe_action_loss_weight(
                states.shape[0],
                keyframe_indices,
                weight=keyframe_loss_weight,
                radius=keyframe_loss_radius,
                enabled=keyframe_loss_enabled,
            ).reshape(-1, 1)
    n = states.shape[0]
    if not (
        actions.shape[0]
        == wrist_first.shape[0]
        == wrist_second.shape[0]
        == front_first.shape[0]
        == front_second.shape[0]
        == subtasks.shape[0]
        == action_loss_weight.shape[0]
        == is_recovery.shape[0]
        == n
    ):
        raise ValueError(f"Episode {path} has inconsistent lengths.")
    for camera_name, camera_pose in (
        ("front", front_camera_pose_base),
        ("wrist", wrist_camera_pose_base),
    ):
        if camera_pose is not None and camera_pose.shape[0] != n:
            raise ValueError(
                f"Episode {path} {camera_name} camera pose length is inconsistent with frame count."
            )
    if end_signal is not None and end_signal.shape[0] != n:
        raise ValueError(f"Episode {path} has end_signal length inconsistent with frame count.")
    if interaction_state is not None and any(value.shape[0] != n for value in interaction_state.values()):
        raise ValueError(f"Episode {path} has interaction-state length inconsistent with frame count.")
    if point_cloud is not None and point_cloud.shape != (n, point_cloud_config.num_points, 3):
        raise ValueError(
            f"Episode {path} has invalid point-cloud shape {point_cloud.shape}; expected "
            f"{(n, point_cloud_config.num_points, 3)}."
        )
    if front_handle_bbox_xyxy is not None and (
        front_handle_bbox_xyxy.shape[0] != n
        or front_handle_bbox_valid.shape[0] != n
        or wrist_handle_bbox_xyxy.shape[0] != n
        or wrist_handle_bbox_valid.shape[0] != n
    ):
        raise ValueError(f"Episode {path} has handle bbox length inconsistent with frame count.")
    return {
        "path_name": Path(path).name,
        "task": task,
        "states": states,
        "actions": actions,
        "wrist_first": wrist_first,
        "wrist_second": wrist_second,
        "front_first": front_first,
        "front_second": front_second,
        "subtasks": subtasks,
        "action_loss_weight": action_loss_weight,
        "is_recovery": is_recovery,
        "end_signal": end_signal,
        "interaction_state": interaction_state,
        "point_cloud": point_cloud,
        "door_asset_name": door_asset_name,
        "front_camera_pose_base": front_camera_pose_base,
        "wrist_camera_pose_base": wrist_camera_pose_base,
        "has_camera_pose": front_camera_pose_base is not None and wrist_camera_pose_base is not None,
        "has_handle_bbox": front_handle_bbox_xyxy is not None,
        "front_handle_bbox_xyxy": front_handle_bbox_xyxy,
        "front_handle_bbox_valid": front_handle_bbox_valid,
        "wrist_handle_bbox_xyxy": wrist_handle_bbox_xyxy,
        "wrist_handle_bbox_valid": wrist_handle_bbox_valid,
        "n": n,
    }


def iter_episode_payloads(files, num_workers, **kwargs):
    if num_workers <= 1:
        for ep_idx, path in enumerate(files):
            yield ep_idx, load_episode_payload(path, **kwargs)
        return

    # Keep output deterministic: at most num_workers episodes are prepared ahead,
    # and payloads are yielded in sorted filename order for the single writer.
    def submit_next(executor, iterator, pending):
        try:
            ep_idx, path = next(iterator)
        except StopIteration:
            return False
        pending.append((ep_idx, executor.submit(load_episode_payload, path, **kwargs)))
        return True

    iterator = iter(enumerate(files))
    pending = deque()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(min(num_workers, len(files))):
            submit_next(executor, iterator, pending)
        while pending:
            ep_idx, future = pending.popleft()
            payload = future.result()
            yield ep_idx, payload
            submit_next(executor, iterator, pending)


def main():
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    point_cloud_enabled = args.point_cloud_views is not None
    point_cloud_config = None
    point_cloud_camera_intrinsics = None
    if point_cloud_enabled:
        point_cloud_config = FrontDepthPointCloudConfig(
            num_points=int(args.point_cloud_num_points),
            candidate_rows=int(args.point_cloud_candidate_rows),
            candidate_cols=int(args.point_cloud_candidate_cols),
            workspace_min=parse_xyz_csv(args.point_cloud_workspace_min, "--point_cloud_workspace_min"),
            workspace_max=parse_xyz_csv(args.point_cloud_workspace_max, "--point_cloud_workspace_max"),
            near_clip_m=0.20,
            far_clip_m=1.50,
            point_frame="robot_base",
        )
        point_cloud_config.validate()
        if args.point_cloud_views == "front,wrist" and point_cloud_config.num_points % 2:
            raise ValueError("--point_cloud_num_points must be even for front,wrist fusion.")
    if args.keyframe_loss_weight is not None and args.keyframe_loss_weight <= 0.0:
        raise ValueError("--keyframe_loss_weight must be > 0")
    if args.keyframe_loss_radius is not None and args.keyframe_loss_radius < 0:
        raise ValueError("--keyframe_loss_radius must be >= 0")
    raw_roots = [Path(args.raw_root), *(Path(value) for value in args.additional_raw_root)]
    files = [path for root in raw_roots for path in episode_files(root)]
    raw_root = raw_roots[0]
    sidecar = load_sidecar(raw_root)
    first = np.load(files[0], allow_pickle=True)
    validate_calibrated_camera_metadata(sidecar, first, files[0])
    if args.state_action_mode in JOINT_STATE9_MODES:
        state_names = list(TRACIK_JOINT_STATE9_NAMES)
        action_names = list(TRACIK_JOINT_ACTION9_NAMES)
    else:
        if sidecar and "state" in sidecar:
            state_names = list(sidecar["state"])
        elif "state_feature_names" in first:
            state_names = [str(x) for x in first["state_feature_names"].tolist()]
        else:
            state_names = [f"state_{i}" for i in range(first["state"].shape[-1])]
        action_names = detect_action_names(first, sidecar)
    first_states, first_actions = state_action_arrays(first, args.state_action_mode, files[0])
    has_front_camera_pose = RAW_FRONT_CAMERA_POSE_KEY in first.files
    has_wrist_camera_pose = RAW_WRIST_CAMERA_POSE_KEY in first.files
    has_camera_pose = has_front_camera_pose and has_wrist_camera_pose
    if not point_cloud_enabled and has_front_camera_pose != has_wrist_camera_pose:
        raise ValueError(
            "Raw data has incomplete camera pose fields in the first episode: "
            f"{RAW_FRONT_CAMERA_POSE_KEY}={RAW_FRONT_CAMERA_POSE_KEY in first.files}, "
            f"{RAW_WRIST_CAMERA_POSE_KEY}={RAW_WRIST_CAMERA_POSE_KEY in first.files}."
        )
    if point_cloud_enabled:
        missing_pose_views = [
            view
            for view in args.point_cloud_views.split(",")
            if not {"front": has_front_camera_pose, "wrist": has_wrist_camera_pose}[view]
        ]
        if missing_pose_views:
            raise ValueError(
                f"Point-cloud conversion for {args.point_cloud_views!r} is missing raw camera poses "
                f"for {missing_pose_views}."
            )
        point_cloud_camera_intrinsics = (sidecar or {}).get("camera_intrinsics")
        if not point_cloud_camera_intrinsics and "camera_intrinsics" in first.files:
            point_cloud_camera_intrinsics = np.asarray(first["camera_intrinsics"]).item()
        if not point_cloud_camera_intrinsics:
            raise ValueError("Point-cloud conversion requires camera_intrinsics metadata for the requested view(s).")
        requested = tuple(args.point_cloud_views.split(","))
        for view in requested:
            intrinsics_dict(point_cloud_camera_intrinsics, camera=view)
    has_handle_bbox = bool(args.add_handle_latent) and (
        RAW_FRONT_HANDLE_BBOX_KEY in first.files or RAW_WRIST_HANDLE_BBOX_KEY in first.files
    )
    if args.add_handle_latent:
        required_bbox = [
            RAW_FRONT_HANDLE_BBOX_KEY,
            RAW_FRONT_HANDLE_BBOX_VALID_KEY,
            RAW_WRIST_HANDLE_BBOX_KEY,
            RAW_WRIST_HANDLE_BBOX_VALID_KEY,
        ]
        missing_bbox = [key for key in required_bbox if key not in first.files]
        if missing_bbox:
            raise ValueError(
                "--add_handle_latent requires raw episodes recorded with --record_handle_bbox. "
                f"The first episode is missing {missing_bbox}."
            )
    if args.handle_latent_crop_size <= 0:
        raise ValueError("--handle_latent_crop_size must be positive")
    if args.handle_latent_bbox_margin < 0.0:
        raise ValueError("--handle_latent_bbox_margin must be non-negative")
    if args.handle_bbox_min_area < 0.0 or args.handle_bbox_min_size < 0.0:
        raise ValueError("--handle_bbox_min_area and --handle_bbox_min_size must be non-negative")
    if args.handle_latent_batch_size <= 0:
        raise ValueError("--handle_latent_batch_size must be positive")
    if first_actions.shape[-1] != len(action_names):
        raise ValueError(
            f"Converted action_dim={first_actions.shape[-1]} does not match action_names={len(action_names)}: "
            f"{action_names}"
        )
    keep_state_indices = list(range(len(state_names)))
    dropped_state_names = []
    if not args.keep_phase_state:
        keep_state_indices = [i for i, name in enumerate(state_names) if not str(name).startswith("phase_")]
        dropped_state_names = [name for i, name in enumerate(state_names) if i not in keep_state_indices]
        state_names = [state_names[i] for i in keep_state_indices]
        if dropped_state_names:
            print(
                f"Dropping {len(dropped_state_names)} legacy phase state columns: {dropped_state_names}",
                flush=True,
            )
    raw_fps = int(first["fps"]) if "fps" in first else 50
    fps = int(args.fps or (sidecar.get("fps") if sidecar else raw_fps))
    depth_only_explicit = "--depth_only" in sys.argv[1:]
    if args.rgb and depth_only_explicit:
        raise ValueError("--rgb and --depth_only are mutually exclusive.")
    if args.rgb:
        args.depth_only = False
    vision_mode = "rgb" if args.rgb else ("depth_only" if args.depth_only else "depth")
    if point_cloud_enabled and vision_mode == "rgb":
        raise ValueError(
            "Point-cloud conversion requires depth observations; RGB-only raw data has no metric depth field."
        )
    raw_vision_mode = detect_raw_vision_mode(first, sidecar)
    raw_action_frame = detect_action_frame(first, sidecar)
    raw_ikpush_state_version = detect_ikpush_state_version(first, sidecar)
    action_frame = "joint_command" if args.state_action_mode in JOINT_STATE9_MODES else raw_action_frame
    ikpush_state_version = (
        "a2w_last_command_joint_state9"
        if args.state_action_mode in JOINT_STATE9_MODES
        else raw_ikpush_state_version
    )
    controller_mode = detect_controller_mode(first, sidecar)
    if raw_action_frame not in ("world", "base", "robot_base_full", "base_full", "arm_base", "robot_base", "true_base"):
        raise ValueError(
            f"Unsupported raw action_frame={raw_action_frame!r}; "
            "expected 'world', 'base', or 'robot_base_full'."
        )
    if raw_vision_mode != vision_mode:
        raise ValueError(
            f"Raw data vision_mode={raw_vision_mode!r}, but converter was run with "
            f"{'--rgb' if args.rgb else ('--depth_only' if args.depth_only else 'depth mode')}. "
            "Use the matching vision flag for the raw data."
        )
    image_keys = raw_image_keys_for_vision_mode(vision_mode)
    out_dir = Path(args.root) / args.repo_id
    existing_sidecar = load_sidecar(out_dir) if out_dir.exists() else None
    state_preprocess_config = None
    action_preprocess_config = None
    if existing_sidecar and not args.overwrite:
        existing_point_cloud_enabled = bool(existing_sidecar.get("point_cloud_conditioning", False))
        if existing_point_cloud_enabled != point_cloud_enabled:
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has point_cloud_conditioning="
                f"{existing_point_cloud_enabled}, but this conversion requested {point_cloud_enabled}; "
                "use --overwrite or a new --repo_id."
            )
        if point_cloud_enabled:
            expected_point_cloud_metadata = {
                "point_cloud_views": str(args.point_cloud_views),
                "point_cloud_num_points": int(args.point_cloud_num_points),
                "point_cloud_candidate_rows": int(args.point_cloud_candidate_rows),
                "point_cloud_candidate_cols": int(args.point_cloud_candidate_cols),
                "point_cloud_workspace_min": list(point_cloud_config.workspace_min),
                "point_cloud_workspace_max": list(point_cloud_config.workspace_max),
                "point_cloud_empty_depth_policy": str(args.point_cloud_empty_depth_policy),
                "point_cloud_frame": "robot_base",
            }
            mismatched = {
                key: (existing_sidecar.get(key), value)
                for key, value in expected_point_cloud_metadata.items()
                if existing_sidecar.get(key) != value
            }
            if mismatched:
                raise ValueError(
                    f"Existing point-cloud dataset at {out_dir} has incompatible geometry metadata: "
                    f"{mismatched}. Use --overwrite or a new --repo_id."
                )
        existing_vision_mode = normalize_vision_mode(existing_sidecar.get("vision_mode", "depth"))
        if existing_vision_mode != vision_mode:
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has vision_mode={existing_vision_mode!r}; "
                "use a different --repo_id or pass --overwrite."
            )
        existing_action_frame = str(
            existing_sidecar.get("action_frame", existing_sidecar.get("action_pose_frame", "world"))
        ).lower()
        if existing_action_frame != action_frame:
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has action_frame={existing_action_frame!r}, "
                f"but raw data has action_frame={action_frame!r}; use a different --repo_id or pass --overwrite."
            )
        existing_state_version = str(existing_sidecar.get("ikpush_state_version", "legacy"))
        if existing_state_version != ikpush_state_version:
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has ikpush_state_version={existing_state_version!r}, "
                f"but raw data has {ikpush_state_version!r}; use a different --repo_id or pass --overwrite."
            )
        existing_controller_mode = str(existing_sidecar.get("door_dp_mode", existing_sidecar.get("controller_mode", "legacy")))
        if existing_controller_mode != controller_mode:
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has door_dp_mode={existing_controller_mode!r}, "
                f"but raw data has {controller_mode!r}; use a different --repo_id or pass --overwrite."
            )
        existing_state_names = list(existing_sidecar.get("state", []))
        if existing_state_names and existing_state_names != state_names:
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has different state feature names; "
                "use --overwrite or a new --repo_id."
            )
        existing_state_preprocess = existing_sidecar.get("state_preprocess")
        existing_applied = bool(existing_state_preprocess and existing_state_preprocess.get("applied", False))
        requested_applied = args.state_preprocess != "none"
        if existing_applied != requested_applied:
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has state_preprocess applied={existing_applied}, "
                f"but this conversion requested applied={requested_applied}; use --overwrite or a new --repo_id."
            )
        if existing_applied:
            state_preprocess_config = existing_state_preprocess
            print(
                f"Reusing existing state_preprocess={state_preprocess_config.get('version')} "
                f"from {out_dir / 'door_dp_feature_names.json'}",
                flush=True,
            )
        existing_action_names = list(existing_sidecar.get("action", []))
        if existing_action_names and existing_action_names != action_names:
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has different action names; "
                "use --overwrite or a new --repo_id."
            )
        existing_action_preprocess = existing_sidecar.get("action_preprocess")
        existing_action_applied = bool(existing_action_preprocess and existing_action_preprocess.get("applied", False))
        requested_action_applied = args.action_preprocess != "none"
        if existing_action_applied != requested_action_applied:
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has action_preprocess applied={existing_action_applied}, "
                f"but this conversion requested applied={requested_action_applied}; use --overwrite or a new --repo_id."
            )
        if existing_action_applied:
            action_preprocess_config = existing_action_preprocess
            print(
                f"Reusing existing action_preprocess={action_preprocess_config.get('version')} "
                f"from {out_dir / 'door_dp_feature_names.json'}",
                flush=True,
            )
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    inherited_metadata = {}
    if sidecar:
        for key in DATASET_METADATA_KEYS:
            if key in sidecar:
                inherited_metadata[key] = sidecar[key]
    if has_camera_pose:
        inherited_metadata.setdefault("camera_pose_features", [FRONT_CAMERA_POSE_FEATURE, WRIST_CAMERA_POSE_FEATURE])
        inherited_metadata.setdefault("camera_pose_frame", "robot_base")
        inherited_metadata.setdefault("camera_pose_convention", "optical_frame")
    if has_handle_bbox:
        inherited_metadata.setdefault(
            "handle_bbox_features",
            [
                RAW_FRONT_HANDLE_BBOX_KEY,
                RAW_FRONT_HANDLE_BBOX_VALID_KEY,
                RAW_WRIST_HANDLE_BBOX_KEY,
                RAW_WRIST_HANDLE_BBOX_VALID_KEY,
            ],
        )
        inherited_metadata.setdefault("handle_bbox_convention", "xyxy_half_open_pixels")
    converted_keyframe_loss_weight = (
        float(args.keyframe_loss_weight)
        if args.keyframe_loss_weight is not None
        else float(inherited_metadata.get("keyframe_loss_weight", DEFAULT_KEYFRAME_LOSS_WEIGHT))
    )
    converted_keyframe_loss_radius = (
        int(args.keyframe_loss_radius)
        if args.keyframe_loss_radius is not None
        else int(inherited_metadata.get("keyframe_loss_radius", DEFAULT_KEYFRAME_LOSS_RADIUS))
    )

    if state_preprocess_config is None:
        if args.state_preprocess == "robust_quantile":
            state_preprocess_config = fit_state_preprocess_from_episodes(
                files,
                sidecar,
                args.state_action_mode,
                keep_state_indices,
                state_names,
                raw_action_frame,
                raw_ikpush_state_version,
                controller_mode,
                args.state_quantile_low,
                args.state_quantile_high,
                args.state_preprocess_eps,
            )
        else:
            state_preprocess_config = {"applied": False, "version": "none", "mode": "identity"}
    if action_preprocess_config is None:
        if args.action_preprocess == "robust_quantile":
            action_preprocess_config = fit_action_preprocess_from_episodes(
                files,
                sidecar,
                args.state_action_mode,
                action_names,
                raw_action_frame,
                raw_ikpush_state_version,
                controller_mode,
                args.action_quantile_low,
                args.action_quantile_high,
                args.action_preprocess_eps,
            )
        else:
            action_preprocess_config = {"applied": False, "version": "none", "mode": "identity"}

    initial_task = scalar_str(first["task"]) if "task" in first else "door open"
    converted_state_normalized = bool(state_preprocess_config.get("applied", False))
    state_sanitize_config = make_door_dp_sanitize_config(state_names, eps=args.near_zero_rate_eps)
    action_sanitize_config = make_door_dp_sanitize_config(action_names, eps=args.near_zero_rate_eps)
    handle_latent_teacher = None
    if args.add_handle_latent:
        print(
            f"Loading frozen handle-latent teacher {args.handle_latent_teacher_model!r} "
            f"for {len(files)} raw episodes...",
            flush=True,
        )
        handle_latent_teacher = DINOv2HandleLatentTeacher(args.handle_latent_teacher_model)
    recorder = DoorDPLeRobotRecorder(
        root=args.root,
        repo_id=args.repo_id,
        fps=fps,
        state_feature_names=state_names,
        action_feature_names=action_names,
        task=initial_task,
        resume=not args.overwrite,
        vision_mode=vision_mode,
        image_storage=args.image_storage,
        video_codec=args.video_codec,
        include_action_loss_weight=True,
        include_recovery_indicator=bool(args.include_recovery_indicator),
        include_camera_pose=has_camera_pose and not point_cloud_enabled,
        include_handle_latent=bool(args.add_handle_latent),
        include_end_signal=bool(args.add_end_signal),
        include_interaction_state=bool(args.add_interaction_state),
        include_images=not point_cloud_enabled,
        include_point_cloud=point_cloud_enabled,
        point_cloud_num_points=int(args.point_cloud_num_points),
        metadata={
            **inherited_metadata,
            "action_frame": action_frame,
            "action_pose_frame": action_frame,
            "target_pose_frame": action_frame,
            "ikpush_state_version": ikpush_state_version,
            "state_action_mode": args.state_action_mode,
            "action_source": (
                "base_command_plus_tracik_smoothed_joint_command"
                if args.state_action_mode == TRACIK_JOINT_STATE9_MODE
                else (
                    "base_command_plus_a2w_z1_joint_targets"
                    if args.state_action_mode == DIRECT_JOINT_STATE9_MODE
                    else inherited_metadata.get("action_source", "raw")
                )
            ),
            "door_dp_mode": controller_mode,
            "controller_mode": controller_mode,
            "image_storage": args.image_storage,
            "video_codec": args.video_codec,
            **(
                {
                    "point_cloud_conditioning": True,
                    "point_cloud_feature": POINT_CLOUD_FEATURE,
                    "point_cloud_views": args.point_cloud_views,
                    "point_cloud_num_points": int(args.point_cloud_num_points),
                    "point_cloud_candidate_rows": int(args.point_cloud_candidate_rows),
                    "point_cloud_candidate_cols": int(args.point_cloud_candidate_cols),
                    "point_cloud_workspace_min": list(point_cloud_config.workspace_min),
                    "point_cloud_workspace_max": list(point_cloud_config.workspace_max),
                    "point_cloud_near_clip_m": float(point_cloud_config.near_clip_m),
                    "point_cloud_far_clip_m": float(point_cloud_config.far_clip_m),
                    "point_cloud_empty_depth_policy": args.point_cloud_empty_depth_policy,
                    "point_cloud_storage": args.point_cloud_storage,
                    "point_cloud_frame": "robot_base",
                    **{
                        f"{view}_camera_intrinsics": intrinsics_dict(
                            point_cloud_camera_intrinsics, camera=view
                        )
                        for view in args.point_cloud_views.split(",")
                    },
                }
                if point_cloud_enabled
                else {}
            ),
            "state_sanitize": state_sanitize_config,
            "action_sanitize": action_sanitize_config,
            "state_preprocess": state_preprocess_config,
            "action_preprocess": action_preprocess_config,
            "state_normalized": converted_state_normalized,
            "action_loss_weight_feature": ACTION_LOSS_WEIGHT_FEATURE,
            **(
                {
                    "end_signal_enabled": True,
                    "end_signal_feature": END_SIGNAL_FEATURE,
                    "end_signal_positive_phases": parse_csv_names(args.end_signal_positive_phases),
                    "end_signal_version": "phase_dense_v1",
                }
                if args.add_end_signal
                else {}
            ),
            **(
                {
                    "interaction_state_features": list(INTERACTION_STATE_FEATURES),
                    "interaction_state_version": "causal_cummax_v1",
                    "interaction_contact_min_consecutive_frames": int(
                        args.interaction_contact_min_consecutive_frames
                    ),
                    "interaction_handle_unlock_angle_deg": float(args.interaction_handle_unlock_angle_deg),
                    "interaction_door_goal_angle_deg": float(args.interaction_door_goal_angle_deg),
                }
                if args.add_interaction_state
                else {}
            ),
            "keyframe_loss_weight": converted_keyframe_loss_weight,
            "keyframe_loss_radius": converted_keyframe_loss_radius,
            **(
                {
                    "handle_latent_features": [
                        FRONT_HANDLE_LATENT_FEATURE,
                        FRONT_HANDLE_LATENT_VALID_FEATURE,
                        WRIST_HANDLE_LATENT_FEATURE,
                        WRIST_HANDLE_LATENT_VALID_FEATURE,
                    ],
                    "handle_latent_teacher_model": args.handle_latent_teacher_model,
                    "handle_latent_crop_size": int(args.handle_latent_crop_size),
                    "handle_latent_bbox_margin": float(args.handle_latent_bbox_margin),
                    "handle_bbox_min_area": float(args.handle_bbox_min_area),
                    "handle_bbox_min_size": float(args.handle_bbox_min_size),
                }
                if args.add_handle_latent
                else {}
            ),
        },
    )
    payloads = iter_episode_payloads(
        files,
        args.num_workers,
        sidecar=sidecar,
        image_keys=image_keys,
        state_action_mode=args.state_action_mode,
        keep_state_indices=keep_state_indices,
        action_frame=raw_action_frame,
        ikpush_state_version=raw_ikpush_state_version,
        controller_mode=controller_mode,
        vision_mode=vision_mode,
        initial_task=initial_task,
        keyframe_loss_weight_override=args.keyframe_loss_weight,
        keyframe_loss_radius_override=args.keyframe_loss_radius,
        load_handle_bbox=bool(args.add_handle_latent),
        add_end_signal=bool(args.add_end_signal),
        end_signal_positive_phases=parse_csv_names(args.end_signal_positive_phases),
        add_interaction_state=bool(args.add_interaction_state),
        interaction_contact_min_consecutive_frames=int(args.interaction_contact_min_consecutive_frames),
        interaction_handle_unlock_angle_deg=float(args.interaction_handle_unlock_angle_deg),
        interaction_door_goal_angle_deg=float(args.interaction_door_goal_angle_deg),
        point_cloud_views=args.point_cloud_views,
        point_cloud_config=point_cloud_config,
        point_cloud_camera_intrinsics=point_cloud_camera_intrinsics,
        point_cloud_empty_depth_policy=args.point_cloud_empty_depth_policy,
    )
    end_positive_frames = 0
    end_total_frames = 0
    end_zero_positive_episodes = 0
    interaction_stats = {}
    for ep_idx, payload in payloads:
        task = payload["task"]
        recorder.task = task
        states = payload["states"]
        states = sanitize_door_dp_state(states, state_names=state_names, eps=args.near_zero_rate_eps)
        states = apply_door_dp_state_preprocess(states, state_names=state_names, config=state_preprocess_config)
        actions = payload["actions"]
        actions = sanitize_door_dp_action(actions, action_names=action_names, eps=args.near_zero_rate_eps)
        actions = apply_door_dp_action_preprocess(actions, action_names=action_names, config=action_preprocess_config)
        wrist_first = payload["wrist_first"]
        wrist_second = payload["wrist_second"]
        front_first = payload["front_first"]
        front_second = payload["front_second"]
        subtasks = payload["subtasks"]
        action_loss_weight = payload["action_loss_weight"]
        is_recovery = payload["is_recovery"]
        end_signal = payload.get("end_signal")
        interaction_state = payload.get("interaction_state")
        point_cloud = payload.get("point_cloud")
        if end_signal is not None:
            episode_positive = int(np.count_nonzero(end_signal > 0.5))
            end_positive_frames += episode_positive
            end_total_frames += int(end_signal.shape[0])
            end_zero_positive_episodes += int(episode_positive == 0)
            print(
                f"End signal {payload['path_name']}: positive={episode_positive}/{end_signal.shape[0]}",
                flush=True,
            )
        if interaction_state is not None:
            door_name = str(payload.get("door_asset_name", "unknown"))
            door_stats = interaction_stats.setdefault(
                door_name,
                {
                    key: {"sum": 0.0, "count": 0, "min": 1.0, "max": 0.0, "first_one": []}
                    for key in INTERACTION_STATE_FEATURES
                },
            )
            episode_summary = []
            for key in INTERACTION_STATE_FEATURES:
                values = np.asarray(interaction_state[key], dtype=np.float32).reshape(-1)
                stat = door_stats[key]
                stat["sum"] += float(values.sum())
                stat["count"] += int(values.size)
                stat["min"] = min(float(stat["min"]), float(values.min()))
                stat["max"] = max(float(stat["max"]), float(values.max()))
                ones = np.flatnonzero(values >= 1.0 - 1.0e-6)
                first_one = None if ones.size == 0 else int(ones[0])
                if first_one is not None:
                    stat["first_one"].append(first_one)
                episode_summary.append(
                    f"{key.rsplit('.', 1)[-1]}:mean={float(values.mean()):.3f},first1={first_one}"
                )
            print(
                f"Interaction state {payload['path_name']} door={door_name}: " + " ".join(episode_summary),
                flush=True,
            )
        if not point_cloud_enabled and bool(payload.get("has_camera_pose", False)) != bool(has_camera_pose):
            raise ValueError(
                f"Episode {payload['path_name']} camera-pose presence does not match the first episode. "
                "Do not mix Plücker-ready and legacy raw episodes in one conversion."
            )
        if args.add_handle_latent and bool(payload.get("has_handle_bbox", False)) != bool(has_handle_bbox):
            raise ValueError(
                f"Episode {payload['path_name']} handle-bbox presence does not match the first episode. "
                "Do not mix handle-latent-ready and legacy raw episodes in one conversion."
            )
        front_camera_pose_base = payload.get("front_camera_pose_base")
        wrist_camera_pose_base = payload.get("wrist_camera_pose_base")
        handle_latents = None
        if args.add_handle_latent:
            handle_latents = compute_episode_handle_latents(
                payload,
                handle_latent_teacher,
                crop_size=args.handle_latent_crop_size,
                margin=args.handle_latent_bbox_margin,
                min_area=args.handle_bbox_min_area,
                min_size=args.handle_bbox_min_size,
                batch_size=args.handle_latent_batch_size,
            )
        n = payload["n"]
        for i in range(n):
            if point_cloud_enabled:
                wrist_first_frame = wrist_first[i]
                wrist_second_frame = wrist_second[i]
                front_first_frame = front_first[i]
                front_second_frame = front_second[i]
            else:
                wrist_first_frame = image_to_three_channel_uint8(wrist_first[i])
                wrist_second_frame = image_to_three_channel_uint8(wrist_second[i])
                front_first_frame = image_to_three_channel_uint8(front_first[i])
                front_second_frame = image_to_three_channel_uint8(front_second[i])
            recorder.add_frame(
                states[i],
                wrist_first_frame,
                wrist_second_frame,
                actions[i],
                int(subtasks[i]),
                front_mask_rgb=front_first_frame,
                front_second_rgb=front_second_frame,
                action_loss_weight=action_loss_weight[i],
                is_recovery=is_recovery[i] if args.include_recovery_indicator else None,
                front_camera_pose_base=None if front_camera_pose_base is None else front_camera_pose_base[i],
                wrist_camera_pose_base=None if wrist_camera_pose_base is None else wrist_camera_pose_base[i],
                front_handle_latent=None if handle_latents is None else handle_latents["front_handle_latent"][i],
                front_handle_latent_valid=None if handle_latents is None else handle_latents["front_handle_latent_valid"][i],
                wrist_handle_latent=None if handle_latents is None else handle_latents["wrist_handle_latent"][i],
                wrist_handle_latent_valid=None if handle_latents is None else handle_latents["wrist_handle_latent_valid"][i],
                end_signal=None if end_signal is None else end_signal[i],
                interaction_contact=(
                    None if interaction_state is None else interaction_state[INTERACTION_CONTACT_FEATURE][i]
                ),
                interaction_handle_progress=(
                    None if interaction_state is None else interaction_state[INTERACTION_HANDLE_PROGRESS_FEATURE][i]
                ),
                interaction_door_progress=(
                    None if interaction_state is None else interaction_state[INTERACTION_DOOR_PROGRESS_FEATURE][i]
                ),
                point_cloud=None if point_cloud is None else point_cloud[i],
            )
        recorder.save_episode()
        print(f"Converted {payload['path_name']}: {n} frames task={task!r} ({ep_idx + 1}/{len(files)})", flush=True)
    recorder.finalize()
    if args.add_end_signal:
        ratio = float(end_positive_frames) / max(1, int(end_total_frames))
        print(
            "End signal summary: "
            f"positive={end_positive_frames} negative={end_total_frames - end_positive_frames} "
            f"total={end_total_frames} positive_ratio={ratio:.6f} "
            f"zero_positive_episodes={end_zero_positive_episodes}/{len(files)}",
            flush=True,
        )
    if args.add_interaction_state:
        print("Interaction state summary by door:", flush=True)
        for door_name in sorted(interaction_stats):
            pieces = []
            for key in INTERACTION_STATE_FEATURES:
                stat = interaction_stats[door_name][key]
                first_values = stat["first_one"]
                first_text = "none" if not first_values else f"{min(first_values)}..{max(first_values)}"
                pieces.append(
                    f"{key.rsplit('.', 1)[-1]}(min={stat['min']:.3f},max={stat['max']:.3f},"
                    f"mean={stat['sum']/max(1, stat['count']):.3f},first1={first_text})"
                )
            print(f"  {door_name}: " + " ".join(pieces), flush=True)
    feature_sidecar = Path(args.root) / args.repo_id / "door_dp_feature_names.json"
    feature_sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar_payload = {
        "fps": fps,
        "state": state_names,
        "action": action_names,
        "image_features": [] if point_cloud_enabled else lerobot_image_keys_for_vision_mode(vision_mode),
        "source_raw_root": str(raw_root),
        "source_raw_roots": [str(root) for root in raw_roots],
        "action_frame": action_frame,
        "action_pose_frame": action_frame,
        "target_pose_frame": action_frame,
        "ikpush_state_version": ikpush_state_version,
        "state_action_mode": args.state_action_mode,
        "action_source": (
            "base_command_plus_tracik_smoothed_joint_command"
            if args.state_action_mode == TRACIK_JOINT_STATE9_MODE
            else (
                "base_command_plus_a2w_z1_joint_targets"
                if args.state_action_mode == DIRECT_JOINT_STATE9_MODE
                else inherited_metadata.get("action_source", "raw")
            )
        ),
        "door_dp_mode": controller_mode,
        "controller_mode": controller_mode,
        "image_storage": args.image_storage,
        "video_codec": args.video_codec,
        "state_sanitize": state_sanitize_config,
        "action_sanitize": action_sanitize_config,
        "state_preprocess": state_preprocess_config,
        "action_preprocess": action_preprocess_config,
        "state_normalized": converted_state_normalized,
        "action_loss_weight_feature": ACTION_LOSS_WEIGHT_FEATURE,
        **(
            {"recovery_indicator_feature": RECOVERY_INDICATOR_FEATURE}
            if args.include_recovery_indicator
            else {}
        ),
        "keyframe_loss_weight": converted_keyframe_loss_weight,
        "keyframe_loss_radius": converted_keyframe_loss_radius,
    }
    if has_camera_pose and not point_cloud_enabled:
        sidecar_payload["camera_pose_features"] = [FRONT_CAMERA_POSE_FEATURE, WRIST_CAMERA_POSE_FEATURE]
        sidecar_payload["camera_pose_frame"] = "robot_base"
        sidecar_payload["camera_pose_convention"] = "optical_frame"
    if has_handle_bbox:
        sidecar_payload["handle_bbox_features"] = [
            RAW_FRONT_HANDLE_BBOX_KEY,
            RAW_FRONT_HANDLE_BBOX_VALID_KEY,
            RAW_WRIST_HANDLE_BBOX_KEY,
            RAW_WRIST_HANDLE_BBOX_VALID_KEY,
        ]
        sidecar_payload["handle_bbox_convention"] = "xyxy_half_open_pixels"
    if args.add_handle_latent:
        sidecar_payload["handle_latent_features"] = [
            FRONT_HANDLE_LATENT_FEATURE,
            FRONT_HANDLE_LATENT_VALID_FEATURE,
            WRIST_HANDLE_LATENT_FEATURE,
            WRIST_HANDLE_LATENT_VALID_FEATURE,
        ]
        sidecar_payload["handle_latent_teacher_model"] = args.handle_latent_teacher_model
        sidecar_payload["handle_latent_crop_size"] = int(args.handle_latent_crop_size)
        sidecar_payload["handle_latent_bbox_margin"] = float(args.handle_latent_bbox_margin)
        sidecar_payload["handle_bbox_min_area"] = float(args.handle_bbox_min_area)
        sidecar_payload["handle_bbox_min_size"] = float(args.handle_bbox_min_size)
    if args.add_interaction_state:
        sidecar_payload["interaction_state_features"] = list(INTERACTION_STATE_FEATURES)
        sidecar_payload["interaction_state_version"] = "causal_cummax_v1"
        sidecar_payload["interaction_contact_min_consecutive_frames"] = int(
            args.interaction_contact_min_consecutive_frames
        )
        sidecar_payload["interaction_handle_unlock_angle_deg"] = float(
            args.interaction_handle_unlock_angle_deg
        )
        sidecar_payload["interaction_door_goal_angle_deg"] = float(
            args.interaction_door_goal_angle_deg
        )
    if point_cloud_enabled:
        sidecar_payload.update(
            {
                "point_cloud_conditioning": True,
                "point_cloud_feature": POINT_CLOUD_FEATURE,
                "point_cloud_views": args.point_cloud_views,
                "point_cloud_num_points": int(args.point_cloud_num_points),
                "point_cloud_candidate_rows": int(args.point_cloud_candidate_rows),
                "point_cloud_candidate_cols": int(args.point_cloud_candidate_cols),
                "point_cloud_workspace_min": list(point_cloud_config.workspace_min),
                "point_cloud_workspace_max": list(point_cloud_config.workspace_max),
                "point_cloud_near_clip_m": float(point_cloud_config.near_clip_m),
                "point_cloud_far_clip_m": float(point_cloud_config.far_clip_m),
                "point_cloud_empty_depth_policy": args.point_cloud_empty_depth_policy,
                "point_cloud_storage": args.point_cloud_storage,
                "point_cloud_frame": "robot_base",
                **{
                    f"{view}_camera_intrinsics": intrinsics_dict(
                        point_cloud_camera_intrinsics, camera=view
                    )
                    for view in args.point_cloud_views.split(",")
                },
            }
        )
    if sidecar:
        for key in DATASET_METADATA_KEYS:
            if point_cloud_enabled and key in {
                "camera_pose_features",
                "handle_bbox_features",
                "handle_latent_features",
            }:
                continue
            if key in sidecar and key not in sidecar_payload:
                sidecar_payload[key] = sidecar[key]
    if vision_mode != "depth":
        sidecar_payload["vision_mode"] = vision_mode
    with open(feature_sidecar, "w", encoding="utf-8") as f:
        json.dump(sidecar_payload, f, indent=2)
    print(f"Done. LeRobotDataset written to {Path(args.root) / args.repo_id}")


if __name__ == "__main__":
    main()
