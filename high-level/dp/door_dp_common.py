import json
import math
import os
from pathlib import Path
from collections import deque

import numpy as np
import torch

try:
    from dp.depth_camera_aug import DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH
except Exception:
    from depth_camera_aug import DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH

IMAGE_HEIGHT = DEPTH_IMAGE_HEIGHT
IMAGE_WIDTH = DEPTH_IMAGE_WIDTH
DEFAULT_DEPTH_LOWER_METERS = 0.02
DEFAULT_DEPTH_FAR_METERS = 2.0
DEPTH_IMAGE_KEYS = ["wrist_handle_mask", "wrist_masked_depth", "front_handle_mask", "front_masked_depth"]
DEPTH_ONLY_IMAGE_KEYS = ["wrist_masked_depth", "front_masked_depth"]
RGB_IMAGE_KEYS = ["wrist_handle_mask", "wrist_rgb", "front_handle_mask", "front_rgb"]
DEPTH_LEROBOT_IMAGE_KEYS = [
    "observation.images.wrist_handle_mask",
    "observation.images.wrist_masked_depth",
    "observation.images.front_handle_mask",
    "observation.images.front_masked_depth",
]
DEPTH_ONLY_LEROBOT_IMAGE_KEYS = [
    "observation.images.wrist_masked_depth",
    "observation.images.front_masked_depth",
]
RGB_LEROBOT_IMAGE_KEYS = [
    "observation.images.wrist_handle_mask",
    "observation.images.wrist_rgb",
    "observation.images.front_handle_mask",
    "observation.images.front_rgb",
]
FRONT_CAMERA_POSE_FEATURE = "observation.camera_pose.front"
WRIST_CAMERA_POSE_FEATURE = "observation.camera_pose.wrist"
CAMERA_POSE_FEATURES = [FRONT_CAMERA_POSE_FEATURE, WRIST_CAMERA_POSE_FEATURE]
CAMERA_POSE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
RAW_FRONT_CAMERA_POSE_KEY = "front_camera_pose_base"
RAW_WRIST_CAMERA_POSE_KEY = "wrist_camera_pose_base"
RAW_FRONT_HANDLE_BBOX_KEY = "front_handle_bbox_xyxy"
RAW_FRONT_HANDLE_BBOX_VALID_KEY = "front_handle_bbox_valid"
RAW_WRIST_HANDLE_BBOX_KEY = "wrist_handle_bbox_xyxy"
RAW_WRIST_HANDLE_BBOX_VALID_KEY = "wrist_handle_bbox_valid"
FRONT_HANDLE_LATENT_FEATURE = "aux.front_handle_latent"
FRONT_HANDLE_LATENT_VALID_FEATURE = "aux.front_handle_latent_valid"
WRIST_HANDLE_LATENT_FEATURE = "aux.wrist_handle_latent"
WRIST_HANDLE_LATENT_VALID_FEATURE = "aux.wrist_handle_latent_valid"
HANDLE_LATENT_FEATURES = [
    FRONT_HANDLE_LATENT_FEATURE,
    FRONT_HANDLE_LATENT_VALID_FEATURE,
    WRIST_HANDLE_LATENT_FEATURE,
    WRIST_HANDLE_LATENT_VALID_FEATURE,
]
DATASET_METADATA_KEYS = (
    "action_frame",
    "action_pose_frame",
    "target_pose_frame",
    "state_pose_frame",
    "ee_pose_frame",
    "ikpush_state_version",
    "door_dp_mode",
    "controller_mode",
    "state_format",
    "state_source",
    "action_format",
    "action_source",
    "state_action_mode",
    "state_normalized",
    "pi05_state_action_aligned",
    "state_preprocess",
    "action_preprocess",
    "state_sanitize",
    "action_sanitize",
    "camera_fps",
    "camera_sample_stride",
    "camera_hold_last_frame",
    "image_width",
    "image_height",
    "depth_noise_enabled",
    "depth_noise_config",
    "depth_camera_randomization_config",
    "camera_intrinsics",
    "camera_pose_frame",
    "camera_pose_convention",
    "camera_pose_features",
    "handle_bbox_features",
    "handle_bbox_convention",
    "handle_latent_features",
    "handle_latent_teacher_model",
    "handle_latent_crop_size",
    "handle_latent_bbox_margin",
    "handle_bbox_min_area",
    "handle_bbox_min_size",
    "phase_names",
    "keyframe_loss_enabled",
    "keyframe_loss_weight",
    "keyframe_loss_radius",
    "keyframe_loss_feature",
    "action_loss_weight_feature",
    "keyframe_extraction_rules",
    "end_signal_enabled",
    "end_signal_feature",
    "end_signal_positive_phases",
    "end_signal_version",
    "interaction_state_features",
    "interaction_state_version",
    "interaction_contact_min_consecutive_frames",
    "interaction_handle_unlock_angle_deg",
    "interaction_door_goal_angle_deg",
    "handle_closed_angle",
    "door_closed_angle",
    "handle_unlock_threshold",
    "door_progress_goal_angle",
)
ACTION_LOSS_WEIGHT_FEATURE = "loss.action_weight"
RECOVERY_INDICATOR_FEATURE = "aux.is_recovery"
RAW_ACTION_LOSS_WEIGHT_KEY = "action_loss_weight"
END_SIGNAL_FEATURE = "aux.end_signal"
RAW_END_SIGNAL_KEY = "end_signal"
DEFAULT_END_SIGNAL_POSITIVE_PHASES = ("return_home", "hold_home")
INTERACTION_CONTACT_FEATURE = "aux.interaction_contact"
INTERACTION_HANDLE_PROGRESS_FEATURE = "aux.interaction_handle_progress"
INTERACTION_DOOR_PROGRESS_FEATURE = "aux.interaction_door_progress"
INTERACTION_STATE_FEATURES = (
    INTERACTION_CONTACT_FEATURE,
    INTERACTION_HANDLE_PROGRESS_FEATURE,
    INTERACTION_DOOR_PROGRESS_FEATURE,
)
DEFAULT_KEYFRAME_LOSS_WEIGHT = 8.0
DEFAULT_KEYFRAME_LOSS_RADIUS = 3
DEFAULT_KEYFRAME_NAMES = (
    "start",
    "stop_before_door",
    "pregrasp",
    "grasp",
    "rotate",
)
DEFAULT_KEYFRAME_PHASE_TARGETS = {
    "start": (),
    "stop_before_door": ("initial_hold",),
    # First grasp-frame is the frame immediately after the pregrasp hold/move.
    "pregrasp": ("grasp",),
    # At grasp_hold/close_gripper the target has reached the grasp point.
    "grasp": ("grasp_hold", "close_gripper"),
    # First push_door-frame is immediately after rotate_handle completes.
    "rotate": ("push_door",),
}
DEFAULT_A2W_PHASE_NAMES = (
    "walk",
    "initial_hold",
    "grasp",
    "grasp_hold",
    "close_gripper",
    "rotate_handle",
    "push_door",
    "return_home",
    "hold_home",
)
DEFAULT_MOTION_KEYFRAME_CONFIG = {
    "window": 5,
    "min_separation": 15,
    "merge_tolerance": 2,
    "dedup_window": 10,
    "manual_union": True,
    # Event thresholds are intentionally relative to each episode, because
    # randomized starts/resistance can change absolute magnitudes a lot.
    "base_speed_change_quantile": 0.95,
    "base_speed_change_max_count": 4,
    "arm_joint_motion_quantile": 0.85,
    "arm_joint_motion_max_count": 6,
    "handle_speed_change_quantile": 0.95,
    "handle_speed_change_max_count": 4,
    "door_speed_change_quantile": 0.95,
    "door_speed_change_max_count": 2,
    "gripper_contact_min_delta": 1.0e-4,
}
STATE_PREPROCESS_VERSION = "door_dp_state_robust_quantile_v1"
ACTION_PREPROCESS_VERSION = "door_dp_action_robust_quantile_v1"
SANITIZE_VERSION = "door_dp_sanitize_near_zero_rate_v1"
DEFAULT_NEAR_ZERO_RATE_EPS = 1.0e-5
ACTION_NAMES = [
    "vx",
    "yaw",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "ee_qw",
    "gripper",
]
A2W_JOINT_ACTION_NAMES = [
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


def normalize_vision_mode(vision_mode):
    mode = str(vision_mode or "depth").lower().replace("-", "_")
    if mode in ("depthonly", "depth2"):
        mode = "depth_only"
    if mode not in ("depth", "depth_only", "rgb"):
        raise ValueError(f"Unsupported Door DP vision mode: {vision_mode!r}")
    return mode


def raw_image_keys_for_vision_mode(vision_mode):
    mode = normalize_vision_mode(vision_mode)
    if mode == "rgb":
        return RGB_IMAGE_KEYS
    if mode == "depth_only":
        return DEPTH_ONLY_IMAGE_KEYS
    return DEPTH_IMAGE_KEYS


def lerobot_image_keys_for_vision_mode(vision_mode):
    mode = normalize_vision_mode(vision_mode)
    if mode == "rgb":
        return RGB_LEROBOT_IMAGE_KEYS
    if mode == "depth_only":
        return DEPTH_ONLY_LEROBOT_IMAGE_KEYS
    return DEPTH_LEROBOT_IMAGE_KEYS


def depth_image_to_single_channel_uint8(image):
    """Store colorized depth without three identical channel copies."""
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        return np.ascontiguousarray(array)
    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        return np.ascontiguousarray(array[..., 0])
    raise ValueError(f"Expected a 2D or HWC depth image, got shape={array.shape}.")


def image_to_three_channel_uint8(image):
    """Expand compact single-channel raw depth for RGB image consumers."""
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=-1)
    if array.ndim == 3 and array.shape[-1] == 1:
        return np.repeat(array, 3, axis=-1)
    if array.ndim == 3 and array.shape[-1] >= 3:
        return np.ascontiguousarray(array[..., :3])
    raise ValueError(f"Expected a 2D or HWC image, got shape={array.shape}.")


class DoorDPJsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def write(self, record):
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_list(x, precision=5):
    arr = _to_numpy(x).astype(np.float64)
    return np.round(arr, precision).tolist()


def _state_to_2d(states):
    arr = np.asarray(states, dtype=np.float32)
    original_shape = arr.shape
    if arr.ndim == 1:
        return arr.reshape(1, -1), original_shape
    if arr.ndim < 1:
        raise ValueError(f"Expected state with at least 1 dimension, got shape {arr.shape}")
    return arr.reshape(-1, arr.shape[-1]), original_shape


def _restore_state_shape(states_2d, original_shape):
    arr = np.asarray(states_2d, dtype=np.float32)
    return arr.reshape(original_shape).astype(np.float32)


def _wrap_to_pi_array(values):
    return (values + np.float32(np.pi)) % np.float32(2.0 * np.pi) - np.float32(np.pi)


def _state_angle_feature_indices(state_names):
    indices = []
    for idx, name in enumerate(state_names):
        lowered = str(name).lower()
        if any(skip in lowered for skip in ("vel", "rate", "ang_vel", "command", "last_low_action")):
            continue
        if any(token in lowered for token in ("roll", "pitch", "yaw")):
            indices.append(idx)
    return indices


def _near_zero_rate_feature_indices(feature_names):
    indices = []
    for idx, name in enumerate(feature_names):
        lowered = str(name).lower()
        if lowered in ("yaw", "yaw_rate", "base_yaw_rate", "vyaw", "last_command_vyaw"):
            indices.append(idx)
            continue
        if any(token in lowered for token in ("yaw_rate", "ang_vel", "angular_vel")):
            indices.append(idx)
    return indices


def _state_quaternion_groups(state_names):
    name_to_idx = {str(name): idx for idx, name in enumerate(state_names)}
    groups = []
    seen = set()
    for name in state_names:
        name = str(name)
        if not name.endswith("_qx"):
            continue
        prefix = name[:-3]
        group = [f"{prefix}_{suffix}" for suffix in ("qx", "qy", "qz", "qw")]
        if all(part in name_to_idx for part in group):
            indices = tuple(name_to_idx[part] for part in group)
            if indices not in seen:
                groups.append(indices)
                seen.add(indices)
    return groups


def sanitize_door_dp_state(states, state_names, eps=1.0e-6):
    """Clean raw Door DP state before dataset-level state preprocessing.

    This keeps angular features continuous, canonicalizes quaternions, and removes
    non-finite values before fitting/applying the robust quantile transform.
    """

    state_names = list(state_names)
    states_2d, original_shape = _state_to_2d(states)
    if states_2d.shape[-1] != len(state_names):
        raise ValueError(f"State dim {states_2d.shape[-1]} does not match {len(state_names)} state feature names.")
    out = np.nan_to_num(states_2d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=True)
    for idx in _state_angle_feature_indices(state_names):
        out[:, idx] = _wrap_to_pi_array(out[:, idx])
    for idx in _near_zero_rate_feature_indices(state_names):
        values = out[:, idx]
        out[:, idx] = np.where(np.abs(values) <= float(eps), 0.0, values).astype(np.float32)
    for group in _state_quaternion_groups(state_names):
        quat = out[:, group].astype(np.float32, copy=True)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        valid = norm[:, 0] >= float(eps)
        quat[valid] = quat[valid] / norm[valid]
        quat[~valid] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        sign = np.where(quat[:, 3:4] < 0.0, -1.0, 1.0).astype(np.float32)
        out[:, group] = quat * sign
    return _restore_state_shape(out, original_shape)


def fit_door_dp_state_preprocess(
    states,
    state_names,
    lower_quantile=0.01,
    upper_quantile=0.99,
    eps=1.0e-6,
):
    state_names = list(state_names)
    lower_quantile = float(lower_quantile)
    upper_quantile = float(upper_quantile)
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("--state_quantile_low/high must satisfy 0 <= low < high <= 1.")
    clean = sanitize_door_dp_state(states, state_names, eps=eps)
    clean_2d, _ = _state_to_2d(clean)
    q_low = np.quantile(clean_2d, lower_quantile, axis=0).astype(np.float32)
    q_high = np.quantile(clean_2d, upper_quantile, axis=0).astype(np.float32)
    denom = q_high - q_low
    constant_mask = np.abs(denom) < float(eps)
    return {
        "applied": True,
        "version": STATE_PREPROCESS_VERSION,
        "mode": "robust_quantile",
        "feature_names": list(state_names),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "eps": float(eps),
        "q_low": q_low.tolist(),
        "q_high": q_high.tolist(),
        "constant_mask": constant_mask.astype(bool).tolist(),
        "angle_wrapped_features": [state_names[i] for i in _state_angle_feature_indices(state_names)],
        "quaternion_groups": [[state_names[i] for i in group] for group in _state_quaternion_groups(state_names)],
    }


def apply_door_dp_state_preprocess(states, state_names=None, config=None):
    if not config or not bool(config.get("applied", False)):
        return np.asarray(states, dtype=np.float32)
    if str(config.get("version", "")) != STATE_PREPROCESS_VERSION:
        raise ValueError(f"Unsupported Door DP state_preprocess version: {config.get('version')!r}")
    if str(config.get("mode", "")) != "robust_quantile":
        raise ValueError(f"Unsupported Door DP state_preprocess mode: {config.get('mode')!r}")

    feature_names = list(config.get("feature_names") or state_names or [])
    if not feature_names:
        raise ValueError("Door DP state_preprocess metadata is missing feature_names.")
    if state_names is not None:
        state_names = list(state_names)
        if len(state_names) != len(feature_names):
            raise ValueError(
                f"Runtime state_names has {len(state_names)} features, but state_preprocess expects {len(feature_names)}."
            )
        if state_names != feature_names:
            raise ValueError("Runtime state feature names do not match state_preprocess feature_names.")

    clean = sanitize_door_dp_state(states, feature_names, eps=float(config.get("eps", 1.0e-6)))
    clean_2d, original_shape = _state_to_2d(clean)
    q_low = np.asarray(config["q_low"], dtype=np.float32)
    q_high = np.asarray(config["q_high"], dtype=np.float32)
    if q_low.shape != (clean_2d.shape[-1],) or q_high.shape != (clean_2d.shape[-1],):
        raise ValueError(
            f"state_preprocess q_low/q_high shape mismatch: state_dim={clean_2d.shape[-1]} "
            f"q_low={q_low.shape} q_high={q_high.shape}"
        )
    eps = float(config.get("eps", 1.0e-6))
    denom = q_high - q_low
    valid = np.abs(denom) >= eps
    out = np.zeros_like(clean_2d, dtype=np.float32)
    if np.any(valid):
        clipped = np.minimum(np.maximum(clean_2d[:, valid], q_low[valid]), q_high[valid])
        out[:, valid] = 2.0 * (clipped - q_low[valid]) / denom[valid] - 1.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    return _restore_state_shape(out, original_shape)


def invert_door_dp_state_preprocess(states, state_names=None, config=None):
    if not config or not bool(config.get("applied", False)):
        return np.asarray(states, dtype=np.float32)
    if str(config.get("version", "")) != STATE_PREPROCESS_VERSION:
        raise ValueError(f"Unsupported Door DP state_preprocess version: {config.get('version')!r}")
    if str(config.get("mode", "")) != "robust_quantile":
        raise ValueError(f"Unsupported Door DP state_preprocess mode: {config.get('mode')!r}")

    feature_names = list(config.get("feature_names") or state_names or [])
    if not feature_names:
        raise ValueError("Door DP state_preprocess metadata is missing feature_names.")
    if state_names is not None:
        state_names = list(state_names)
        if len(state_names) != len(feature_names):
            raise ValueError(
                f"Runtime state_names has {len(state_names)} features, but state_preprocess expects {len(feature_names)}."
            )
        if state_names != feature_names:
            raise ValueError("Runtime state feature names do not match state_preprocess feature_names.")

    states_2d, original_shape = _state_to_2d(states)
    q_low = np.asarray(config["q_low"], dtype=np.float32)
    q_high = np.asarray(config["q_high"], dtype=np.float32)
    if q_low.shape != (states_2d.shape[-1],) or q_high.shape != (states_2d.shape[-1],):
        raise ValueError(
            f"state_preprocess q_low/q_high shape mismatch: state_dim={states_2d.shape[-1]} "
            f"q_low={q_low.shape} q_high={q_high.shape}"
        )
    eps = float(config.get("eps", 1.0e-6))
    clipped = np.clip(np.nan_to_num(states_2d, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)
    denom = q_high - q_low
    valid = np.abs(denom) >= eps
    out = np.zeros_like(clipped, dtype=np.float32)
    if np.any(valid):
        out[:, valid] = (clipped[:, valid] + 1.0) * denom[valid] / 2.0 + q_low[valid]
    if np.any(~valid):
        out[:, ~valid] = 0.5 * (q_low[~valid] + q_high[~valid])
    out = sanitize_door_dp_state(out, feature_names, eps=eps)
    return _restore_state_shape(out, original_shape)


def sanitize_door_dp_action(actions, action_names=None, eps=1.0e-6):
    action_names = list(action_names or ACTION_NAMES)
    actions_2d, original_shape = _state_to_2d(actions)
    if actions_2d.shape[-1] != len(action_names):
        raise ValueError(f"Action dim {actions_2d.shape[-1]} does not match {len(action_names)} action names.")
    out = np.nan_to_num(actions_2d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=True)
    for idx in _near_zero_rate_feature_indices(action_names):
        values = out[:, idx]
        out[:, idx] = np.where(np.abs(values) <= float(eps), 0.0, values).astype(np.float32)
    for group in _state_quaternion_groups(action_names):
        quat = out[:, group].astype(np.float32, copy=True)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        valid = norm[:, 0] >= float(eps)
        quat[valid] = quat[valid] / norm[valid]
        quat[~valid] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        sign = np.where(quat[:, 3:4] < 0.0, -1.0, 1.0).astype(np.float32)
        out[:, group] = quat * sign
    return _restore_state_shape(out, original_shape)


def make_door_dp_sanitize_config(feature_names, eps=DEFAULT_NEAR_ZERO_RATE_EPS):
    feature_names = list(feature_names)
    rate_indices = _near_zero_rate_feature_indices(feature_names)
    return {
        "applied": True,
        "version": SANITIZE_VERSION,
        "eps": float(eps),
        "near_zero_rate_features": [feature_names[i] for i in rate_indices],
        "quaternion_groups": [[feature_names[i] for i in group] for group in _state_quaternion_groups(feature_names)],
    }


def fit_door_dp_action_preprocess(
    actions,
    action_names=None,
    lower_quantile=0.01,
    upper_quantile=0.99,
    eps=1.0e-6,
):
    action_names = list(action_names or ACTION_NAMES)
    lower_quantile = float(lower_quantile)
    upper_quantile = float(upper_quantile)
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("--action_quantile_low/high must satisfy 0 <= low < high <= 1.")
    clean = sanitize_door_dp_action(actions, action_names, eps=eps)
    clean_2d, _ = _state_to_2d(clean)
    q_low = np.quantile(clean_2d, lower_quantile, axis=0).astype(np.float32)
    q_high = np.quantile(clean_2d, upper_quantile, axis=0).astype(np.float32)
    denom = q_high - q_low
    constant_mask = np.abs(denom) < float(eps)
    return {
        "applied": True,
        "version": ACTION_PREPROCESS_VERSION,
        "mode": "robust_quantile",
        "feature_names": list(action_names),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "eps": float(eps),
        "q_low": q_low.tolist(),
        "q_high": q_high.tolist(),
        "constant_mask": constant_mask.astype(bool).tolist(),
        "quaternion_groups": [[action_names[i] for i in group] for group in _state_quaternion_groups(action_names)],
    }


def apply_door_dp_action_preprocess(actions, action_names=None, config=None):
    if not config or not bool(config.get("applied", False)):
        return np.asarray(actions, dtype=np.float32)
    if str(config.get("version", "")) != ACTION_PREPROCESS_VERSION:
        raise ValueError(f"Unsupported Door DP action_preprocess version: {config.get('version')!r}")
    if str(config.get("mode", "")) != "robust_quantile":
        raise ValueError(f"Unsupported Door DP action_preprocess mode: {config.get('mode')!r}")

    feature_names = list(config.get("feature_names") or action_names or ACTION_NAMES)
    if action_names is not None:
        action_names = list(action_names)
        if len(action_names) != len(feature_names):
            raise ValueError(
                f"Runtime action_names has {len(action_names)} features, but action_preprocess expects {len(feature_names)}."
            )
        if action_names != feature_names:
            raise ValueError("Runtime action names do not match action_preprocess feature_names.")

    clean = sanitize_door_dp_action(actions, feature_names, eps=float(config.get("eps", 1.0e-6)))
    clean_2d, original_shape = _state_to_2d(clean)
    q_low = np.asarray(config["q_low"], dtype=np.float32)
    q_high = np.asarray(config["q_high"], dtype=np.float32)
    if q_low.shape != (clean_2d.shape[-1],) or q_high.shape != (clean_2d.shape[-1],):
        raise ValueError(
            f"action_preprocess q_low/q_high shape mismatch: action_dim={clean_2d.shape[-1]} "
            f"q_low={q_low.shape} q_high={q_high.shape}"
        )
    eps = float(config.get("eps", 1.0e-6))
    denom = q_high - q_low
    valid = np.abs(denom) >= eps
    out = np.zeros_like(clean_2d, dtype=np.float32)
    if np.any(valid):
        clipped = np.minimum(np.maximum(clean_2d[:, valid], q_low[valid]), q_high[valid])
        out[:, valid] = 2.0 * (clipped - q_low[valid]) / denom[valid] - 1.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    return _restore_state_shape(out, original_shape)


def invert_door_dp_action_preprocess(actions, action_names=None, config=None):
    if not config or not bool(config.get("applied", False)):
        return np.asarray(actions, dtype=np.float32)
    if str(config.get("version", "")) != ACTION_PREPROCESS_VERSION:
        raise ValueError(f"Unsupported Door DP action_preprocess version: {config.get('version')!r}")
    if str(config.get("mode", "")) != "robust_quantile":
        raise ValueError(f"Unsupported Door DP action_preprocess mode: {config.get('mode')!r}")

    feature_names = list(config.get("feature_names") or action_names or ACTION_NAMES)
    if action_names is not None:
        action_names = list(action_names)
        if len(action_names) != len(feature_names):
            raise ValueError(
                f"Runtime action_names has {len(action_names)} features, but action_preprocess expects {len(feature_names)}."
            )
        if action_names != feature_names:
            raise ValueError("Runtime action names do not match action_preprocess feature_names.")

    actions_2d, original_shape = _state_to_2d(actions)
    q_low = np.asarray(config["q_low"], dtype=np.float32)
    q_high = np.asarray(config["q_high"], dtype=np.float32)
    if q_low.shape != (actions_2d.shape[-1],) or q_high.shape != (actions_2d.shape[-1],):
        raise ValueError(
            f"action_preprocess q_low/q_high shape mismatch: action_dim={actions_2d.shape[-1]} "
            f"q_low={q_low.shape} q_high={q_high.shape}"
        )
    eps = float(config.get("eps", 1.0e-6))
    clipped = np.clip(np.nan_to_num(actions_2d, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)
    denom = q_high - q_low
    valid = np.abs(denom) >= eps
    out = np.zeros_like(clipped, dtype=np.float32)
    if np.any(valid):
        out[:, valid] = (clipped[:, valid] + 1.0) * denom[valid] / 2.0 + q_low[valid]
    if np.any(~valid):
        out[:, ~valid] = 0.5 * (q_low[~valid] + q_high[~valid])
    out = sanitize_door_dp_action(out, feature_names, eps=eps)
    return _restore_state_shape(out, original_shape)


def make_state_feature_names(num_dofs, num_actions, phase_names):
    names = ["base_roll", "base_pitch", "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
    names += [f"dof_pos_{i}" for i in range(num_dofs)]
    names += [f"dof_vel_{i}" for i in range(num_dofs)]
    names += [f"last_low_action_{i}" for i in range(num_actions)]
    names += [f"foot_contact_{i}" for i in range(4)]
    names += ["ee_base_x", "ee_base_y", "ee_base_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
    names += ["gripper_pos"]
    return names


def get_door_dp_state(env, phase_id, phase_names, env_id=0):
    from isaacgym.torch_utils import euler_from_quat, quat_rotate_inverse

    env_id = int(env_id)
    roll, pitch, _ = euler_from_quat(env.root_states[:, 3:7])
    arm_base_pos = env.base_pos
    if hasattr(env, "arm_base_offset"):
        from isaacgym.torch_utils import quat_apply

        arm_base_pos = env.base_pos + quat_apply(env.base_yaw_quat, env.arm_base_offset)
    ee_pos_base = quat_rotate_inverse(env.root_states[:, 3:7], env.ee_pos - arm_base_pos)
    gripper_pos = env.dof_pos[:, -env.cfg.env.num_gripper_joints :].mean(dim=-1, keepdim=True)
    parts = [
        torch.stack([roll, pitch], dim=-1),
        env.base_ang_vel,
        env.dof_pos,
        env.dof_vel,
        env.last_actions,
        env._reindex_feet(env.foot_contacts_from_sensor).to(torch.float32),
        ee_pos_base,
        env.ee_orn / torch.clamp(torch.norm(env.ee_orn, dim=-1, keepdim=True), min=1e-6),
        gripper_pos,
    ]
    return torch.cat(parts, dim=-1)[env_id].detach().cpu().to(torch.float32).numpy()


def get_door_dp_action(env, env_id=0):
    env_id = int(env_id)
    quat = env.ee_goal_orn_quat / torch.clamp(torch.norm(env.ee_goal_orn_quat, dim=-1, keepdim=True), min=1e-6)
    gripper = env.external_gripper_target[:, :1].mean(dim=-1)
    action = torch.cat(
        [
            env.commands[:, 0:1],
            env.commands[:, 2:3],
            env.curr_ee_goal_cart_world[:, :3],
            quat[:, :4],
            gripper[:, None],
        ],
        dim=-1,
    )
    return action[env_id].detach().cpu().to(torch.float32).numpy()


def make_door_dp_replay_snapshot(env, env_id=0):
    env_id = int(env_id)
    snapshot = {
        "replay_root_state": _to_numpy(env.root_states[env_id]).astype(np.float32),
        "replay_dof_pos": _to_numpy(env.dof_pos[env_id]).astype(np.float32),
        "replay_dof_vel": _to_numpy(env.dof_vel[env_id]).astype(np.float32),
        "replay_ee_pos": _to_numpy(env.ee_pos[env_id]).astype(np.float32),
        "replay_ee_quat": _to_numpy(env.ee_orn[env_id]).astype(np.float32),
    }
    if hasattr(env, "door_root_state"):
        snapshot["replay_door_root_state"] = _to_numpy(env.door_root_state[env_id]).astype(np.float32)
    if hasattr(env, "box_root_state"):
        snapshot["replay_box_root_state"] = _to_numpy(env.box_root_state[env_id]).astype(np.float32)
    if hasattr(env, "_door_dof_pos"):
        snapshot["replay_door_dof_pos"] = _to_numpy(env._door_dof_pos[env_id]).astype(np.float32)
    elif hasattr(env, "door_dof_pos"):
        snapshot["replay_door_dof_pos"] = _to_numpy(env.door_dof_pos[env_id]).astype(np.float32)
    if hasattr(env, "_door_dof_vel"):
        snapshot["replay_door_dof_vel"] = _to_numpy(env._door_dof_vel[env_id]).astype(np.float32)
    elif hasattr(env, "door_dof_vel"):
        snapshot["replay_door_dof_vel"] = _to_numpy(env.door_dof_vel[env_id]).astype(np.float32)
    return snapshot


def make_door_dp_log_record(env, step, dp_action, env_id=0, phase_id=None, phase_names=None, extra=None):
    env_id = int(env_id)
    quat = env.ee_goal_orn_quat / torch.clamp(torch.norm(env.ee_goal_orn_quat, dim=-1, keepdim=True), min=1e-6)
    ee_orn = env.ee_orn / torch.clamp(torch.norm(env.ee_orn, dim=-1, keepdim=True), min=1e-6)
    phase_value = None
    phase_name = None
    if phase_id is not None:
        phase_value = int(_to_numpy(phase_id[env_id]).item())
        if phase_names is not None and 0 <= phase_value < len(phase_names):
            phase_name = phase_names[phase_value]
    record = {
        "step": int(step),
        "controlled_env_id": env_id,
        "num_envs": int(env.num_envs),
        "only_controlled_env_uses_dp": True,
        "phase_id": phase_value,
        "phase_name": phase_name,
        "dp_action_names": ACTION_NAMES,
        "dp_action_raw": _to_list(dp_action),
        "applied_action": _to_list(get_door_dp_action(env, env_id)),
        "robot_command": {
            "vx": float(_to_numpy(env.commands[env_id, 0]).item()),
            "vy": float(_to_numpy(env.commands[env_id, 1]).item()),
            "yaw": float(_to_numpy(env.commands[env_id, 2]).item()),
        },
        "base": {
            "xy": _to_list(env.root_states[env_id, :2]),
            "height": float(_to_numpy(env.root_states[env_id, 2]).item()),
            "lin_vel": _to_list(env.base_lin_vel[env_id]),
            "ang_vel": _to_list(env.base_ang_vel[env_id]),
        },
        "ee": {
            "target_pos_world": _to_list(env.curr_ee_goal_cart_world[env_id, :3]),
            "target_quat": _to_list(quat[env_id, :4]),
            "target_delta_rpy": _to_list(env.ee_goal_orn_delta_rpy[env_id]),
            "actual_pos_world": _to_list(env.ee_pos[env_id, :3]),
            "actual_quat": _to_list(ee_orn[env_id, :4]),
            "pos_error": _to_list(env.curr_ee_goal_cart_world[env_id, :3] - env.ee_pos[env_id, :3]),
        },
        "gripper": {
            "target": _to_list(env.external_gripper_target[env_id]),
            "actual_pos": _to_list(env.dof_pos[env_id, -env.cfg.env.num_gripper_joints :]),
        },
    }
    if hasattr(env, "_door_dof_pos"):
        record["door"] = {
            "dof": _to_list(env._door_dof_pos[env_id]),
        }
    elif hasattr(env, "door_dof_pos"):
        record["door"] = {
            "dof": _to_list(env.door_dof_pos[env_id]),
        }
    if extra:
        record["extra"] = extra
    return record


def print_door_dp_log_record(record):
    action = record["dp_action_raw"]
    cmd = record["robot_command"]
    ee = record["ee"]
    gripper = record["gripper"]
    print(
        "[DoorDP]"
        f" step={record['step']}"
        f" env={record['controlled_env_id']}/{record['num_envs']}"
        f" phase={record.get('phase_name')}"
        f" action(vx,yaw,ee,grip)=({action[0]:.3f}, {action[1]:.3f}, "
        f"[{action[2]:.3f}, {action[3]:.3f}, {action[4]:.3f}], {action[9]:.3f})"
        f" cmd=({cmd['vx']:.3f}, {cmd['yaw']:.3f})"
        f" ee_target={ee['target_pos_world']}"
        f" ee_actual={ee['actual_pos_world']}"
        f" ee_err={ee['pos_error']}"
        f" grip={gripper['target']}",
        flush=True,
    )


def _image_pair_from_camera_tensors(
    camera_images,
    mask_key,
    depth_key,
    env_id=0,
    depth_lower=DEFAULT_DEPTH_LOWER_METERS,
    depth_far=DEFAULT_DEPTH_FAR_METERS,
):
    env_id = int(env_id)
    if mask_key not in camera_images or depth_key not in camera_images:
        return None, None
    mask = np.squeeze(_to_numpy(camera_images[mask_key][env_id])).astype(np.float32)
    depth_image = np.squeeze(_to_numpy(camera_images[depth_key][env_id])).astype(np.float32)
    mask_u8 = (255.0 * np.clip(mask, 0.0, 1.0)).astype(np.uint8)
    valid = depth_image[np.isfinite(depth_image) & (depth_image > 0.0)]
    valid = valid[np.isfinite(valid) & (valid > 0.0)]
    depth_u8 = np.zeros_like(mask_u8)
    if valid.size > 0:
        scaled = (depth_image - float(depth_lower)) / max(float(depth_far) - float(depth_lower), 1e-4)
        depth_u8 = (255.0 * np.clip(scaled, 0.0, 1.0)).astype(np.uint8)
    # Store as RGB-compatible images for LeRobot/video tools and the shared 3-channel CNN encoder.
    # Depth is a single grayscale full-depth visualization; all three channels are identical.
    return np.repeat(mask_u8[..., None], 3, axis=-1), np.repeat(depth_u8[..., None], 3, axis=-1)


def _mask_from_camera_tensors(camera_images, mask_key, env_id=0):
    env_id = int(env_id)
    if mask_key not in camera_images:
        return None
    mask = np.squeeze(_to_numpy(camera_images[mask_key][env_id])).astype(np.float32)
    mask_u8 = (255.0 * np.clip(mask, 0.0, 1.0)).astype(np.uint8)
    return np.repeat(mask_u8[..., None], 3, axis=-1)


def _rgb_from_camera_tensors(camera_images, rgb_key, env_id=0):
    env_id = int(env_id)
    if rgb_key not in camera_images:
        return None
    rgb = _to_numpy(camera_images[rgb_key][env_id])
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    elif rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.transpose(rgb, (1, 2, 0))
    elif rgb.ndim != 3:
        raise ValueError(f"Expected RGB image with 2 or 3 dims, got shape {rgb.shape}")
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        max_value = float(finite.max()) if finite.size else 0.0
        if max_value <= 1.5:
            rgb = rgb * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def images_from_camera_tensors(
    camera_images,
    env_id=0,
    depth_lower=DEFAULT_DEPTH_LOWER_METERS,
    depth_far=DEFAULT_DEPTH_FAR_METERS,
):
    mask_key = "wrist_handle_mask" if "wrist_handle_mask" in camera_images else "handle_mask"
    depth_key = "wrist_handle_masked_depth" if "wrist_handle_masked_depth" in camera_images else "handle_masked_depth"
    return _image_pair_from_camera_tensors(
        camera_images,
        mask_key,
        depth_key,
        env_id,
        depth_lower=depth_lower,
        depth_far=depth_far,
    )


def dp_image_inputs_from_camera_tensors(
    camera_images,
    env_id=0,
    vision_mode="depth",
    depth_lower=DEFAULT_DEPTH_LOWER_METERS,
    depth_far=DEFAULT_DEPTH_FAR_METERS,
):
    vision_mode = normalize_vision_mode(vision_mode)
    if vision_mode == "depth":
        wrist_mask, wrist_depth = images_from_camera_tensors(
            camera_images,
            env_id,
            depth_lower=depth_lower,
            depth_far=depth_far,
        )
        front_mask, front_depth = _image_pair_from_camera_tensors(
            camera_images,
            "front_handle_mask",
            "front_handle_masked_depth",
            env_id,
            depth_lower=depth_lower,
            depth_far=depth_far,
        )
        return wrist_mask, wrist_depth, front_mask, front_depth
    mask_key = "wrist_handle_mask" if "wrist_handle_mask" in camera_images else "handle_mask"
    wrist_mask = _mask_from_camera_tensors(camera_images, mask_key, env_id)
    front_mask = _mask_from_camera_tensors(camera_images, "front_handle_mask", env_id)
    wrist_rgb = _rgb_from_camera_tensors(camera_images, "wrist_rgb" if "wrist_rgb" in camera_images else "rgb", env_id)
    front_rgb = _rgb_from_camera_tensors(camera_images, "front_rgb", env_id)
    return wrist_mask, wrist_rgb, front_mask, front_rgb


def _zero_image_like(image):
    if image is None:
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    return np.zeros_like(np.asarray(image, dtype=np.uint8))


def _to_str_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def make_end_signal_from_phase_ids(
    phase_ids,
    phase_names,
    positive_phases=DEFAULT_END_SIGNAL_POSITIVE_PHASES,
):
    """Build a dense binary end target aligned one-to-one with recorded frames."""
    phase_ids = np.asarray(phase_ids, dtype=np.int64).reshape(-1)
    names = _to_str_list(phase_names)
    wanted = {str(name).strip() for name in positive_phases if str(name).strip()}
    positive_ids = {idx for idx, name in enumerate(names) if name in wanted}
    values = np.isin(phase_ids, list(positive_ids)).astype(np.float32)
    return values.reshape(-1, 1)


def make_interaction_state_targets(
    contact_both,
    door_dof_pos,
    *,
    contact_min_consecutive_frames=3,
    handle_closed_angle=None,
    door_closed_angle=None,
    handle_unlock_delta_rad=math.radians(40.0),
    door_goal_delta_rad=math.radians(90.0),
):
    """Create causal contact and cumulative articulation progress targets."""
    contact_both = np.asarray(contact_both, dtype=np.float32).reshape(-1)
    door_dof_pos = np.asarray(door_dof_pos, dtype=np.float32)
    if door_dof_pos.ndim != 2 or door_dof_pos.shape[1] < 2:
        raise ValueError(f"door_dof_pos must have shape (T, >=2), got {door_dof_pos.shape}.")
    if contact_both.shape[0] != door_dof_pos.shape[0]:
        raise ValueError(
            f"contact/door frame count mismatch: {contact_both.shape[0]} vs {door_dof_pos.shape[0]}."
        )
    if not np.all(np.isfinite(contact_both)) or not np.all(np.isfinite(door_dof_pos[:, :2])):
        raise ValueError("Interaction-state source arrays must contain only finite values.")
    min_frames = int(contact_min_consecutive_frames)
    if min_frames <= 0:
        raise ValueError("contact_min_consecutive_frames must be positive.")
    handle_unlock_delta_rad = float(handle_unlock_delta_rad)
    door_goal_delta_rad = float(door_goal_delta_rad)
    if handle_unlock_delta_rad <= 0.0 or door_goal_delta_rad <= 0.0:
        raise ValueError("Interaction progress denominators must be positive.")

    contact = np.zeros(contact_both.shape[0], dtype=np.float32)
    run_length = 0
    for idx, active in enumerate(contact_both > 0.5):
        run_length = run_length + 1 if bool(active) else 0
        contact[idx] = float(run_length >= min_frames)

    handle_closed = float(door_dof_pos[0, 1] if handle_closed_angle is None else handle_closed_angle)
    door_closed = float(door_dof_pos[0, 0] if door_closed_angle is None else door_closed_angle)
    raw_handle = np.clip(
        np.abs(door_dof_pos[:, 1] - handle_closed) / handle_unlock_delta_rad,
        0.0,
        1.0,
    )
    raw_door = np.clip(
        np.abs(door_dof_pos[:, 0] - door_closed) / door_goal_delta_rad,
        0.0,
        1.0,
    )
    # Physics limits commonly stop a few microradians short of the nominal
    # target; snap numerically equivalent completion values to exactly one.
    raw_handle[raw_handle >= 1.0 - 1.0e-4] = 1.0
    raw_door[raw_door >= 1.0 - 1.0e-4] = 1.0
    handle_progress = np.maximum.accumulate(raw_handle).astype(np.float32, copy=False)
    door_progress = np.maximum.accumulate(raw_door).astype(np.float32, copy=False)
    return {
        INTERACTION_CONTACT_FEATURE: contact.reshape(-1, 1),
        INTERACTION_HANDLE_PROGRESS_FEATURE: handle_progress.reshape(-1, 1),
        INTERACTION_DOOR_PROGRESS_FEATURE: door_progress.reshape(-1, 1),
    }


def first_phase_index(phase_ids, phase_names, target_phase_names):
    phase_names = _to_str_list(phase_names)
    if not phase_names:
        return None
    phase_to_id = {name: idx for idx, name in enumerate(phase_names)}
    target_ids = {phase_to_id[name] for name in target_phase_names if name in phase_to_id}
    if not target_ids:
        return None
    phase_ids = np.asarray(phase_ids, dtype=np.int64).reshape(-1)
    matches = np.nonzero(np.isin(phase_ids, list(target_ids)))[0]
    if matches.size <= 0:
        return None
    return int(matches[0])


def extract_door_keyframes_from_phase_ids(phase_ids, phase_names):
    """Extract semantic keyframe indices from recorded phase ids.

    Returned keyframes follow the scripted A2W/B1Z1 door sequence:
    start, stop before door, pregrasp, grasp, rotate.
    """

    phase_ids = np.asarray(phase_ids, dtype=np.int64).reshape(-1)
    keyframe_names = []
    keyframe_indices = []
    keyframe_target_phase_names = []
    if phase_ids.size <= 0:
        return keyframe_indices, keyframe_names, keyframe_target_phase_names

    for name in DEFAULT_KEYFRAME_NAMES:
        targets = DEFAULT_KEYFRAME_PHASE_TARGETS.get(name, ())
        if name == "start":
            idx = 0
        else:
            idx = first_phase_index(phase_ids, phase_names, targets)
            if idx is None and name == "grasp":
                # With grasp_hold_steps=0 the first close_gripper frame is the
                # grasp keyframe. If both are missing, fall back to the last
                # grasp frame, which is still the closest recorded grasp sample.
                idx = first_phase_index(phase_ids, phase_names, ("grasp",))
                if idx is not None:
                    grasp_matches = np.nonzero(
                        np.asarray(phase_ids, dtype=np.int64)
                        == _to_str_list(phase_names).index("grasp")
                    )[0]
                    if grasp_matches.size > 0:
                        idx = int(grasp_matches[-1])
            if idx is None and name == "rotate":
                rotate_start = first_phase_index(phase_ids, phase_names, ("rotate_handle",))
                if rotate_start is not None:
                    rotate_id = _to_str_list(phase_names).index("rotate_handle")
                    rotate_matches = np.nonzero(np.asarray(phase_ids, dtype=np.int64) == rotate_id)[0]
                    idx = int(rotate_matches[-1]) if rotate_matches.size > 0 else rotate_start
        if idx is None:
            continue
        keyframe_names.append(name)
        keyframe_indices.append(int(idx))
        keyframe_target_phase_names.append(",".join(targets) if targets else "first_frame")

    # Keep deterministic order and drop duplicate frame indices while preserving
    # the first semantic name assigned to that frame.
    dedup_indices = []
    dedup_names = []
    dedup_targets = []
    seen = set()
    for idx, name, target in sorted(zip(keyframe_indices, keyframe_names, keyframe_target_phase_names), key=lambda item: item[0]):
        if idx in seen:
            continue
        seen.add(idx)
        dedup_indices.append(int(idx))
        dedup_names.append(str(name))
        dedup_targets.append(str(target))
    return dedup_indices, dedup_names, dedup_targets


def sliding_mean_displacement(values, window=5):
    """Mean joint-/signal-space displacement over a trailing sliding window.

    This implements the form used for motion keyframes:

        δ̄_t = 1 / w * Σ_i ||q_{t-i} - q_{t-i-1}||₂

    For the first frames where a full window is not available, the average is
    computed over the available prefix.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.size <= 0:
        return np.zeros((0,), dtype=np.float64)
    n = int(values.shape[0])
    window = max(1, int(window))
    diff = np.zeros(n, dtype=np.float64)
    if n > 1:
        diff[1:] = np.linalg.norm(np.diff(np.nan_to_num(values), axis=0), axis=1)
    cumsum = np.cumsum(np.concatenate([np.zeros(1, dtype=np.float64), diff]))
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        start = max(0, t - window + 1)
        denom = max(1, t - start + 1)
        out[t] = (cumsum[t + 1] - cumsum[start]) / denom
    return out


def _raw_has(raw, key):
    if raw is None:
        return False
    if hasattr(raw, "files"):
        return key in raw.files
    return key in raw


def _raw_get(raw, key, default=None):
    if not _raw_has(raw, key):
        return default
    return raw[key]


def _phase_names_for_raw(raw, phase_ids=None, phase_names=None):
    names = _to_str_list(phase_names)
    if names:
        return names
    if _raw_has(raw, "phase_names"):
        names = _to_str_list(_raw_get(raw, "phase_names"))
        if names:
            return names
    if phase_ids is not None:
        phase_ids = np.asarray(phase_ids, dtype=np.int64).reshape(-1)
        if phase_ids.size > 0 and int(np.max(phase_ids)) < len(DEFAULT_A2W_PHASE_NAMES):
            return list(DEFAULT_A2W_PHASE_NAMES)
    return []


def _find_salient_peaks(metric, quantile=0.95, min_separation=15, max_count=4, min_threshold=0.0):
    metric = np.nan_to_num(np.asarray(metric, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    n = metric.size
    if n <= 0 or max_count <= 0:
        return []
    threshold = max(float(min_threshold), float(np.quantile(metric, float(quantile))))
    if threshold <= 0.0:
        return []

    # Smooth scripted trajectories often create a broad high-motion plateau
    # instead of a sharp local maximum. Select top separated frames over the
    # whole thresholded region so those plateaus still produce representatives.
    candidates = np.nonzero(metric >= threshold)[0]
    if candidates.size <= 0:
        return []

    selected = []
    min_separation = max(1, int(min_separation))
    for idx in sorted(candidates.tolist(), key=lambda i: float(metric[i]), reverse=True):
        if all(abs(int(idx) - int(prev)) >= min_separation for prev in selected):
            selected.append(int(idx))
        if len(selected) >= int(max_count):
            break
    return sorted(selected)


def _append_keyframe(entries, idx, name, rule, n=None):
    if idx is None:
        return
    idx = int(idx)
    if n is not None and not (0 <= idx < int(n)):
        return
    entries.append((idx, str(name), str(rule)))


def _first_gripper_contact_index(raw, phase_ids=None, phase_names=None):
    """Infer gripper-handle contact from raw signals.

    Isaac Gym contact forces were not stored in the raw episodes. The most
    reliable recorded proxy is the instant the gripper starts closing after the
    grasp point. Prefer the scripted close_gripper phase if phase ids are
    available; otherwise use the first significant positive gripper target
    change.
    """

    phase_names = _phase_names_for_raw(raw, phase_ids=phase_ids, phase_names=phase_names)
    if phase_ids is not None:
        idx = first_phase_index(phase_ids, phase_names, ("close_gripper",))
        if idx is not None:
            return int(idx)
        idx = first_phase_index(phase_ids, phase_names, ("grasp_hold",))
        if idx is not None:
            return int(idx)

    gripper = None
    action = _raw_get(raw, "action")
    if action is not None:
        action = np.asarray(action)
        if action.ndim == 2 and action.shape[1] >= 1:
            gripper = action[:, -1]
    if gripper is None:
        dof_pos = _raw_get(raw, "replay_dof_pos")
        if dof_pos is not None:
            dof_pos = np.asarray(dof_pos)
            if dof_pos.ndim == 2 and dof_pos.shape[1] >= 1:
                gripper = dof_pos[:, -1]
    if gripper is None or len(gripper) < 2:
        return None

    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    delta = np.diff(gripper, prepend=gripper[0])
    min_delta = float(DEFAULT_MOTION_KEYFRAME_CONFIG["gripper_contact_min_delta"])
    matches = np.nonzero(np.abs(delta) > min_delta)[0]
    if matches.size <= 0:
        return None
    return int(matches[0])


def extract_motion_keyframes_from_raw_arrays(raw, phase_names=None, config=None):
    """Extract dynamic keyframes from a raw Door DP episode.

    Rules:
    1. base speed changes quickly: sliding mean displacement of actual base
       velocity [vx, vy, yaw_rate].
    2. arm joints move quickly: sliding mean displacement of Z1 joint1..joint6
       positions, taken from the last 7 replay DOFs and excluding gripper.
    3. handle rotation speed changes quickly: sliding mean displacement of
       handle joint velocity.
    4. door hinge rotation speed changes quickly: sliding mean displacement of
       door hinge joint velocity.
    5. gripper-handle contact instant: first close_gripper/grasp_hold phase, or
       first significant gripper target/position change if phase ids are absent.

    The returned keyframes are the union of these dynamic events and the older
    phase-defined semantic keyframes, so start/stop/pregrasp/grasp/rotate are
    preserved whenever subtask ids are available.
    """

    cfg = dict(DEFAULT_MOTION_KEYFRAME_CONFIG)
    if config:
        cfg.update(config)

    subtasks = _raw_get(raw, "subtask_index")
    phase_ids = None
    if subtasks is not None:
        phase_ids = np.asarray(subtasks, dtype=np.int64).reshape(-1)
    names = _phase_names_for_raw(raw, phase_ids=phase_ids, phase_names=phase_names)

    n = None
    for key in ("state", "action", "replay_root_state", "replay_dof_pos", "replay_door_dof_pos"):
        value = _raw_get(raw, key)
        if value is not None:
            n = int(np.asarray(value).shape[0])
            break
    if n is None:
        n = int(phase_ids.size) if phase_ids is not None else 0

    entries = []
    if bool(cfg.get("manual_union", True)) and phase_ids is not None:
        manual_indices, manual_names, manual_targets = extract_door_keyframes_from_phase_ids(phase_ids, names)
        for idx, name, target in zip(manual_indices, manual_names, manual_targets):
            _append_keyframe(entries, idx, f"manual:{name}", f"phase:{target}", n=n)

    window = int(cfg.get("window", 5))
    min_sep = int(cfg.get("min_separation", 15))

    root_state = _raw_get(raw, "replay_root_state")
    if root_state is not None:
        root_state = np.asarray(root_state, dtype=np.float64)
        if root_state.ndim == 2 and root_state.shape[1] >= 13:
            base_vel = np.stack([root_state[:, 7], root_state[:, 8], root_state[:, 12]], axis=1)
            metric = sliding_mean_displacement(base_vel, window=window)
            for idx in _find_salient_peaks(
                metric,
                quantile=cfg.get("base_speed_change_quantile", 0.95),
                min_separation=max(min_sep, 20),
                max_count=cfg.get("base_speed_change_max_count", 4),
            ):
                _append_keyframe(entries, idx, "base_speed_change", "sliding_mean_delta([vx,vy,yaw_rate])", n=n)

    dof_pos = _raw_get(raw, "replay_dof_pos")
    if dof_pos is not None:
        dof_pos = np.asarray(dof_pos, dtype=np.float64)
        if dof_pos.ndim == 2 and dof_pos.shape[1] >= 7:
            arm_joints = dof_pos[:, -7:-1]
            metric = sliding_mean_displacement(arm_joints, window=window)
            for idx in _find_salient_peaks(
                metric,
                quantile=cfg.get("arm_joint_motion_quantile", 0.95),
                min_separation=max(min_sep, 20),
                max_count=cfg.get("arm_joint_motion_max_count", 6),
            ):
                _append_keyframe(entries, idx, "arm_joint_motion_fast", "sliding_mean_delta(joint1..joint6)", n=n)

    door_vel = _raw_get(raw, "replay_door_dof_vel")
    if door_vel is not None:
        door_vel = np.asarray(door_vel, dtype=np.float64)
        if door_vel.ndim == 2 and door_vel.shape[1] >= 2:
            handle_metric = sliding_mean_displacement(door_vel[:, 1], window=window)
            for idx in _find_salient_peaks(
                handle_metric,
                quantile=cfg.get("handle_speed_change_quantile", 0.95),
                min_separation=min_sep,
                max_count=cfg.get("handle_speed_change_max_count", 4),
            ):
                _append_keyframe(entries, idx, "handle_speed_change", "sliding_mean_delta(handle_angular_velocity)", n=n)

            door_metric = sliding_mean_displacement(door_vel[:, 0], window=window)
            for idx in _find_salient_peaks(
                door_metric,
                quantile=cfg.get("door_speed_change_quantile", 0.95),
                min_separation=min_sep,
                max_count=cfg.get("door_speed_change_max_count", 4),
            ):
                _append_keyframe(entries, idx, "door_hinge_speed_change", "sliding_mean_delta(door_angular_velocity)", n=n)

    contact_idx = _first_gripper_contact_index(raw, phase_ids=phase_ids, phase_names=names)
    _append_keyframe(entries, contact_idx, "gripper_handle_contact", "first_close_gripper_or_gripper_motion", n=n)

    # Final cleanup: collapse keyframes that are too close. Within each
    # dedup_window cluster, prefer automatically detected motion/contact events;
    # if the cluster has no automatic event, keep the manual phase keyframe.
    dedup_window = max(0, int(cfg.get("dedup_window", 10)))
    clusters = []
    for idx, name, rule in sorted(entries, key=lambda item: int(item[0])):
        item = {"idx": int(idx), "name": str(name), "rule": str(rule)}
        if clusters and item["idx"] - clusters[-1][-1]["idx"] <= dedup_window:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    def auto_priority(item):
        is_auto = not item["name"].startswith("manual:")
        # auto first, then frames that carry more semantic labels, then earlier.
        return (1 if is_auto else 0, item["name"].count("+"), -item["idx"])

    selected_groups = []
    for cluster in clusters:
        selected = max(cluster, key=auto_priority)
        ordered = [selected] + [item for item in cluster if item is not selected]
        selected_groups.append({
            "idx": selected["idx"],
            "names": [item["name"] for item in ordered],
            "rules": [item["rule"] for item in ordered],
        })

    indices = [int(group["idx"]) for group in selected_groups]
    keyframe_names = ["+".join(dict.fromkeys(group["names"])) for group in selected_groups]
    keyframe_rules = [";".join(dict.fromkeys(group["rules"])) for group in selected_groups]
    return indices, keyframe_names, keyframe_rules


def make_keyframe_action_loss_weight(
    num_frames,
    keyframe_indices,
    weight=DEFAULT_KEYFRAME_LOSS_WEIGHT,
    radius=DEFAULT_KEYFRAME_LOSS_RADIUS,
    enabled=True,
):
    weights = np.ones(int(num_frames), dtype=np.float32)
    if not enabled:
        return weights
    if num_frames <= 0:
        return weights
    weight = float(weight)
    radius = int(radius)
    if weight <= 1.0 or radius < 0:
        return weights
    keyframe_indices = np.asarray(keyframe_indices, dtype=np.int64).reshape(-1)
    if keyframe_indices.size <= 0:
        return weights
    frames = np.arange(int(num_frames), dtype=np.int64)
    near = np.min(np.abs(frames[:, None] - keyframe_indices[None, :]), axis=1) <= radius
    weights[near] = weight
    return weights


class DoorDPLeRobotRecorder:
    def __init__(
        self,
        root,
        repo_id,
        fps,
        state_feature_names,
        task,
        resume=True,
        vision_mode="depth",
        metadata=None,
        image_storage="video",
        video_codec="h264",
        action_feature_names=None,
        include_action_loss_weight=False,
        include_recovery_indicator=False,
        include_camera_pose=False,
        include_handle_latent=False,
        include_end_signal=False,
        include_interaction_state=False,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = int(fps)
        self.task = task
        self.vision_mode = normalize_vision_mode(vision_mode)
        self.image_storage = str(image_storage).lower()
        if self.image_storage not in ("video", "image"):
            raise ValueError(f"Unsupported image_storage={image_storage!r}; expected 'video' or 'image'.")
        self.video_codec = str(video_codec)
        self.state_feature_names = list(state_feature_names)
        self.action_names = list(action_feature_names or ACTION_NAMES)
        self.include_action_loss_weight = bool(include_action_loss_weight)
        self.include_recovery_indicator = bool(include_recovery_indicator)
        self.include_camera_pose = bool(include_camera_pose)
        self.include_handle_latent = bool(include_handle_latent)
        self.include_end_signal = bool(include_end_signal)
        self.include_interaction_state = bool(include_interaction_state)
        self.metadata = dict(metadata or {})
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_root = self.root / repo_id
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_feature_names),),
                "names": self.state_feature_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.action_names),),
                "names": self.action_names,
            },
            "subtask_index": {"dtype": "int64", "shape": (1,), "names": ["subtask_index"]},
        }
        if self.include_action_loss_weight:
            features[ACTION_LOSS_WEIGHT_FEATURE] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["action_loss_weight"],
            }
        if self.include_recovery_indicator:
            features[RECOVERY_INDICATOR_FEATURE] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["is_recovery"],
            }
        if self.include_end_signal:
            features[END_SIGNAL_FEATURE] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["end_signal"],
            }
        if self.include_interaction_state:
            for key, name in (
                (INTERACTION_CONTACT_FEATURE, "contact"),
                (INTERACTION_HANDLE_PROGRESS_FEATURE, "handle_progress"),
                (INTERACTION_DOOR_PROGRESS_FEATURE, "door_progress"),
            ):
                features[key] = {"dtype": "float32", "shape": (1,), "names": [name]}
        if self.include_camera_pose:
            for key in CAMERA_POSE_FEATURES:
                features[key] = {
                    "dtype": "float32",
                    "shape": (len(CAMERA_POSE_NAMES),),
                    "names": CAMERA_POSE_NAMES,
                }
        if self.include_handle_latent:
            features[FRONT_HANDLE_LATENT_FEATURE] = {
                "dtype": "float32",
                "shape": (384,),
                "names": [f"dino_patch_mean_{i}" for i in range(384)],
            }
            features[FRONT_HANDLE_LATENT_VALID_FEATURE] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["valid"],
            }
            features[WRIST_HANDLE_LATENT_FEATURE] = {
                "dtype": "float32",
                "shape": (384,),
                "names": [f"dino_patch_mean_{i}" for i in range(384)],
            }
            features[WRIST_HANDLE_LATENT_VALID_FEATURE] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["valid"],
            }
        for key in lerobot_image_keys_for_vision_mode(self.vision_mode):
            features[key] = {
                "dtype": self.image_storage,
                "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "names": ["height", "width", "channels"],
            }
        if resume and self.dataset_root.exists():
            try:
                self.dataset = LeRobotDataset(repo_id=repo_id, root=str(self.dataset_root))
            except TypeError:
                self.dataset = LeRobotDataset(repo_id, root=str(self.dataset_root))
        else:
            try:
                self.dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    root=str(self.dataset_root),
                    fps=self.fps,
                    features=features,
                    use_videos=self.image_storage == "video",
                    vcodec=self.video_codec,
                )
            except TypeError:
                self.dataset = LeRobotDataset.create(repo_id, fps=self.fps, root=str(self.dataset_root), features=features)
        self.frame_count = 0
        self.episode_count = 0
        self._write_feature_sidecar()

    def _write_feature_sidecar(self):
        sidecar = {
            "state": self.state_feature_names,
            "action": self.action_names,
            "image_features": lerobot_image_keys_for_vision_mode(self.vision_mode),
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "image_storage": self.image_storage,
            "video_codec": self.video_codec,
        }
        if self.include_action_loss_weight:
            sidecar["action_loss_weight_feature"] = ACTION_LOSS_WEIGHT_FEATURE
        if self.include_recovery_indicator:
            sidecar["recovery_indicator_feature"] = RECOVERY_INDICATOR_FEATURE
        if self.include_end_signal:
            sidecar["end_signal_feature"] = END_SIGNAL_FEATURE
        if self.include_interaction_state:
            sidecar["interaction_state_features"] = list(INTERACTION_STATE_FEATURES)
        if self.include_camera_pose:
            sidecar["camera_pose_features"] = CAMERA_POSE_FEATURES
            sidecar.setdefault("camera_pose_frame", "robot_base")
            sidecar.setdefault("camera_pose_convention", "optical_frame")
        if self.include_handle_latent:
            sidecar["handle_latent_features"] = HANDLE_LATENT_FEATURES
        if self.vision_mode != "depth":
            sidecar["vision_mode"] = self.vision_mode
        for key in DATASET_METADATA_KEYS:
            if hasattr(self, "metadata") and key in self.metadata:
                sidecar[key] = self.metadata[key]
        out = self.dataset_root / "door_dp_feature_names.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

    def add_frame(
        self,
        state,
        wrist_mask_rgb,
        wrist_second_rgb,
        action,
        subtask_index,
        front_mask_rgb=None,
        front_second_rgb=None,
        action_loss_weight=None,
        is_recovery=None,
        front_camera_pose_base=None,
        wrist_camera_pose_base=None,
        front_handle_latent=None,
        front_handle_latent_valid=None,
        wrist_handle_latent=None,
        wrist_handle_latent_valid=None,
        end_signal=None,
        interaction_contact=None,
        interaction_handle_progress=None,
        interaction_door_progress=None,
    ):
        if self.vision_mode == "depth":
            front_mask_rgb = _zero_image_like(wrist_mask_rgb) if front_mask_rgb is None else front_mask_rgb
            front_second_rgb = _zero_image_like(wrist_second_rgb) if front_second_rgb is None else front_second_rgb
        elif self.vision_mode == "depth_only":
            front_second_rgb = _zero_image_like(wrist_second_rgb) if front_second_rgb is None else front_second_rgb
        elif front_mask_rgb is None or front_second_rgb is None:
            raise ValueError("RGB Door DP LeRobot frames require wrist/front RGB and mask images.")
        image_keys = lerobot_image_keys_for_vision_mode(self.vision_mode)
        frame = {
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "subtask_index": np.asarray([subtask_index], dtype=np.int64),
            "task": self.task,
        }
        if self.include_action_loss_weight:
            weight = 1.0 if action_loss_weight is None else float(np.asarray(action_loss_weight).reshape(-1)[0])
            frame[ACTION_LOSS_WEIGHT_FEATURE] = np.asarray([weight], dtype=np.float32)
        if self.include_recovery_indicator:
            value = 0.0 if is_recovery is None else float(np.asarray(is_recovery).reshape(-1)[0])
            frame[RECOVERY_INDICATOR_FEATURE] = np.asarray([value], dtype=np.float32)
        if self.include_end_signal:
            value = 0.0 if end_signal is None else float(np.asarray(end_signal).reshape(-1)[0])
            frame[END_SIGNAL_FEATURE] = np.asarray([value], dtype=np.float32)
        if self.include_interaction_state:
            values = {
                INTERACTION_CONTACT_FEATURE: interaction_contact,
                INTERACTION_HANDLE_PROGRESS_FEATURE: interaction_handle_progress,
                INTERACTION_DOOR_PROGRESS_FEATURE: interaction_door_progress,
            }
            for key, value in values.items():
                if value is None:
                    raise ValueError(f"LeRobot interaction-state frame requires {key!r}.")
                scalar = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
                if not np.isfinite(scalar) or scalar < 0.0 or scalar > 1.0:
                    raise ValueError(f"Invalid {key}={scalar}; expected a finite value in [0, 1].")
                frame[key] = np.asarray([scalar], dtype=np.float32)
        if self.include_camera_pose:
            if front_camera_pose_base is None or wrist_camera_pose_base is None:
                raise ValueError("LeRobot camera-pose dataset frames require both front and wrist camera poses.")
            frame[FRONT_CAMERA_POSE_FEATURE] = np.asarray(front_camera_pose_base, dtype=np.float32).reshape(7)
            frame[WRIST_CAMERA_POSE_FEATURE] = np.asarray(wrist_camera_pose_base, dtype=np.float32).reshape(7)
        if self.include_handle_latent:
            if front_handle_latent is None or wrist_handle_latent is None:
                raise ValueError("LeRobot handle-latent frames require both front and wrist latent arrays.")
            frame[FRONT_HANDLE_LATENT_FEATURE] = np.asarray(front_handle_latent, dtype=np.float32).reshape(384)
            frame[FRONT_HANDLE_LATENT_VALID_FEATURE] = np.asarray(
                [0.0 if front_handle_latent_valid is None else float(np.asarray(front_handle_latent_valid).reshape(-1)[0])],
                dtype=np.float32,
            )
            frame[WRIST_HANDLE_LATENT_FEATURE] = np.asarray(wrist_handle_latent, dtype=np.float32).reshape(384)
            frame[WRIST_HANDLE_LATENT_VALID_FEATURE] = np.asarray(
                [0.0 if wrist_handle_latent_valid is None else float(np.asarray(wrist_handle_latent_valid).reshape(-1)[0])],
                dtype=np.float32,
            )
        if self.vision_mode == "depth_only":
            frame[image_keys[0]] = np.asarray(wrist_second_rgb, dtype=np.uint8)
            frame[image_keys[1]] = np.asarray(front_second_rgb, dtype=np.uint8)
        else:
            frame[image_keys[0]] = np.asarray(wrist_mask_rgb, dtype=np.uint8)
            frame[image_keys[1]] = np.asarray(wrist_second_rgb, dtype=np.uint8)
            frame[image_keys[2]] = np.asarray(front_mask_rgb, dtype=np.uint8)
            frame[image_keys[3]] = np.asarray(front_second_rgb, dtype=np.uint8)
        try:
            self.dataset.add_frame(frame)
        except TypeError:
            frame_without_task = dict(frame)
            frame_without_task.pop("task", None)
            self.dataset.add_frame(frame_without_task, task=self.task)
        self.frame_count += 1

    def save_episode(self):
        try:
            self.dataset.save_episode(task=self.task)
        except TypeError:
            self.dataset.save_episode()
        self.episode_count += 1

    def finalize(self):
        if hasattr(self.dataset, "finalize"):
            self.dataset.finalize()


class RawDoorDPRecorder:
    def __init__(
        self,
        raw_root,
        fps,
        state_feature_names,
        task,
        metadata=None,
        vision_mode="depth",
        action_feature_names=None,
    ):
        self.raw_root = Path(raw_root)
        self.fps = int(fps)
        self.task = str(task)
        self.vision_mode = normalize_vision_mode(vision_mode)
        self.image_keys = raw_image_keys_for_vision_mode(self.vision_mode)
        self.state_feature_names = list(state_feature_names)
        self.action_names = list(action_feature_names or ACTION_NAMES)
        self.metadata = dict(metadata or {})
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.frames = {
            "state": [],
            "action": [],
            "subtask_index": [],
        }
        self.include_end_signal = bool(self.metadata.get("end_signal_enabled", False))
        if self.include_end_signal:
            self.frames[RAW_END_SIGNAL_KEY] = []
        for key in self.image_keys:
            self.frames[key] = []
        self.frame_count = 0
        self.episode_count = 0
        self._write_feature_sidecar()

    def _write_feature_sidecar(self):
        sidecar = {
            "fps": self.fps,
            "state": self.state_feature_names,
            "action": self.action_names,
            "image_features": self.image_keys,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "format": "door_dp_raw_npz_v1",
        }
        if self.vision_mode == "depth_only":
            sidecar["depth_storage_channels"] = 1
        if self.vision_mode != "depth":
            sidecar["vision_mode"] = self.vision_mode
        for key in DATASET_METADATA_KEYS:
            if key in self.metadata:
                sidecar[key] = self.metadata[key]
        out = self.raw_root / "door_dp_feature_names.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

    def add_frame(
        self,
        state,
        wrist_mask_rgb,
        wrist_second_rgb,
        action,
        subtask_index,
        front_mask_rgb=None,
        front_second_rgb=None,
        replay_snapshot=None,
        end_signal=None,
    ):
        if self.vision_mode == "depth":
            front_mask_rgb = _zero_image_like(wrist_mask_rgb) if front_mask_rgb is None else front_mask_rgb
            front_second_rgb = _zero_image_like(wrist_second_rgb) if front_second_rgb is None else front_second_rgb
        elif self.vision_mode == "depth_only":
            front_second_rgb = _zero_image_like(wrist_second_rgb) if front_second_rgb is None else front_second_rgb
        elif front_mask_rgb is None or front_second_rgb is None:
            raise ValueError("RGB Door DP raw frames require wrist/front RGB and mask images.")
        self.frames["state"].append(np.asarray(state, dtype=np.float32).copy())
        self.frames["action"].append(np.asarray(action, dtype=np.float32).copy())
        if self.vision_mode == "depth_only":
            self.frames[self.image_keys[0]].append(depth_image_to_single_channel_uint8(wrist_second_rgb))
            self.frames[self.image_keys[1]].append(depth_image_to_single_channel_uint8(front_second_rgb))
        else:
            self.frames[self.image_keys[0]].append(np.asarray(wrist_mask_rgb, dtype=np.uint8).copy())
            self.frames[self.image_keys[1]].append(np.asarray(wrist_second_rgb, dtype=np.uint8).copy())
            self.frames[self.image_keys[2]].append(np.asarray(front_mask_rgb, dtype=np.uint8).copy())
            self.frames[self.image_keys[3]].append(np.asarray(front_second_rgb, dtype=np.uint8).copy())
        self.frames["subtask_index"].append(np.asarray([subtask_index], dtype=np.int64).copy())
        if self.include_end_signal:
            value = 0.0 if end_signal is None else float(np.asarray(end_signal).reshape(-1)[0])
            self.frames[RAW_END_SIGNAL_KEY].append(np.asarray([value], dtype=np.float32))
        if replay_snapshot:
            for key, value in replay_snapshot.items():
                self.frames.setdefault(key, []).append(np.asarray(value, dtype=np.float32).copy())
        self.frame_count += 1

    def _next_episode_path(self):
        existing = sorted(self.raw_root.glob("episode_*.npz"))
        if not existing:
            return self.raw_root / "episode_000000.npz"
        max_idx = -1
        for path in existing:
            try:
                max_idx = max(max_idx, int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        return self.raw_root / f"episode_{max_idx + 1:06d}.npz"

    def save_episode(self):
        if self.frame_count <= 0:
            print("Warning: RawDoorDPRecorder has no frames; skipped saving episode.")
            return
        out = self._next_episode_path()
        subtasks = np.stack(self.frames["subtask_index"], axis=0).astype(np.int64, copy=False)
        phase_names = _to_str_list(self.metadata.get("phase_names", []))
        keyframe_raw = {"subtask_index": subtasks}
        for key in (
            "state",
            "action",
            "replay_root_state",
            "replay_dof_pos",
            "replay_dof_vel",
            "replay_door_dof_pos",
            "replay_door_dof_vel",
            "replay_ee_pos",
            "replay_ee_quat",
        ):
            values = self.frames.get(key)
            if values and len(values) == self.frame_count:
                keyframe_raw[key] = np.stack(values, axis=0)
        keyframe_indices, keyframe_names, keyframe_target_phase_names = extract_motion_keyframes_from_raw_arrays(
            keyframe_raw,
            phase_names=phase_names,
        )
        keyframe_loss_enabled = bool(self.metadata.get("keyframe_loss_enabled", True))
        keyframe_loss_weight = float(
            self.metadata.get("keyframe_loss_weight", DEFAULT_KEYFRAME_LOSS_WEIGHT)
        )
        keyframe_loss_radius = int(
            self.metadata.get("keyframe_loss_radius", DEFAULT_KEYFRAME_LOSS_RADIUS)
        )
        action_loss_weight = make_keyframe_action_loss_weight(
            self.frame_count,
            keyframe_indices,
            weight=keyframe_loss_weight,
            radius=keyframe_loss_radius,
            enabled=keyframe_loss_enabled,
        )
        keyframe_mask = np.zeros(self.frame_count, dtype=np.uint8)
        if keyframe_indices:
            valid_indices = np.asarray(keyframe_indices, dtype=np.int64)
            valid_indices = valid_indices[(0 <= valid_indices) & (valid_indices < self.frame_count)]
            keyframe_mask[valid_indices] = 1
        payload = {
            "state": np.stack(self.frames["state"], axis=0).astype(np.float32, copy=False),
            "action": np.stack(self.frames["action"], axis=0).astype(np.float32, copy=False),
            "subtask_index": subtasks,
            RAW_ACTION_LOSS_WEIGHT_KEY: action_loss_weight.reshape(-1, 1).astype(np.float32, copy=False),
            "keyframe_mask": keyframe_mask.reshape(-1, 1).astype(np.uint8, copy=False),
            "keyframe_indices": np.asarray(keyframe_indices, dtype=np.int64),
            "keyframe_names": np.asarray(keyframe_names, dtype=object),
            "keyframe_target_phase_names": np.asarray(keyframe_target_phase_names, dtype=object),
            "keyframe_extraction_rules": np.asarray(keyframe_target_phase_names, dtype=object),
            "task": np.asarray(self.task),
            "fps": np.asarray(self.fps, dtype=np.int64),
            "state_feature_names": np.asarray(self.state_feature_names, dtype=object),
            "action_names": np.asarray(self.action_names, dtype=object),
            "phase_names": np.asarray(phase_names, dtype=object),
            "keyframe_loss_enabled": np.asarray(keyframe_loss_enabled),
            "keyframe_loss_weight": np.asarray(keyframe_loss_weight, dtype=np.float32),
            "keyframe_loss_radius": np.asarray(keyframe_loss_radius, dtype=np.int64),
            "keyframe_loss_feature": np.asarray(ACTION_LOSS_WEIGHT_FEATURE),
        }
        if self.include_end_signal:
            payload[RAW_END_SIGNAL_KEY] = np.stack(
                self.frames[RAW_END_SIGNAL_KEY], axis=0
            ).astype(np.float32, copy=False)
        if self.vision_mode != "depth":
            payload["vision_mode"] = np.asarray(self.vision_mode)
        if self.vision_mode == "depth_only":
            payload["depth_storage_channels"] = np.asarray(1, dtype=np.int64)
        for key in self.image_keys:
            payload[key] = np.stack(self.frames[key], axis=0).astype(np.uint8, copy=False)
        for key, value in self.metadata.items():
            if key not in payload:
                payload[key] = np.asarray(value)
        for key, values in self.frames.items():
            if key in payload or not values or len(values) != self.frame_count:
                continue
            payload[key] = np.stack(values, axis=0).astype(np.float32, copy=False)
        np.savez_compressed(out, **payload)
        self.episode_count += 1
        print(f"Saved raw Door DP episode: {out}")
        self._clear_frames()

    def finalize(self):
        self._clear_frames()

    def _clear_frames(self):
        for values in self.frames.values():
            values.clear()
        self.frame_count = 0


def import_lerobot_or_raise():
    try:
        import lerobot  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "LeRobot is required for this DP dataset pipeline. Install dependencies with "
            "`pip install -r high-level/dp/requirements_dp.txt` inside a Python>=3.10 training environment."
        ) from exc


class DoorDPPolicyController:
    def __init__(
        self,
        checkpoint,
        device=None,
        num_inference_steps=10,
        action_horizon=None,
        noise_scheduler_type="DDIM",
    ):
        try:
            from .door_policy_backend import DoorPolicyController, DoorPolicySubprocessController
        except ImportError:
            from door_policy_backend import DoorPolicyController, DoorPolicySubprocessController
        try:
            self._impl = DoorPolicyController(
                checkpoint,
                device=device,
                num_inference_steps=num_inference_steps,
                action_horizon=action_horizon,
                noise_scheduler_type=noise_scheduler_type,
            )
        except RuntimeError as exc:
            self._impl = DoorPolicySubprocessController(
                checkpoint,
                device=device,
                num_inference_steps=num_inference_steps,
                action_horizon=action_horizon,
                noise_scheduler_type=noise_scheduler_type,
                startup_error=exc,
            )

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def reset(self):
        return self._impl.reset()

    def reset_envs(self, env_ids=None):
        return self._impl.reset_envs(env_ids)

    def _normalize_state(self, state):
        return self._impl._normalize_state(state)

    def append_observation(
        self,
        state,
        mask_rgb,
        masked_depth_rgb,
        front_mask_rgb=None,
        front_masked_depth_rgb=None,
        front_camera_pose_base=None,
        wrist_camera_pose_base=None,
    ):
        return self._impl.append_observation(
            state,
            mask_rgb,
            masked_depth_rgb,
            front_mask_rgb,
            front_masked_depth_rgb,
            front_camera_pose_base,
            wrist_camera_pose_base,
        )

    def append_observation_for_env(
        self,
        env_id,
        state,
        mask_rgb,
        masked_depth_rgb,
        front_mask_rgb=None,
        front_masked_depth_rgb=None,
        front_camera_pose_base=None,
        wrist_camera_pose_base=None,
    ):
        return self._impl.append_observation_for_env(
            env_id,
            state,
            mask_rgb,
            masked_depth_rgb,
            front_mask_rgb,
            front_masked_depth_rgb,
            front_camera_pose_base,
            wrist_camera_pose_base,
        )

    def _denormalize_action(self, action):
        return self._impl._denormalize_action(action)

    @torch.no_grad()
    def sample_action_chunk(self, noise=None):
        return self._impl.sample_action_chunk(noise=noise)

    def act(
        self,
        state,
        mask_rgb,
        masked_depth_rgb,
        front_mask_rgb=None,
        front_masked_depth_rgb=None,
        front_camera_pose_base=None,
        wrist_camera_pose_base=None,
    ):
        return self._impl.act(
            state,
            mask_rgb,
            masked_depth_rgb,
            front_mask_rgb,
            front_masked_depth_rgb,
            front_camera_pose_base,
            wrist_camera_pose_base,
        )

    def act_batch(
        self,
        env_ids,
        states,
        mask_rgbs,
        masked_depth_rgbs,
        front_mask_rgbs=None,
        front_masked_depth_rgbs=None,
        front_camera_pose_bases=None,
        wrist_camera_pose_bases=None,
    ):
        return self._impl.act_batch(
            env_ids,
            states,
            mask_rgbs,
            masked_depth_rgbs,
            front_mask_rgbs,
            front_masked_depth_rgbs,
            front_camera_pose_bases,
            wrist_camera_pose_bases,
        )

    def predict_action_chunks_for_envs(self, env_ids, noise=None):
        return self._impl.predict_action_chunks_for_envs(env_ids, noise=noise)


def apply_door_dp_action(env, action, env_id=0, delta_rpy_fn=None):
    env_id = int(env_id)
    action = np.asarray(action, dtype=np.float32)
    quat = torch.as_tensor(action[5:9], dtype=torch.float32, device=env.device)
    quat = quat / torch.clamp(torch.norm(quat), min=1e-6)
    env.commands[env_id, 0] = float(action[0])
    env.commands[env_id, 1] = 0.0
    env.commands[env_id, 2] = float(action[1])
    env.curr_ee_goal_cart_world[env_id, :3] = torch.as_tensor(action[2:5], dtype=torch.float32, device=env.device)
    env.ee_goal_orn_quat[env_id, :4] = quat
    if delta_rpy_fn is not None:
        target_pos = env.curr_ee_goal_cart_world[env_id : env_id + 1]
        target_quat = env.ee_goal_orn_quat[env_id : env_id + 1]
        env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
        try:
            delta_rpy = delta_rpy_fn(target_pos, target_quat, env_ids=env_ids)
        except TypeError:
            delta_rpy = delta_rpy_fn(target_pos, target_quat)
        env.ee_goal_orn_delta_rpy[env_id : env_id + 1] = delta_rpy
    env.external_gripper_target[env_id, :] = float(action[9])
    if hasattr(env, "freeze_arm_default"):
        env.freeze_arm_default[env_id] = False
    if hasattr(env, "freeze_arm_zero"):
        env.freeze_arm_zero[env_id] = False
