import argparse
import json
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    from .door_dp_common import (
        ACTION_NAMES,
        ACTION_LOSS_WEIGHT_FEATURE,
        DATASET_METADATA_KEYS,
        DEFAULT_KEYFRAME_LOSS_RADIUS,
        DEFAULT_KEYFRAME_LOSS_WEIGHT,
        DEFAULT_NEAR_ZERO_RATE_EPS,
        DoorDPLeRobotRecorder,
        RAW_ACTION_LOSS_WEIGHT_KEY,
        apply_door_dp_action_preprocess,
        apply_door_dp_state_preprocess,
        extract_motion_keyframes_from_raw_arrays,
        fit_door_dp_action_preprocess,
        fit_door_dp_state_preprocess,
        image_to_three_channel_uint8,
        lerobot_image_keys_for_vision_mode,
        make_keyframe_action_loss_weight,
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
        DATASET_METADATA_KEYS,
        DEFAULT_KEYFRAME_LOSS_RADIUS,
        DEFAULT_KEYFRAME_LOSS_WEIGHT,
        DEFAULT_NEAR_ZERO_RATE_EPS,
        DoorDPLeRobotRecorder,
        RAW_ACTION_LOSS_WEIGHT_KEY,
        apply_door_dp_action_preprocess,
        apply_door_dp_state_preprocess,
        extract_motion_keyframes_from_raw_arrays,
        fit_door_dp_action_preprocess,
        fit_door_dp_state_preprocess,
        image_to_three_channel_uint8,
        lerobot_image_keys_for_vision_mode,
        make_keyframe_action_loss_weight,
        make_door_dp_sanitize_config,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
        sanitize_door_dp_action,
        sanitize_door_dp_state,
    )


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw Door DP .npz episodes into a local LeRobotDataset.")
    parser.add_argument("--raw_root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp"))
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rgb", action="store_true", help="Convert raw RGB+mask Door DP data. Required for RGB raw data.")
    parser.add_argument("--depth_only", action="store_true", help="Convert raw depth-only Door DP data with no mask image fields.")
    parser.add_argument(
        "--image_storage",
        choices=["video", "image"],
        default="video",
        help="Store visual observations as LeRobot v3 videos by default; use 'image' for embedded parquet images.",
    )
    parser.add_argument("--video_codec", type=str, default="h264", help="Video codec used when --image_storage video.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=(
            "Number of worker threads used to preload/validate raw npz episodes. "
            "LeRobot writing stays ordered in the main process. Default 1 preserves the old serial path."
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
    return parser.parse_args()


def load_sidecar(raw_root):
    sidecar = Path(raw_root) / "door_dp_feature_names.json"
    if not sidecar.exists():
        return None
    with open(sidecar, "r", encoding="utf-8") as f:
        return json.load(f)


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


def array_to_str_list(value):
    return [str(item) for item in np.asarray(value, dtype=object).reshape(-1).tolist()]


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
            states = data["state"].astype(np.float32)
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
            actions = data["action"].astype(np.float32)
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
    keep_state_indices,
    action_frame,
    ikpush_state_version,
    controller_mode,
    vision_mode,
    initial_task,
    keyframe_loss_weight_override=None,
    keyframe_loss_radius_override=None,
):
    with np.load(path, allow_pickle=True) as data:
        validate_episode_metadata(path, data, sidecar, action_frame, ikpush_state_version, controller_mode)
        task = scalar_str(data["task"]) if "task" in data else initial_task
        states = data["state"].astype(np.float32)
        if keep_state_indices and states.shape[-1] <= max(keep_state_indices):
            raise ValueError(f"Episode {path} has state_dim={states.shape[-1]}, cannot apply selected state columns.")
        states = states[:, keep_state_indices]
        actions = data["action"].astype(np.float32)
        if vision_mode == "depth_only":
            require_fields(data, image_keys, path)
            wrist_second = data[image_keys[0]].astype(np.uint8)
            front_second = data[image_keys[1]].astype(np.uint8)
            wrist_first = np.zeros_like(wrist_second)
            front_first = np.zeros_like(front_second)
        else:
            require_fields(data, image_keys[:2], path)
            wrist_first = data[image_keys[0]].astype(np.uint8)
            wrist_second = data[image_keys[1]].astype(np.uint8)
        if vision_mode == "rgb":
            require_fields(data, image_keys[2:], path)
            front_first = data[image_keys[2]].astype(np.uint8)
            front_second = data[image_keys[3]].astype(np.uint8)
        elif vision_mode == "depth":
            front_first = data[image_keys[2]].astype(np.uint8) if image_keys[2] in data else np.zeros_like(wrist_first)
            front_second = data[image_keys[3]].astype(np.uint8) if image_keys[3] in data else np.zeros_like(wrist_second)
        subtasks = data["subtask_index"].astype(np.int64).reshape(-1)
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
        == n
    ):
        raise ValueError(f"Episode {path} has inconsistent lengths.")
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
    if args.keyframe_loss_weight is not None and args.keyframe_loss_weight <= 0.0:
        raise ValueError("--keyframe_loss_weight must be > 0")
    if args.keyframe_loss_radius is not None and args.keyframe_loss_radius < 0:
        raise ValueError("--keyframe_loss_radius must be >= 0")
    raw_root = Path(args.raw_root)
    files = episode_files(raw_root)
    sidecar = load_sidecar(raw_root)
    first = np.load(files[0], allow_pickle=True)
    if sidecar and "state" in sidecar:
        state_names = list(sidecar["state"])
    elif "state_feature_names" in first:
        state_names = [str(x) for x in first["state_feature_names"].tolist()]
    else:
        state_names = [f"state_{i}" for i in range(first["state"].shape[-1])]
    action_names = detect_action_names(first, sidecar)
    if first["action"].shape[-1] != len(action_names):
        raise ValueError(
            f"Raw action_dim={first['action'].shape[-1]} does not match action_names={len(action_names)}: "
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
    if args.rgb and args.depth_only:
        raise ValueError("--rgb and --depth_only are mutually exclusive.")
    vision_mode = "rgb" if args.rgb else ("depth_only" if args.depth_only else "depth")
    raw_vision_mode = detect_raw_vision_mode(first, sidecar)
    action_frame = detect_action_frame(first, sidecar)
    ikpush_state_version = detect_ikpush_state_version(first, sidecar)
    controller_mode = detect_controller_mode(first, sidecar)
    if action_frame not in ("world", "base"):
        raise ValueError(f"Unsupported raw action_frame={action_frame!r}; expected 'world' or 'base'.")
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
                keep_state_indices,
                state_names,
                action_frame,
                ikpush_state_version,
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
                action_names,
                action_frame,
                ikpush_state_version,
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
        metadata={
            **inherited_metadata,
            "action_frame": action_frame,
            "action_pose_frame": action_frame,
            "target_pose_frame": action_frame,
            "ikpush_state_version": ikpush_state_version,
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
            "keyframe_loss_weight": converted_keyframe_loss_weight,
            "keyframe_loss_radius": converted_keyframe_loss_radius,
        },
    )
    payloads = iter_episode_payloads(
        files,
        args.num_workers,
        sidecar=sidecar,
        image_keys=image_keys,
        keep_state_indices=keep_state_indices,
        action_frame=action_frame,
        ikpush_state_version=ikpush_state_version,
        controller_mode=controller_mode,
        vision_mode=vision_mode,
        initial_task=initial_task,
        keyframe_loss_weight_override=args.keyframe_loss_weight,
        keyframe_loss_radius_override=args.keyframe_loss_radius,
    )
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
        n = payload["n"]
        for i in range(n):
            recorder.add_frame(
                states[i],
                image_to_three_channel_uint8(wrist_first[i]),
                image_to_three_channel_uint8(wrist_second[i]),
                actions[i],
                int(subtasks[i]),
                front_mask_rgb=image_to_three_channel_uint8(front_first[i]),
                front_second_rgb=image_to_three_channel_uint8(front_second[i]),
                action_loss_weight=action_loss_weight[i],
            )
        recorder.save_episode()
        print(f"Converted {payload['path_name']}: {n} frames task={task!r} ({ep_idx + 1}/{len(files)})", flush=True)
    recorder.finalize()
    feature_sidecar = Path(args.root) / args.repo_id / "door_dp_feature_names.json"
    feature_sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar_payload = {
        "fps": fps,
        "state": state_names,
        "action": action_names,
        "image_features": lerobot_image_keys_for_vision_mode(vision_mode),
        "source_raw_root": str(raw_root),
        "action_frame": action_frame,
        "action_pose_frame": action_frame,
        "target_pose_frame": action_frame,
        "ikpush_state_version": ikpush_state_version,
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
        "keyframe_loss_weight": converted_keyframe_loss_weight,
        "keyframe_loss_radius": converted_keyframe_loss_radius,
    }
    if sidecar:
        for key in DATASET_METADATA_KEYS:
            if key in sidecar and key not in sidecar_payload:
                sidecar_payload[key] = sidecar[key]
    if vision_mode != "depth":
        sidecar_payload["vision_mode"] = vision_mode
    with open(feature_sidecar, "w", encoding="utf-8") as f:
        json.dump(sidecar_payload, f, indent=2)
    print(f"Done. LeRobotDataset written to {Path(args.root) / args.repo_id}")


if __name__ == "__main__":
    main()
