import argparse
import json
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    from .convert_door_raw_to_lerobot import (
        HIGH_LEVEL_ROOT,
        detect_action_frame,
        detect_controller_mode,
        detect_ikpush_state_version,
        detect_raw_vision_mode,
        episode_files,
        fit_action_preprocess_from_episodes,
        load_sidecar,
        require_fields,
        scalar_str,
        validate_episode_metadata,
    )
    from .door_dp_common import (
        ACTION_NAMES,
        DoorDPLeRobotRecorder,
        apply_door_dp_action_preprocess,
        apply_door_dp_state_preprocess,
        fit_door_dp_state_preprocess,
        lerobot_image_keys_for_vision_mode,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
    )
except ImportError:
    from convert_door_raw_to_lerobot import (
        HIGH_LEVEL_ROOT,
        detect_action_frame,
        detect_controller_mode,
        detect_ikpush_state_version,
        detect_raw_vision_mode,
        episode_files,
        fit_action_preprocess_from_episodes,
        load_sidecar,
        require_fields,
        scalar_str,
        validate_episode_metadata,
    )
    from door_dp_common import (
        ACTION_NAMES,
        DoorDPLeRobotRecorder,
        apply_door_dp_action_preprocess,
        apply_door_dp_state_preprocess,
        fit_door_dp_state_preprocess,
        lerobot_image_keys_for_vision_mode,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
    )


PI05_STATE_NAMES = [
    "vx",
    "yaw_rate",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "ee_qw",
    "gripper",
]
LAST_COMMAND_FEATURES = [f"last_low_action_{i}" for i in range(len(PI05_STATE_NAMES))]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw Door DP .npz episodes into a LeRobotDataset whose observation.state is "
            "a 10D PI0.5-friendly command state aligned with the action semantics."
        )
    )
    parser.add_argument("--raw_root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp"))
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_pi05_state10")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rgb", action="store_true", help="Convert raw RGB+mask Door DP data. Required for RGB raw data.")
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
        help="Number of worker threads used to preload/validate raw npz episodes.",
    )
    parser.add_argument(
        "--pi05_state_source",
        choices=["last_command", "current_action"],
        default="last_command",
        help=(
            "last_command uses raw observation.state last_low_action_0..9 as the 10D state. "
            "current_action copies the current raw action into state and is mainly useful for diagnostics."
        ),
    )
    parser.add_argument(
        "--state_preprocess",
        choices=["robust_quantile", "none"],
        default="robust_quantile",
        help="Preprocess the 10D observation.state while converting raw npz to LeRobot.",
    )
    parser.add_argument("--state_quantile_low", type=float, default=0.01)
    parser.add_argument("--state_quantile_high", type=float, default=0.99)
    parser.add_argument("--state_preprocess_eps", type=float, default=1.0e-6)
    parser.add_argument(
        "--action_preprocess",
        choices=["robust_quantile", "none"],
        default="robust_quantile",
        help="Preprocess action while converting raw npz to LeRobot.",
    )
    parser.add_argument("--action_quantile_low", type=float, default=0.01)
    parser.add_argument("--action_quantile_high", type=float, default=0.99)
    parser.add_argument("--action_preprocess_eps", type=float, default=1.0e-6)
    return parser.parse_args()


def raw_state_names(first, sidecar):
    if sidecar and "state" in sidecar:
        return [str(name) for name in sidecar["state"]]
    if "state_feature_names" in first:
        return [str(x) for x in first["state_feature_names"].tolist()]
    return [f"state_{i}" for i in range(first["state"].shape[-1])]


def last_command_state_indices(state_names):
    name_to_index = {str(name): idx for idx, name in enumerate(state_names)}
    missing = [name for name in LAST_COMMAND_FEATURES if name not in name_to_index]
    if missing:
        raise ValueError(
            "Cannot build PI0.5 10D state from raw observation.state; missing columns: "
            f"{missing}. Use --pi05_state_source current_action if you intentionally want state=action."
        )
    return [name_to_index[name] for name in LAST_COMMAND_FEATURES]


def make_pi05_states(raw_states, actions, state_source, state_indices):
    if state_source == "last_command":
        if raw_states.shape[-1] <= max(state_indices):
            raise ValueError(
                f"Raw state_dim={raw_states.shape[-1]} cannot provide last-command indices {state_indices}."
            )
        return raw_states[:, state_indices].astype(np.float32)
    if state_source == "current_action":
        return actions.astype(np.float32).copy()
    raise ValueError(f"Unsupported --pi05_state_source={state_source!r}")


def fit_pi05_state_preprocess_from_episodes(
    files,
    sidecar,
    raw_state_indices,
    state_source,
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
            raw_states = data["state"].astype(np.float32)
            actions = data["action"].astype(np.float32)
            states = make_pi05_states(raw_states, actions, state_source, raw_state_indices)
            chunks.append(states)
            total_frames += int(states.shape[0])
    if not chunks:
        raise ValueError("Cannot fit PI0.5 state preprocessing without raw states.")
    config = fit_door_dp_state_preprocess(
        np.concatenate(chunks, axis=0),
        PI05_STATE_NAMES,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        eps=eps,
    )
    constant_count = int(np.asarray(config.get("constant_mask", []), dtype=bool).sum())
    print(
        f"Fitted PI0.5 state_preprocess={config['version']} frames={total_frames} "
        f"state_dim={len(PI05_STATE_NAMES)} constant_dims={constant_count} "
        f"source={state_source} quantiles=({lower_quantile}, {upper_quantile})",
        flush=True,
    )
    if config.get("quaternion_groups"):
        print(f"Quaternion-normalized state groups: {config['quaternion_groups']}", flush=True)
    return config


def load_episode_payload_pi05_state10(
    path,
    sidecar,
    image_keys,
    raw_state_indices,
    state_source,
    action_frame,
    ikpush_state_version,
    controller_mode,
    vision_mode,
    initial_task,
):
    with np.load(path, allow_pickle=True) as data:
        validate_episode_metadata(path, data, sidecar, action_frame, ikpush_state_version, controller_mode)
        task = scalar_str(data["task"]) if "task" in data else initial_task
        raw_states = data["state"].astype(np.float32)
        actions = data["action"].astype(np.float32)
        states = make_pi05_states(raw_states, actions, state_source, raw_state_indices)
        require_fields(data, image_keys[:2], path)
        wrist_first = data[image_keys[0]].astype(np.uint8)
        wrist_second = data[image_keys[1]].astype(np.uint8)
        if vision_mode == "rgb":
            require_fields(data, image_keys[2:], path)
            front_first = data[image_keys[2]].astype(np.uint8)
            front_second = data[image_keys[3]].astype(np.uint8)
        else:
            front_first = data[image_keys[2]].astype(np.uint8) if image_keys[2] in data else np.zeros_like(wrist_first)
            front_second = data[image_keys[3]].astype(np.uint8) if image_keys[3] in data else np.zeros_like(wrist_second)
        subtasks = data["subtask_index"].astype(np.int64).reshape(-1)
    n = states.shape[0]
    if not (
        actions.shape[0]
        == wrist_first.shape[0]
        == wrist_second.shape[0]
        == front_first.shape[0]
        == front_second.shape[0]
        == subtasks.shape[0]
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
        "n": n,
    }


def iter_pi05_episode_payloads(files, num_workers, **kwargs):
    if num_workers <= 1:
        for ep_idx, path in enumerate(files):
            yield ep_idx, load_episode_payload_pi05_state10(path, **kwargs)
        return

    def submit_next(executor, iterator, pending):
        try:
            ep_idx, path = next(iterator)
        except StopIteration:
            return False
        pending.append((ep_idx, executor.submit(load_episode_payload_pi05_state10, path, **kwargs)))
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


def existing_pi05_state_metadata_matches(existing_sidecar, state_source, raw_state_indices):
    existing_source = existing_sidecar.get("pi05_state_source")
    if existing_source is not None and existing_source != state_source:
        return False
    existing_indices = existing_sidecar.get("pi05_state_source_raw_indices")
    if existing_indices is not None and list(existing_indices) != list(raw_state_indices):
        return False
    return True


def main():
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    raw_root = Path(args.raw_root)
    files = episode_files(raw_root)
    sidecar = load_sidecar(raw_root)
    first = np.load(files[0], allow_pickle=True)
    source_state_names = raw_state_names(first, sidecar)
    if args.pi05_state_source == "last_command":
        raw_state_indices = last_command_state_indices(source_state_names)
        raw_state_features = LAST_COMMAND_FEATURES
    else:
        raw_state_indices = []
        raw_state_features = ["action"]

    raw_fps = int(first["fps"]) if "fps" in first else 50
    fps = int(args.fps or (sidecar.get("fps") if sidecar else raw_fps))
    vision_mode = "rgb" if args.rgb else "depth"
    raw_vision_mode = detect_raw_vision_mode(first, sidecar)
    action_frame = detect_action_frame(first, sidecar)
    ikpush_state_version = detect_ikpush_state_version(first, sidecar)
    controller_mode = detect_controller_mode(first, sidecar)
    if action_frame not in ("world", "base"):
        raise ValueError(f"Unsupported raw action_frame={action_frame!r}; expected 'world' or 'base'.")
    if raw_vision_mode != vision_mode:
        raise ValueError(
            f"Raw data vision_mode={raw_vision_mode!r}, but converter was run with "
            f"{'--rgb' if args.rgb else 'depth mode'}. Use --rgb only for RGB raw data."
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
        if existing_state_names and existing_state_names != PI05_STATE_NAMES:
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has different state feature names; "
                "use --overwrite or a new --repo_id."
            )
        if not existing_pi05_state_metadata_matches(existing_sidecar, args.pi05_state_source, raw_state_indices):
            raise ValueError(
                f"Existing LeRobot dataset at {out_dir} has different PI0.5 state-source metadata; "
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
        if existing_action_names and existing_action_names != ACTION_NAMES:
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

    if state_preprocess_config is None:
        if args.state_preprocess == "robust_quantile":
            state_preprocess_config = fit_pi05_state_preprocess_from_episodes(
                files,
                sidecar,
                raw_state_indices,
                args.pi05_state_source,
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
    recorder = DoorDPLeRobotRecorder(
        root=args.root,
        repo_id=args.repo_id,
        fps=fps,
        state_feature_names=PI05_STATE_NAMES,
        task=initial_task,
        resume=not args.overwrite,
        vision_mode=vision_mode,
        image_storage=args.image_storage,
        video_codec=args.video_codec,
        metadata={
            "action_frame": action_frame,
            "action_pose_frame": action_frame,
            "target_pose_frame": action_frame,
            "ikpush_state_version": ikpush_state_version,
            "door_dp_mode": controller_mode,
            "controller_mode": controller_mode,
            "image_storage": args.image_storage,
            "video_codec": args.video_codec,
            "state_preprocess": state_preprocess_config,
            "action_preprocess": action_preprocess_config,
        },
    )
    payloads = iter_pi05_episode_payloads(
        files,
        args.num_workers,
        sidecar=sidecar,
        image_keys=image_keys,
        raw_state_indices=raw_state_indices,
        state_source=args.pi05_state_source,
        action_frame=action_frame,
        ikpush_state_version=ikpush_state_version,
        controller_mode=controller_mode,
        vision_mode=vision_mode,
        initial_task=initial_task,
    )
    for ep_idx, payload in payloads:
        task = payload["task"]
        recorder.task = task
        states = payload["states"]
        states = apply_door_dp_state_preprocess(states, state_names=PI05_STATE_NAMES, config=state_preprocess_config)
        actions = payload["actions"]
        actions = apply_door_dp_action_preprocess(actions, action_names=ACTION_NAMES, config=action_preprocess_config)
        wrist_first = payload["wrist_first"]
        wrist_second = payload["wrist_second"]
        front_first = payload["front_first"]
        front_second = payload["front_second"]
        subtasks = payload["subtasks"]
        n = payload["n"]
        for i in range(n):
            recorder.add_frame(
                states[i],
                wrist_first[i],
                wrist_second[i],
                actions[i],
                int(subtasks[i]),
                front_mask_rgb=front_first[i],
                front_second_rgb=front_second[i],
            )
        recorder.save_episode()
        print(f"Converted {payload['path_name']}: {n} frames task={task!r} ({ep_idx + 1}/{len(files)})", flush=True)
    recorder.finalize()
    feature_sidecar = Path(args.root) / args.repo_id / "door_dp_feature_names.json"
    feature_sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar_payload = {
        "fps": fps,
        "state": PI05_STATE_NAMES,
        "action": ACTION_NAMES,
        "image_features": lerobot_image_keys_for_vision_mode(vision_mode),
        "source_raw_root": str(raw_root),
        "pi05_state_dim": len(PI05_STATE_NAMES),
        "pi05_state_source": args.pi05_state_source,
        "pi05_state_source_raw_features": raw_state_features,
        "pi05_state_source_raw_indices": raw_state_indices,
        "pi05_state_action_aligned": True,
        "pi05_state_action_names": ACTION_NAMES,
        "action_frame": action_frame,
        "action_pose_frame": action_frame,
        "target_pose_frame": action_frame,
        "ikpush_state_version": ikpush_state_version,
        "door_dp_mode": controller_mode,
        "controller_mode": controller_mode,
        "image_storage": args.image_storage,
        "video_codec": args.video_codec,
        "state_preprocess": state_preprocess_config,
        "action_preprocess": action_preprocess_config,
    }
    if vision_mode == "rgb":
        sidecar_payload["vision_mode"] = vision_mode
    with open(feature_sidecar, "w", encoding="utf-8") as f:
        json.dump(sidecar_payload, f, indent=2)
    print(
        f"Done. PI0.5 10D-state LeRobotDataset written to {Path(args.root) / args.repo_id} "
        f"state_source={args.pi05_state_source}",
        flush=True,
    )


if __name__ == "__main__":
    main()
