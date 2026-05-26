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
        DoorDPLeRobotRecorder,
        lerobot_image_keys_for_vision_mode,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
    )
except ImportError:
    from door_dp_common import (
        ACTION_NAMES,
        DoorDPLeRobotRecorder,
        lerobot_image_keys_for_vision_mode,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
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
    if "vision_mode" in data.files:
        return normalize_vision_mode(scalar_str(data["vision_mode"]))
    if "wrist_rgb" in data.files or "front_rgb" in data.files:
        return "rgb"
    return "depth"


def require_fields(data, keys, path):
    missing = [key for key in keys if key not in data.files]
    if missing:
        raise KeyError(f"{path} is missing required fields for this vision mode: {missing}")


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
):
    with np.load(path, allow_pickle=True) as data:
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
        task = scalar_str(data["task"]) if "task" in data else initial_task
        states = data["state"].astype(np.float32)
        if keep_state_indices and states.shape[-1] <= max(keep_state_indices):
            raise ValueError(f"Episode {path} has state_dim={states.shape[-1]}, cannot apply selected state columns.")
        states = states[:, keep_state_indices]
        actions = data["action"].astype(np.float32)
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
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    initial_task = scalar_str(first["task"]) if "task" in first else "door open"
    recorder = DoorDPLeRobotRecorder(
        root=args.root,
        repo_id=args.repo_id,
        fps=fps,
        state_feature_names=state_names,
        task=initial_task,
        resume=not args.overwrite,
        vision_mode=vision_mode,
        metadata={
            "action_frame": action_frame,
            "action_pose_frame": action_frame,
            "target_pose_frame": action_frame,
            "ikpush_state_version": ikpush_state_version,
            "door_dp_mode": controller_mode,
            "controller_mode": controller_mode,
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
    )
    for ep_idx, payload in payloads:
        task = payload["task"]
        recorder.task = task
        states = payload["states"]
        actions = payload["actions"]
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
        "state": state_names,
        "action": ACTION_NAMES,
        "image_features": lerobot_image_keys_for_vision_mode(vision_mode),
        "source_raw_root": str(raw_root),
        "action_frame": action_frame,
        "action_pose_frame": action_frame,
        "target_pose_frame": action_frame,
        "ikpush_state_version": ikpush_state_version,
        "door_dp_mode": controller_mode,
        "controller_mode": controller_mode,
    }
    if vision_mode == "rgb":
        sidecar_payload["vision_mode"] = vision_mode
    with open(feature_sidecar, "w", encoding="utf-8") as f:
        json.dump(sidecar_payload, f, indent=2)
    print(f"Done. LeRobotDataset written to {Path(args.root) / args.repo_id}")


if __name__ == "__main__":
    main()
