import argparse
import json
import shutil
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


def main():
    args = parse_args()
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
    )
    for ep_idx, path in enumerate(files):
        data = np.load(path, allow_pickle=True)
        task = scalar_str(data["task"]) if "task" in data else initial_task
        recorder.task = task
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
        print(f"Converted {path.name}: {n} frames task={task!r} ({ep_idx + 1}/{len(files)})", flush=True)
    recorder.finalize()
    feature_sidecar = Path(args.root) / args.repo_id / "door_dp_feature_names.json"
    feature_sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar_payload = {
        "fps": fps,
        "state": state_names,
        "action": ACTION_NAMES,
        "image_features": lerobot_image_keys_for_vision_mode(vision_mode),
        "source_raw_root": str(raw_root),
    }
    if vision_mode == "rgb":
        sidecar_payload["vision_mode"] = vision_mode
    with open(feature_sidecar, "w", encoding="utf-8") as f:
        json.dump(sidecar_payload, f, indent=2)
    print(f"Done. LeRobotDataset written to {Path(args.root) / args.repo_id}")


if __name__ == "__main__":
    main()
