import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from door_dp_common import ACTION_NAMES, DoorDPLeRobotRecorder


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw Door DP .npz episodes into a local LeRobotDataset.")
    parser.add_argument("--raw_root", type=str, default=str(Path(__file__).resolve().parent / "data" / "door_dp_raw" / "local_door_dp"))
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
    raw_fps = int(first["fps"]) if "fps" in first else 50
    fps = int(args.fps or (sidecar.get("fps") if sidecar else raw_fps))
    out_dir = Path(args.root) / args.repo_id
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
    )
    for ep_idx, path in enumerate(files):
        data = np.load(path, allow_pickle=True)
        task = scalar_str(data["task"]) if "task" in data else initial_task
        recorder.task = task
        states = data["state"].astype(np.float32)
        actions = data["action"].astype(np.float32)
        masks = data["wrist_handle_mask"].astype(np.uint8)
        depths = data["wrist_masked_depth"].astype(np.uint8)
        subtasks = data["subtask_index"].astype(np.int64).reshape(-1)
        n = states.shape[0]
        if not (actions.shape[0] == masks.shape[0] == depths.shape[0] == subtasks.shape[0] == n):
            raise ValueError(f"Episode {path} has inconsistent lengths.")
        for i in range(n):
            recorder.add_frame(states[i], masks[i], depths[i], actions[i], int(subtasks[i]))
        recorder.save_episode()
        print(f"Converted {path.name}: {n} frames task={task!r} ({ep_idx + 1}/{len(files)})", flush=True)
    recorder.finalize()
    feature_sidecar = Path(args.root) / args.repo_id / "door_dp_feature_names.json"
    feature_sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_sidecar, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fps": fps,
                "state": state_names,
                "action": ACTION_NAMES,
                "image_features": ["observation.images.wrist_handle_mask", "observation.images.wrist_masked_depth"],
                "source_raw_root": str(raw_root),
            },
            f,
            indent=2,
        )
    print(f"Done. LeRobotDataset written to {Path(args.root) / args.repo_id}")


if __name__ == "__main__":
    main()
