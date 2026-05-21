import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

try:
    from .door_dp_common import (
        ACTION_NAMES,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        import_lerobot_or_raise,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
    )
except ImportError:
    from door_dp_common import (
        ACTION_NAMES,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        import_lerobot_or_raise,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
    )


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent

RAW_DP_KEYS = {
    "state",
    "action",
    "wrist_handle_mask",
    "wrist_masked_depth",
    "wrist_rgb",
    "front_handle_mask",
    "front_masked_depth",
    "front_rgb",
    "subtask_index",
    "task",
    "fps",
    "vision_mode",
    "state_feature_names",
    "action_names",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Decode a Door DP LeRobotDataset back to raw .npz episodes, stitch replay-only fields "
            "from the source raw episodes, and compare converted fields against the source raw data."
        )
    )
    parser.add_argument("--raw_root", type=str, required=True, help="Original raw Door DP episode directory.")
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument(
        "--out_raw_root",
        type=str,
        default=str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "lerobot_roundtrip_check"),
        help="Output directory for reconstructed raw .npz episodes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rgb", action="store_true", help="Roundtrip/check RGB+mask Door DP data.")
    parser.add_argument("--state_atol", type=float, default=1e-5)
    parser.add_argument("--action_atol", type=float, default=1e-5)
    parser.add_argument(
        "--image_atol",
        type=float,
        default=0.0,
        help="Allowed per-pixel uint8 difference for image pass/fail. Keep 0 for exact checking.",
    )
    parser.add_argument(
        "--fail_on_mismatch",
        action="store_true",
        help="Exit non-zero if any checked converted field fails.",
    )
    return parser.parse_args()


def raw_episode_files(raw_root):
    files = sorted(Path(raw_root).glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found under {raw_root}")
    return files


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_lerobot_root(root, repo_id):
    root = Path(root).expanduser().resolve()
    if (root / "meta" / "info.json").is_file():
        return root
    nested = root / repo_id
    if (nested / "meta" / "info.json").is_file():
        return nested
    return root


def load_lerobot_dataset(root, repo_id):
    import_lerobot_or_raise()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = resolve_lerobot_root(root, repo_id)
    try:
        return LeRobotDataset(repo_id=repo_id, root=str(dataset_root)), dataset_root
    except TypeError:
        return LeRobotDataset(repo_id, root=str(dataset_root)), dataset_root


def field(frame, key):
    if key in frame:
        return frame[key]
    if hasattr(frame, "__getitem__"):
        return frame[key]
    raise KeyError(key)


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        from PIL import Image

        if isinstance(value, Image.Image):
            return np.asarray(value)
    except Exception:
        pass
    return np.asarray(value)


def scalar_int(value, default=0):
    if value is None:
        return int(default)
    return int(as_numpy(value).reshape(-1)[0])


def scalar_str(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def lerobot_image_key(raw_key):
    return f"observation.images.{raw_key}"


def detect_raw_vision_mode(raw_data, raw_sidecar):
    if raw_sidecar and raw_sidecar.get("vision_mode") is not None:
        return normalize_vision_mode(raw_sidecar["vision_mode"])
    if "vision_mode" in raw_data.files:
        return normalize_vision_mode(scalar_str(raw_data["vision_mode"]))
    if "wrist_rgb" in raw_data.files or "front_rgb" in raw_data.files:
        return "rgb"
    return "depth"


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


def detect_lerobot_vision_mode(dataset_root):
    sidecar = load_json(Path(dataset_root) / "door_dp_feature_names.json")
    if sidecar and sidecar.get("vision_mode") is not None:
        return normalize_vision_mode(sidecar["vision_mode"])
    return "depth"


def detect_lerobot_action_frame(dataset_root):
    sidecar = load_json(Path(dataset_root) / "door_dp_feature_names.json")
    if not sidecar:
        return "world"
    return str(sidecar.get("action_frame", sidecar.get("action_pose_frame", "world"))).lower()


def detect_lerobot_ikpush_state_version(dataset_root):
    sidecar = load_json(Path(dataset_root) / "door_dp_feature_names.json")
    if not sidecar:
        return "legacy"
    return str(sidecar.get("ikpush_state_version", "legacy"))


def as_state(value):
    return np.asarray(as_numpy(value), dtype=np.float32).reshape(-1)


def as_action(value):
    return np.asarray(as_numpy(value), dtype=np.float32).reshape(-1)


def as_subtask(value, fallback=0):
    try:
        return np.asarray([scalar_int(value)], dtype=np.int64)
    except Exception:
        return np.asarray([fallback], dtype=np.int64)


def as_hwc_u8(value):
    arr = as_numpy(value)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        import cv2

        arr = cv2.resize(arr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        max_value = float(finite.max()) if finite.size else 0.0
        if max_value <= 1.5:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def raw_state_names(raw_data, raw_sidecar):
    if raw_sidecar and "state" in raw_sidecar:
        return [str(x) for x in raw_sidecar["state"]]
    if "state_feature_names" in raw_data.files:
        return [str(x) for x in np.asarray(raw_data["state_feature_names"]).tolist()]
    return [f"state_{i}" for i in range(raw_data["state"].shape[-1])]


def lerobot_state_names(dataset_root, raw_names):
    sidecar = load_json(Path(dataset_root) / "door_dp_feature_names.json")
    if sidecar and "state" in sidecar:
        return [str(x) for x in sidecar["state"]]
    return list(raw_names)


def state_reference(raw_state, raw_names, converted_names):
    if raw_state.shape[-1] == len(converted_names) and raw_names == converted_names:
        return raw_state
    name_to_idx = {name: idx for idx, name in enumerate(raw_names)}
    missing = [name for name in converted_names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Cannot project raw state to converted state names; missing columns: {missing}")
    indices = [name_to_idx[name] for name in converted_names]
    return raw_state[:, indices]


def raw_lengths(raw_files):
    lengths = []
    for path in raw_files:
        with np.load(path, allow_pickle=True) as data:
            lengths.append(int(data["state"].shape[0]))
    return lengths


def group_lerobot_indices(dataset, raw_files):
    episode_ids = []
    has_episode_index = True
    for idx in range(len(dataset)):
        frame = dataset[idx]
        try:
            episode_ids.append(scalar_int(field(frame, "episode_index")))
        except Exception:
            has_episode_index = False
            break
    if has_episode_index and episode_ids:
        groups = []
        start = 0
        while start < len(episode_ids):
            ep = episode_ids[start]
            end = start + 1
            while end < len(episode_ids) and episode_ids[end] == ep:
                end += 1
            groups.append(list(range(start, end)))
            start = end
        return groups

    groups = []
    cursor = 0
    for length in raw_lengths(raw_files):
        groups.append(list(range(cursor, cursor + length)))
        cursor += length
    if cursor != len(dataset):
        raise ValueError(
            f"LeRobot dataset has {len(dataset)} frames, but raw episode lengths sum to {cursor}; "
            "cannot infer episode grouping."
        )
    return groups


def decode_episode_from_lerobot(dataset, indices, image_keys):
    states = []
    actions = []
    images = {key: [] for key in image_keys}
    subtasks = []
    for idx in indices:
        frame = dataset[idx]
        states.append(as_state(field(frame, "observation.state")))
        actions.append(as_action(field(frame, "action")))
        for raw_key in image_keys:
            images[raw_key].append(as_hwc_u8(field(frame, lerobot_image_key(raw_key))))
        try:
            subtasks.append(as_subtask(field(frame, "subtask_index")))
        except Exception:
            subtasks.append(np.asarray([0], dtype=np.int64))
    payload = {
        "state": np.stack(states, axis=0).astype(np.float32),
        "action": np.stack(actions, axis=0).astype(np.float32),
        "subtask_index": np.stack(subtasks, axis=0).astype(np.int64),
    }
    for key, values in images.items():
        payload[key] = np.stack(values, axis=0).astype(np.uint8)
    return payload


def copy_replay_and_metadata(raw_data, payload):
    for key in raw_data.files:
        if key in RAW_DP_KEYS:
            continue
        payload[key] = np.asarray(raw_data[key])


def write_sidecar(out_root, fps, state_names, vision_mode, image_keys, action_frame, ikpush_state_version):
    sidecar = {
        "fps": int(fps),
        "state": list(state_names),
        "action": list(ACTION_NAMES),
        "image_features": list(image_keys),
        "format": "door_dp_raw_npz_v1_lerobot_roundtrip",
        "action_frame": action_frame,
        "action_pose_frame": action_frame,
        "target_pose_frame": action_frame,
        "ikpush_state_version": ikpush_state_version,
    }
    if vision_mode == "rgb":
        sidecar["vision_mode"] = vision_mode
    with (Path(out_root) / "door_dp_feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)


def compare_numeric(name, converted, reference, atol):
    converted = np.asarray(converted)
    reference = np.asarray(reference)
    result = {
        "field": name,
        "shape_converted": list(converted.shape),
        "shape_reference": list(reference.shape),
        "dtype_converted": str(converted.dtype),
        "dtype_reference": str(reference.dtype),
        "atol": float(atol),
        "ok": False,
    }
    if converted.shape != reference.shape:
        result["reason"] = "shape_mismatch"
        return result
    diff = np.abs(converted.astype(np.float64) - reference.astype(np.float64))
    result["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
    result["mean_abs_diff"] = float(diff.mean()) if diff.size else 0.0
    result["ok"] = bool(np.all(diff <= float(atol)))
    return result


def compare_exact(name, converted, reference):
    converted = np.asarray(converted)
    reference = np.asarray(reference)
    result = {
        "field": name,
        "shape_converted": list(converted.shape),
        "shape_reference": list(reference.shape),
        "dtype_converted": str(converted.dtype),
        "dtype_reference": str(reference.dtype),
        "ok": False,
    }
    if converted.shape != reference.shape:
        result["reason"] = "shape_mismatch"
        return result
    equal = converted == reference
    result["equal_count"] = int(np.count_nonzero(equal))
    result["total_count"] = int(equal.size)
    result["ok"] = bool(np.all(equal))
    return result


def compare_images(name, converted, reference, image_atol):
    result = compare_numeric(name, converted.astype(np.int16), reference.astype(np.int16), image_atol)
    converted_nonzero = np.sum(converted, axis=tuple(range(1, converted.ndim))) > 0
    reference_nonzero = np.sum(reference, axis=tuple(range(1, reference.ndim))) > 0
    result["converted_nonzero_frames"] = int(np.count_nonzero(converted_nonzero))
    result["reference_nonzero_frames"] = int(np.count_nonzero(reference_nonzero))
    result["exact_equal"] = bool(np.array_equal(converted, reference)) if converted.shape == reference.shape else False
    return result


def summarize_episode(ep_name, checks):
    ok = all(item.get("ok", False) for item in checks)
    print(f"{'PASS' if ok else 'FAIL'} {ep_name}", flush=True)
    for item in checks:
        status = "ok" if item.get("ok", False) else "mismatch"
        extra = ""
        if "max_abs_diff" in item:
            extra = f" max_abs={item['max_abs_diff']:.6g} mean_abs={item['mean_abs_diff']:.6g}"
        if "converted_nonzero_frames" in item:
            extra += (
                f" nonzero_frames={item['converted_nonzero_frames']}/"
                f"{item['reference_nonzero_frames']} exact={item['exact_equal']}"
            )
        print(f"  {status}: {item['field']}{extra}", flush=True)
    return ok


def main():
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_raw_root).expanduser().resolve()
    raw_files = raw_episode_files(raw_root)
    dataset, dataset_root = load_lerobot_dataset(args.root, args.repo_id)
    groups = group_lerobot_indices(dataset, raw_files)
    if len(groups) != len(raw_files):
        raise ValueError(f"Episode count mismatch: raw={len(raw_files)} lerobot={len(groups)}")

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to replace it.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    raw_sidecar = load_json(raw_root / "door_dp_feature_names.json")
    first_raw = np.load(raw_files[0], allow_pickle=True)
    requested_vision_mode = "rgb" if args.rgb else "depth"
    raw_vision_mode = detect_raw_vision_mode(first_raw, raw_sidecar)
    lerobot_vision_mode = detect_lerobot_vision_mode(dataset_root)
    lerobot_action_frame = detect_lerobot_action_frame(dataset_root)
    lerobot_state_version = detect_lerobot_ikpush_state_version(dataset_root)
    if raw_vision_mode != requested_vision_mode:
        raise ValueError(
            f"Raw data vision_mode={raw_vision_mode!r}, but roundtrip was run with "
            f"{'--rgb' if args.rgb else 'depth mode'}."
        )
    if lerobot_vision_mode != requested_vision_mode:
        raise ValueError(
            f"LeRobot data vision_mode={lerobot_vision_mode!r}, but roundtrip was run with "
            f"{'--rgb' if args.rgb else 'depth mode'}."
        )
    image_keys = raw_image_keys_for_vision_mode(requested_vision_mode)
    raw_names = raw_state_names(first_raw, raw_sidecar)
    converted_names = lerobot_state_names(dataset_root, raw_names)
    fps = int(first_raw["fps"]) if "fps" in first_raw.files else int((raw_sidecar or {}).get("fps", 50))
    action_frame = detect_action_frame(first_raw, raw_sidecar)
    ikpush_state_version = detect_ikpush_state_version(first_raw, raw_sidecar)
    if action_frame not in ("world", "base"):
        raise ValueError(f"Unsupported action_frame={action_frame!r}; expected 'world' or 'base'.")
    if lerobot_action_frame != action_frame:
        raise ValueError(
            f"LeRobot data action_frame={lerobot_action_frame!r}, but raw data action_frame={action_frame!r}."
        )
    if lerobot_state_version != ikpush_state_version:
        raise ValueError(
            f"LeRobot data ikpush_state_version={lerobot_state_version!r}, "
            f"but raw data ikpush_state_version={ikpush_state_version!r}."
        )
    first_raw.close()
    write_sidecar(out_root, fps, converted_names, requested_vision_mode, image_keys, action_frame, ikpush_state_version)

    report = {
        "raw_root": str(raw_root),
        "lerobot_root": str(dataset_root),
        "repo_id": args.repo_id,
        "out_raw_root": str(out_root),
        "vision_mode": requested_vision_mode,
        "episodes": [],
    }
    all_ok = True

    for ep_idx, (raw_path, frame_indices) in enumerate(zip(raw_files, groups)):
        converted = decode_episode_from_lerobot(dataset, frame_indices, image_keys)
        with np.load(raw_path, allow_pickle=True) as raw_data:
            episode_vision_mode = detect_raw_vision_mode(raw_data, raw_sidecar)
            if episode_vision_mode != requested_vision_mode:
                raise ValueError(f"{raw_path} has vision_mode={episode_vision_mode!r}, expected {requested_vision_mode!r}")
            episode_action_frame = detect_action_frame(raw_data, raw_sidecar)
            if episode_action_frame != action_frame:
                raise ValueError(
                    f"{raw_path} has action_frame={episode_action_frame!r}, expected {action_frame!r}; "
                    "do not mix world-frame and base-frame action datasets."
                )
            episode_state_version = detect_ikpush_state_version(raw_data, raw_sidecar)
            if episode_state_version != ikpush_state_version:
                raise ValueError(
                    f"{raw_path} has ikpush_state_version={episode_state_version!r}, expected {ikpush_state_version!r}; "
                    "do not mix old and new ikpush state semantics."
                )
            raw_state_ref = state_reference(raw_data["state"].astype(np.float32), raw_names, converted_names)
            payload = {
                **converted,
                "task": np.asarray(raw_data["task"]) if "task" in raw_data.files else np.asarray("door open"),
                "fps": np.asarray(int(raw_data["fps"]) if "fps" in raw_data.files else fps, dtype=np.int64),
                "state_feature_names": np.asarray(converted_names, dtype=str),
                "action_names": np.asarray(ACTION_NAMES, dtype=str),
                "action_frame": np.asarray(action_frame),
                "action_pose_frame": np.asarray(action_frame),
                "target_pose_frame": np.asarray(action_frame),
                "ikpush_state_version": np.asarray(ikpush_state_version),
            }
            if requested_vision_mode == "rgb":
                payload["vision_mode"] = np.asarray(requested_vision_mode)
            copy_replay_and_metadata(raw_data, payload)
            out_path = out_root / raw_path.name
            np.savez_compressed(out_path, **payload)

            checks = [
                compare_numeric("state", converted["state"], raw_state_ref, args.state_atol),
                compare_numeric("action", converted["action"], raw_data["action"].astype(np.float32), args.action_atol),
                compare_exact("subtask_index", converted["subtask_index"], raw_data["subtask_index"].astype(np.int64)),
            ]
            for key in image_keys:
                checks.append(compare_images(key, converted[key], raw_data[key].astype(np.uint8), args.image_atol))
            replay_keys = sorted(key for key in raw_data.files if key.startswith("replay_"))
            copied_keys = sorted(key for key in raw_data.files if key not in RAW_DP_KEYS)
            episode_ok = summarize_episode(raw_path.name, checks)
            all_ok = all_ok and episode_ok
            report["episodes"].append(
                {
                    "episode": raw_path.name,
                    "out_episode": str(out_path),
                    "frames": int(converted["state"].shape[0]),
                    "checks": checks,
                    "copied_non_lerobot_keys": copied_keys,
                    "copied_replay_keys": replay_keys,
                }
            )
            print(
                f"  wrote: {out_path} copied_non_lerobot_keys={len(copied_keys)} replay_keys={len(replay_keys)}",
                flush=True,
            )

    report["ok"] = bool(all_ok)
    report_path = out_root / "roundtrip_check_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(
        f"\nRoundtrip check {'PASSED' if all_ok else 'FAILED'}: "
        f"{len(raw_files)} episodes, output={out_root}, report={report_path}",
        flush=True,
    )
    if args.fail_on_mismatch and not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
