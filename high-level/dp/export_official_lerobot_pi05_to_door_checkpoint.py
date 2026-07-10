import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch


DP_ROOT = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = DP_ROOT.parent
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from door_dp_common import ACTION_NAMES, normalize_vision_mode  # noqa: E402
from door_policy_backend import (  # noqa: E402
    ACTION,
    BACKEND_LEROBOT_PI05,
    BACKEND_LEROBOT_PI05_EVO,
    OBS_STATE,
    DoorPolicyChunkDataset,
    LeRobotPI05DoorPolicyBackend,
    _backend_from_pi05_policy_type,
    _feature_dim_from_stats,
    _normalize_pi05_policy_type,
    _resolve_lerobot_root,
    load_lerobot_policy_normalizer_stats,
    merge_lerobot_processor_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Wrap an official LeRobot pi0.5/pi0.5-evo checkpoint into the Door policy checkpoint "
            "format used by high-level/dp/play/play_door_policy.py."
        )
    )
    parser.add_argument(
        "--official_checkpoint",
        required=True,
        help="Official LeRobot checkpoint dir, e.g. .../checkpoints/050000 or .../checkpoints/050000/pretrained_model.",
    )
    parser.add_argument("--root", type=str, default=str(HIGH_LEVEL_ROOT / "data" / "lerobot"))
    parser.add_argument("--repo_id", type=str, default="local/door_dp")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--manifest_name", type=str, default="model_latest.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--task_prompt", type=str, default="open the door")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "Tokenizer path/name for pi0.5 language tokens. If omitted, the wrapper tries to read it "
            "from policy_preprocessor.json/config.json, then falls back to google/paligemma-3b-pt-224."
        ),
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_pretrained_model(path):
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.parent.name == "pretrained_model":
        policy_dir = path.parent
        official_step_dir = policy_dir.parent
    elif path.name == "pretrained_model":
        policy_dir = path
        official_step_dir = path.parent
    else:
        official_step_dir = path
        policy_dir = path / "pretrained_model"
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"Could not find official pretrained_model directory under: {path}")
    config_path = policy_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Official LeRobot policy config missing: {config_path}")
    return official_step_dir, policy_dir, load_json(config_path)


def load_sidecar(dataset_root):
    sidecar_path = Path(dataset_root) / "door_dp_feature_names.json"
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Door sidecar missing: {sidecar_path}. "
            "This file is required for state/action preprocess alignment."
        )
    return load_json(sidecar_path), sidecar_path


def _find_nested_key(value: Any, key: str) -> Optional[Any]:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_nested_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_nested_key(child, key)
            if found is not None:
                return found
    return None


def resolve_tokenizer_name(policy_dir: Path, policy_config: dict, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    preprocessor_path = policy_dir / "policy_preprocessor.json"
    if preprocessor_path.is_file():
        try:
            found = _find_nested_key(load_json(preprocessor_path), "tokenizer_name")
            if found:
                return str(found)
        except Exception:
            pass
    for key in ("tokenizer_name", "tokenizer_path", "pretrained_model_name_or_path"):
        value = policy_config.get(key)
        if value:
            return str(value)
    return "google/paligemma-3b-pt-224"


def _as_hw_pair(value, default=(224, 224)):
    if value is None:
        return tuple(default)
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    return tuple(default)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    official_step_dir, policy_dir, policy_config = resolve_pretrained_model(args.official_checkpoint)

    policy_type = _normalize_pi05_policy_type(policy_config.get("type", "pi05"))
    backend_name = _backend_from_pi05_policy_type(policy_type)
    if backend_name not in (BACKEND_LEROBOT_PI05, BACKEND_LEROBOT_PI05_EVO):
        raise ValueError(f"Expected official pi0.5 checkpoint, got policy type={policy_config.get('type')!r}")

    dataset_root = _resolve_lerobot_root(args.root, args.repo_id)
    sidecar_data, sidecar_path = load_sidecar(dataset_root)
    if args.rgb and args.depth_only:
        raise ValueError("--rgb and --depth_only are mutually exclusive.")
    vision_mode = "rgb" if args.rgb else ("depth_only" if args.depth_only else "depth")
    dataset_vision_mode = normalize_vision_mode(sidecar_data.get("vision_mode", "depth"))
    if dataset_vision_mode != vision_mode:
        raise ValueError(f"Dataset vision_mode={dataset_vision_mode!r}, export expected {vision_mode!r}.")

    action_frame = str(sidecar_data.get("action_frame", sidecar_data.get("action_pose_frame", "world"))).lower()
    controller_mode = str(sidecar_data.get("door_dp_mode", sidecar_data.get("controller_mode", "legacy")))
    action_names = list(sidecar_data.get("action") or ACTION_NAMES)
    chunk_size = int(policy_config["chunk_size"])
    action_horizon = int(args.action_horizon if args.action_horizon is not None else policy_config["n_action_steps"])
    num_inference_steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else policy_config.get("num_inference_steps", 10)
    )

    dataset = DoorPolicyChunkDataset(dataset_root, args.repo_id, chunk_size, vision_mode=vision_mode)
    processor_stats = load_lerobot_policy_normalizer_stats(policy_dir)
    stats = merge_lerobot_processor_stats(dataset.stats, processor_stats)
    state_dim = _feature_dim_from_stats(stats, OBS_STATE)
    action_dim = _feature_dim_from_stats(stats, ACTION)
    if action_dim != len(action_names):
        raise ValueError(
            f"Door policy action_dim={action_dim} does not match dataset action_names={len(action_names)} "
            f"from {sidecar_path}."
        )

    tokenizer_name = resolve_tokenizer_name(policy_dir, policy_config, args.tokenizer_name)
    backend = LeRobotPI05DoorPolicyBackend.create(
        stats=stats,
        vision_mode=vision_mode,
        action_frame=action_frame,
        sidecar_config=sidecar_data,
        device=device,
        chunk_size=chunk_size,
        action_horizon=action_horizon,
        state_dim=state_dim,
        action_dim=action_dim,
        pretrained_path=str(policy_dir),
        task_prompt=args.task_prompt,
        tokenizer_name=tokenizer_name,
        policy_type=policy_type,
        normalization_mapping=policy_config.get("normalization_mapping"),
        paligemma_variant=policy_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=policy_config.get("action_expert_variant", "gemma_300m"),
        dtype=policy_config.get("dtype", "float32"),
        max_state_dim=int(policy_config.get("max_state_dim", max(128, state_dim))),
        max_action_dim=int(policy_config.get("max_action_dim", max(32, action_dim))),
        num_inference_steps=num_inference_steps,
        image_resolution=_as_hw_pair(policy_config.get("image_resolution"), default=(224, 224)),
        tokenizer_max_length=int(policy_config.get("tokenizer_max_length", 200)),
        gradient_checkpointing=False,
        compile_model=False,
        compile_mode=policy_config.get("compile_mode", "max-autotune"),
        freeze_vision_encoder=bool(policy_config.get("freeze_vision_encoder", False)),
        train_expert_only=bool(policy_config.get("train_expert_only", False)),
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = out_dir.parent / args.manifest_name
    train_config = {
        "backend": backend_name,
        "policy_type": policy_type,
        "source_official_checkpoint": str(official_step_dir),
        "source_official_policy_dir": str(policy_dir),
        "dataset_root": str(dataset_root),
        "repo_id": args.repo_id,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_names": action_names,
        "vision_mode": vision_mode,
        "action_frame": action_frame,
        "ikpush_state_version": str(sidecar_data.get("ikpush_state_version", "legacy")),
        "door_dp_mode": controller_mode,
        "controller_mode": controller_mode,
        "stats_source": "official_lerobot_preprocessor" if processor_stats else "dataset",
        "tokenizer_name": tokenizer_name,
        "task_prompt": args.task_prompt,
    }
    backend.save_checkpoint(out_dir, optimizer=None, extra_config=train_config, manifest_path=manifest_path)
    print(
        f"Wrapped official pi0.5 checkpoint:\n"
        f"  source: {policy_dir}\n"
        f"  door checkpoint: {out_dir}\n"
        f"  manifest: {manifest_path}\n"
        f"  tokenizer: {tokenizer_name}"
    )
    print(f"Sidecar aligned from: {sidecar_path}")


if __name__ == "__main__":
    main()
