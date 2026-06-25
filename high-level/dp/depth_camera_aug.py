"""Depth-camera pose jitter and light sensor-noise helpers for Door DP."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - YAML is available in the project envs, but keep a fallback.
    yaml = None


DEPTH_IMAGE_WIDTH = 640
DEPTH_IMAGE_HEIGHT = 480
DEPTH_CAMERA_RESOLUTION = [DEPTH_IMAGE_WIDTH, DEPTH_IMAGE_HEIGHT]
DEPTH_AUG_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "depth_camera_aug_default.yaml"
DEPTH_AUG_CONFIG_ENV_VAR = "DOOR_DEPTH_AUG_CONFIG"


_BUILTIN_DEPTH_AUG_DEFAULTS: dict[str, Any] = {
    "enable_depth_noise": False,
    "depth_noise_prob": 0.50,
    "depth_gaussian_std_m": 0.005,
    "depth_gaussian_distance_factor": 0.05,
    "depth_edge_noise_prob": 0.10,
    "depth_edge_gradient_threshold_m": 0.05,
    "depth_edge_dilation_kernel_size": 3,
    "depth_hole_noise_prob": 0.01,
    "depth_hole_block_size_min": 3,
    "depth_hole_block_size": 8,
    "depth_hole_white_prob": 0.50,
    "depth_dropout_prob": 0.0,
    "depth_salt_pepper_prob": 0.0,
    "enable_depth_gaussian_blur": False,
    "depth_gaussian_blur_ksize": 15,
    "depth_gaussian_blur_sigma": 4.0,
    "enable_depth_camera_randomization": False,
    "depth_camera_pos_rand_m": 0.01,
    "depth_camera_rot_rand_deg": 2.0,
}


def _depth_aug_config_path(config_path: str | os.PathLike[str] | None = None) -> Path:
    if config_path:
        return Path(config_path).expanduser()
    override = os.environ.get(DEPTH_AUG_CONFIG_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return DEPTH_AUG_DEFAULT_CONFIG_PATH


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError(f"PyYAML is required to read depth augmentation config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Depth augmentation config must be a mapping: {path}")
    return dict(data)


def depth_aug_defaults(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return shared Door DP depth augmentation defaults.

    Defaults come from high-level/dp/config/depth_camera_aug_default.yaml by
    default. Set DOOR_DEPTH_AUG_CONFIG=/path/to/other.yaml to debug with a
    different config without editing code.
    """

    defaults = dict(_BUILTIN_DEPTH_AUG_DEFAULTS)
    config_path = _depth_aug_config_path(config_path)
    loaded = _load_yaml_mapping(config_path)
    unknown = sorted(set(loaded) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown keys in depth augmentation config {config_path}: {unknown}")
    defaults.update(loaded)
    defaults["depth_aug_config_path"] = str(config_path)
    return defaults


_DEPTH_AUG_FLAGS_BY_ATTR: dict[str, tuple[str, ...]] = {
    "enable_depth_noise": ("--enable_depth_noise", "--no_enable_depth_noise"),
    "depth_noise_prob": ("--depth_noise_prob",),
    "depth_gaussian_std_m": ("--depth_gaussian_std_m",),
    "depth_gaussian_distance_factor": ("--depth_gaussian_distance_factor",),
    "depth_edge_noise_prob": ("--depth_edge_noise_prob",),
    "depth_edge_gradient_threshold_m": ("--depth_edge_gradient_threshold_m",),
    "depth_edge_dilation_kernel_size": ("--depth_edge_dilation_kernel_size",),
    "depth_hole_noise_prob": ("--depth_hole_noise_prob",),
    "depth_hole_block_size_min": ("--depth_hole_block_size_min",),
    "depth_hole_block_size": ("--depth_hole_block_size",),
    "depth_hole_white_prob": ("--depth_hole_white_prob",),
    "depth_dropout_prob": ("--depth_dropout_prob",),
    "depth_salt_pepper_prob": ("--depth_salt_pepper_prob",),
    "enable_depth_gaussian_blur": ("--enable_depth_gaussian_blur", "--no_enable_depth_gaussian_blur"),
    "depth_gaussian_blur_ksize": ("--depth_gaussian_blur_ksize",),
    "depth_gaussian_blur_sigma": ("--depth_gaussian_blur_sigma",),
    "enable_depth_camera_randomization": (
        "--enable_depth_camera_randomization",
        "--no_enable_depth_camera_randomization",
    ),
    "depth_camera_pos_rand_m": ("--depth_camera_pos_rand_m",),
    "depth_camera_rot_rand_deg": ("--depth_camera_rot_rand_deg",),
}


def _cli_flag_present(argv: list[str] | tuple[str, ...] | set[str], flags: tuple[str, ...]) -> bool:
    argv_list = list(argv or [])
    argv_set = set(argv_list)
    for flag in flags:
        if flag in argv_set:
            return True
        prefix = flag + "="
        if any(str(item).startswith(prefix) for item in argv_list):
            return True
    return False


def apply_depth_aug_config_defaults(args: Any, argv: list[str] | tuple[str, ...] | set[str] | None = None) -> Any:
    """Apply --depth_aug_config values to parsed args unless CLI explicitly overrides.

    This is mainly for wrappers such as record_door_dp_dataset_a2w_state10.py:
    argparse defaults are created before a user-provided --depth_aug_config is
    parsed, so this helper refreshes the defaults after parsing while preserving
    any explicit command-line overrides.
    """

    argv = list(argv or [])
    config_path = getattr(args, "depth_aug_config", None)
    defaults = depth_aug_defaults(config_path)
    setattr(args, "depth_aug_config", str(defaults["depth_aug_config_path"]))
    for attr, flags in _DEPTH_AUG_FLAGS_BY_ATTR.items():
        if _cli_flag_present(argv, flags):
            continue
        if hasattr(args, attr):
            setattr(args, attr, defaults[attr])
    return args


@dataclass(frozen=True)
class DepthNoiseConfig:
    enabled: bool = False
    noise_prob: float = 0.50
    env_selected: bool | None = None
    gaussian_std_m: float = 0.005
    gaussian_distance_factor: float = 0.05
    edge_noise_prob: float = 0.10
    edge_gradient_threshold_m: float = 0.05
    edge_dilation_kernel_size: int = 3
    hole_noise_prob: float = 0.01
    hole_block_size_min: int = 3
    hole_block_size: int = 8
    hole_white_prob: float = 0.50
    dropout_prob: float = 0.0
    salt_pepper_prob: float = 0.0
    gaussian_blur_enabled: bool = False
    gaussian_blur_ksize: int = 15
    gaussian_blur_sigma: float = 4.0
    near_clip_m: float = 0.02
    far_clip_m: float = 2.0


def add_depth_aug_args(parser: Any) -> None:
    defaults = depth_aug_defaults()
    parser.add_argument(
        "--depth_aug_config",
        type=str,
        default=str(defaults["depth_aug_config_path"]),
        help=(
            "Depth augmentation config used for parser defaults. To use another "
            f"config for defaults, set {DEPTH_AUG_CONFIG_ENV_VAR} before launching."
        ),
    )
    parser.add_argument(
        "--enable_depth_noise",
        dest="enable_depth_noise",
        action="store_true",
        default=bool(defaults["enable_depth_noise"]),
    )
    parser.add_argument("--no_enable_depth_noise", dest="enable_depth_noise", action="store_false")
    parser.add_argument(
        "--depth_noise_prob",
        type=float,
        default=float(defaults["depth_noise_prob"]),
        help="Fraction of envs assigned persistent depth noise; selected envs apply noise to every frame.",
    )
    parser.add_argument("--depth_gaussian_std_m", type=float, default=float(defaults["depth_gaussian_std_m"]))
    parser.add_argument(
        "--depth_gaussian_distance_factor",
        type=float,
        default=float(defaults["depth_gaussian_distance_factor"]),
    )
    parser.add_argument("--depth_edge_noise_prob", type=float, default=float(defaults["depth_edge_noise_prob"]))
    parser.add_argument(
        "--depth_edge_gradient_threshold_m",
        type=float,
        default=float(defaults["depth_edge_gradient_threshold_m"]),
    )
    parser.add_argument(
        "--depth_edge_dilation_kernel_size",
        type=int,
        default=int(defaults["depth_edge_dilation_kernel_size"]),
    )
    parser.add_argument("--depth_hole_noise_prob", type=float, default=float(defaults["depth_hole_noise_prob"]))
    parser.add_argument(
        "--depth_hole_block_size_min",
        type=int,
        default=int(defaults["depth_hole_block_size_min"]),
        help="Minimum block size for blocky depth holes.",
    )
    parser.add_argument(
        "--depth_hole_block_size",
        type=int,
        default=int(defaults["depth_hole_block_size"]),
        help=(
            "Maximum block size for blocky depth holes. Each noise call samples "
            "an integer block size from [depth_hole_block_size_min, depth_hole_block_size]."
        ),
    )
    parser.add_argument(
        "--depth_hole_white_prob",
        type=float,
        default=float(defaults["depth_hole_white_prob"]),
        help="Within selected block holes, probability of filling a block with far depth/white instead of zero/black.",
    )
    parser.add_argument("--depth_dropout_prob", type=float, default=float(defaults["depth_dropout_prob"]))
    parser.add_argument("--depth_salt_pepper_prob", type=float, default=float(defaults["depth_salt_pepper_prob"]))
    parser.add_argument(
        "--enable_depth_gaussian_blur",
        dest="enable_depth_gaussian_blur",
        action="store_true",
        default=bool(defaults["enable_depth_gaussian_blur"]),
        help="Apply NX-style Gaussian blur after all other depth image noise.",
    )
    parser.add_argument("--no_enable_depth_gaussian_blur", dest="enable_depth_gaussian_blur", action="store_false")
    parser.add_argument(
        "--depth_gaussian_blur_ksize",
        type=int,
        default=int(defaults["depth_gaussian_blur_ksize"]),
        help="Odd Gaussian blur kernel size; <=0 disables blur even if the switch is enabled.",
    )
    parser.add_argument(
        "--depth_gaussian_blur_sigma",
        type=float,
        default=float(defaults["depth_gaussian_blur_sigma"]),
    )
    parser.add_argument(
        "--enable_depth_camera_randomization",
        dest="enable_depth_camera_randomization",
        action="store_true",
        default=bool(defaults["enable_depth_camera_randomization"]),
    )
    parser.add_argument("--no_enable_depth_camera_randomization", dest="enable_depth_camera_randomization", action="store_false")
    parser.add_argument("--depth_camera_pos_rand_m", type=float, default=float(defaults["depth_camera_pos_rand_m"]))
    parser.add_argument("--depth_camera_rot_rand_deg", type=float, default=float(defaults["depth_camera_rot_rand_deg"]))


def depth_aug_custom_parameters() -> list[dict[str, Any]]:
    defaults = depth_aug_defaults()
    return [
        {
            "name": "--depth_aug_config",
            "type": str,
            "default": str(defaults["depth_aug_config_path"]),
            "help": (
                "Depth augmentation config used for parser defaults. To use another "
                f"config for defaults, set {DEPTH_AUG_CONFIG_ENV_VAR} before launching."
            ),
        },
        {
            "name": "--enable_depth_noise",
            "dest": "enable_depth_noise",
            "action": "store_true",
            "default": bool(defaults["enable_depth_noise"]),
        },
        {"name": "--no_enable_depth_noise", "dest": "enable_depth_noise", "action": "store_false"},
        {
            "name": "--depth_noise_prob",
            "type": float,
            "default": float(defaults["depth_noise_prob"]),
            "help": "Fraction of envs assigned persistent depth noise; selected envs apply noise to every frame.",
        },
        {"name": "--depth_gaussian_std_m", "type": float, "default": float(defaults["depth_gaussian_std_m"])},
        {
            "name": "--depth_gaussian_distance_factor",
            "type": float,
            "default": float(defaults["depth_gaussian_distance_factor"]),
        },
        {"name": "--depth_edge_noise_prob", "type": float, "default": float(defaults["depth_edge_noise_prob"])},
        {
            "name": "--depth_edge_gradient_threshold_m",
            "type": float,
            "default": float(defaults["depth_edge_gradient_threshold_m"]),
        },
        {
            "name": "--depth_edge_dilation_kernel_size",
            "type": int,
            "default": int(defaults["depth_edge_dilation_kernel_size"]),
        },
        {"name": "--depth_hole_noise_prob", "type": float, "default": float(defaults["depth_hole_noise_prob"])},
        {
            "name": "--depth_hole_block_size_min",
            "type": int,
            "default": int(defaults["depth_hole_block_size_min"]),
            "help": "Minimum block size for blocky depth holes.",
        },
        {
            "name": "--depth_hole_block_size",
            "type": int,
            "default": int(defaults["depth_hole_block_size"]),
            "help": (
                "Maximum block size for blocky depth holes. Each noise call samples "
                "an integer block size from [depth_hole_block_size_min, depth_hole_block_size]."
            ),
        },
        {
            "name": "--depth_hole_white_prob",
            "type": float,
            "default": float(defaults["depth_hole_white_prob"]),
            "help": "Within selected block holes, probability of filling a block with far depth/white instead of zero/black.",
        },
        {"name": "--depth_dropout_prob", "type": float, "default": float(defaults["depth_dropout_prob"])},
        {"name": "--depth_salt_pepper_prob", "type": float, "default": float(defaults["depth_salt_pepper_prob"])},
        {
            "name": "--enable_depth_gaussian_blur",
            "dest": "enable_depth_gaussian_blur",
            "action": "store_true",
            "default": bool(defaults["enable_depth_gaussian_blur"]),
            "help": "Apply NX-style Gaussian blur after all other depth image noise.",
        },
        {"name": "--no_enable_depth_gaussian_blur", "dest": "enable_depth_gaussian_blur", "action": "store_false"},
        {
            "name": "--depth_gaussian_blur_ksize",
            "type": int,
            "default": int(defaults["depth_gaussian_blur_ksize"]),
            "help": "Odd Gaussian blur kernel size; <=0 disables blur even if the switch is enabled.",
        },
        {
            "name": "--depth_gaussian_blur_sigma",
            "type": float,
            "default": float(defaults["depth_gaussian_blur_sigma"]),
        },
        {
            "name": "--enable_depth_camera_randomization",
            "dest": "enable_depth_camera_randomization",
            "action": "store_true",
            "default": bool(defaults["enable_depth_camera_randomization"]),
        },
        {"name": "--no_enable_depth_camera_randomization", "dest": "enable_depth_camera_randomization", "action": "store_false"},
        {"name": "--depth_camera_pos_rand_m", "type": float, "default": float(defaults["depth_camera_pos_rand_m"])},
        {"name": "--depth_camera_rot_rand_deg", "type": float, "default": float(defaults["depth_camera_rot_rand_deg"])},
    ]


def add_depth_aug_command_args(cmd: list[str], args: Any) -> None:
    defaults = depth_aug_defaults()
    if bool(getattr(args, "enable_depth_noise", False)):
        cmd.append("--enable_depth_noise")
    else:
        cmd.append("--no_enable_depth_noise")
    cmd += ["--depth_aug_config", str(getattr(args, "depth_aug_config", defaults["depth_aug_config_path"]))]
    cmd += [
        "--depth_noise_prob",
        str(float(getattr(args, "depth_noise_prob", defaults["depth_noise_prob"]))),
        "--depth_gaussian_std_m",
        str(float(getattr(args, "depth_gaussian_std_m", defaults["depth_gaussian_std_m"]))),
        "--depth_gaussian_distance_factor",
        str(float(getattr(args, "depth_gaussian_distance_factor", defaults["depth_gaussian_distance_factor"]))),
        "--depth_edge_noise_prob",
        str(float(getattr(args, "depth_edge_noise_prob", defaults["depth_edge_noise_prob"]))),
        "--depth_edge_gradient_threshold_m",
        str(float(getattr(args, "depth_edge_gradient_threshold_m", defaults["depth_edge_gradient_threshold_m"]))),
        "--depth_edge_dilation_kernel_size",
        str(int(getattr(args, "depth_edge_dilation_kernel_size", defaults["depth_edge_dilation_kernel_size"]))),
        "--depth_hole_noise_prob",
        str(float(getattr(args, "depth_hole_noise_prob", defaults["depth_hole_noise_prob"]))),
        "--depth_hole_block_size_min",
        str(int(getattr(args, "depth_hole_block_size_min", defaults["depth_hole_block_size_min"]))),
        "--depth_hole_block_size",
        str(int(getattr(args, "depth_hole_block_size", defaults["depth_hole_block_size"]))),
        "--depth_hole_white_prob",
        str(float(getattr(args, "depth_hole_white_prob", defaults["depth_hole_white_prob"]))),
        "--depth_dropout_prob",
        str(float(getattr(args, "depth_dropout_prob", defaults["depth_dropout_prob"]))),
        "--depth_salt_pepper_prob",
        str(float(getattr(args, "depth_salt_pepper_prob", defaults["depth_salt_pepper_prob"]))),
    ]
    if bool(getattr(args, "enable_depth_gaussian_blur", False)):
        cmd.append("--enable_depth_gaussian_blur")
    else:
        cmd.append("--no_enable_depth_gaussian_blur")
    cmd += [
        "--depth_gaussian_blur_ksize",
        str(int(getattr(args, "depth_gaussian_blur_ksize", defaults["depth_gaussian_blur_ksize"]))),
        "--depth_gaussian_blur_sigma",
        str(float(getattr(args, "depth_gaussian_blur_sigma", defaults["depth_gaussian_blur_sigma"]))),
    ]
    if bool(getattr(args, "enable_depth_camera_randomization", False)):
        cmd.append("--enable_depth_camera_randomization")
    else:
        cmd.append("--no_enable_depth_camera_randomization")
    cmd += [
        "--depth_camera_pos_rand_m",
        str(float(getattr(args, "depth_camera_pos_rand_m", defaults["depth_camera_pos_rand_m"]))),
        "--depth_camera_rot_rand_deg",
        str(float(getattr(args, "depth_camera_rot_rand_deg", defaults["depth_camera_rot_rand_deg"]))),
    ]


def configure_depth_noise_for_env(args: Any) -> bool:
    if hasattr(args, "depth_noise_env_selected"):
        return bool(getattr(args, "depth_noise_env_selected"))

    defaults = depth_aug_defaults()
    requested = bool(getattr(args, "enable_depth_noise", False))
    probability = min(1.0, max(0.0, float(getattr(args, "depth_noise_prob", defaults["depth_noise_prob"]))))
    num_envs = max(1, int(getattr(args, "num_envs", 1)))
    env_index = min(num_envs - 1, max(0, int(getattr(args, "parallel_env_id", 0))))

    selected_count = int(math.floor(probability * num_envs + 0.5))
    if requested and probability > 0.0 and selected_count == 0:
        selected_count = 1
    if not requested:
        selected_count = 0
    selected_count = min(num_envs, max(0, selected_count))

    selected = False
    selected_env_ids: list[int] = []
    if requested and selected_count > 0:
        seed = int(getattr(args, "seed", 0)) & 0xFFFFFFFF
        selection_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xD3E7A5]))
        selected_env_ids = sorted(
            int(value)
            for value in selection_rng.choice(num_envs, size=selected_count, replace=False).tolist()
        )
        selected = env_index in selected_env_ids

    args.depth_noise_env_selected = bool(selected)
    args.depth_noise_env_probability = float(probability)
    args.depth_noise_selected_env_count = int(selected_count)
    args.depth_noise_selection_mode = "fixed_env_subset"
    return bool(selected)


def depth_noise_config_from_args(args: Any) -> DepthNoiseConfig:
    defaults = depth_aug_defaults()
    env_selected = configure_depth_noise_for_env(args)
    return DepthNoiseConfig(
        enabled=bool(getattr(args, "enable_depth_noise", False)),
        noise_prob=float(getattr(args, "depth_noise_prob", defaults["depth_noise_prob"])),
        env_selected=bool(env_selected),
        gaussian_std_m=float(getattr(args, "depth_gaussian_std_m", defaults["depth_gaussian_std_m"])),
        gaussian_distance_factor=float(
            getattr(args, "depth_gaussian_distance_factor", defaults["depth_gaussian_distance_factor"])
        ),
        edge_noise_prob=float(getattr(args, "depth_edge_noise_prob", defaults["depth_edge_noise_prob"])),
        edge_gradient_threshold_m=float(
            getattr(args, "depth_edge_gradient_threshold_m", defaults["depth_edge_gradient_threshold_m"])
        ),
        edge_dilation_kernel_size=int(
            getattr(args, "depth_edge_dilation_kernel_size", defaults["depth_edge_dilation_kernel_size"])
        ),
        hole_noise_prob=float(getattr(args, "depth_hole_noise_prob", defaults["depth_hole_noise_prob"])),
        hole_block_size_min=int(getattr(args, "depth_hole_block_size_min", defaults["depth_hole_block_size_min"])),
        hole_block_size=int(getattr(args, "depth_hole_block_size", defaults["depth_hole_block_size"])),
        hole_white_prob=float(getattr(args, "depth_hole_white_prob", defaults["depth_hole_white_prob"])),
        dropout_prob=float(getattr(args, "depth_dropout_prob", defaults["depth_dropout_prob"])),
        salt_pepper_prob=float(getattr(args, "depth_salt_pepper_prob", defaults["depth_salt_pepper_prob"])),
        gaussian_blur_enabled=bool(getattr(args, "enable_depth_gaussian_blur", defaults["enable_depth_gaussian_blur"])),
        gaussian_blur_ksize=int(getattr(args, "depth_gaussian_blur_ksize", defaults["depth_gaussian_blur_ksize"])),
        gaussian_blur_sigma=float(getattr(args, "depth_gaussian_blur_sigma", defaults["depth_gaussian_blur_sigma"])),
        near_clip_m=float(getattr(args, "camera_depth_clip_lower", 0.02)),
        far_clip_m=float(getattr(args, "camera_depth_clip_far", 2.0)),
    )


def depth_aug_metadata_from_args(args: Any) -> dict[str, Any]:
    cfg = depth_noise_config_from_args(args)
    return {
        "image_width": DEPTH_IMAGE_WIDTH,
        "image_height": DEPTH_IMAGE_HEIGHT,
        "depth_noise_enabled": bool(cfg.enabled),
        "depth_noise_selection": {
            "mode": str(getattr(args, "depth_noise_selection_mode", "fixed_env_subset")),
            "env_probability": float(getattr(args, "depth_noise_env_probability", cfg.noise_prob)),
            "env_selected": bool(cfg.env_selected),
            "selected_env_count": int(getattr(args, "depth_noise_selected_env_count", 0)),
            "num_envs": int(getattr(args, "num_envs", 1)),
        },
        "depth_noise_config": {
            "noise_prob": float(cfg.noise_prob),
            "noise_prob_semantics": "fixed_env_fraction",
            "gaussian_std_m": float(cfg.gaussian_std_m),
            "gaussian_distance_factor": float(cfg.gaussian_distance_factor),
            "edge_noise_prob": float(cfg.edge_noise_prob),
            "edge_gradient_threshold_m": float(cfg.edge_gradient_threshold_m),
            "edge_dilation_kernel_size": int(cfg.edge_dilation_kernel_size),
            "hole_noise_prob": float(cfg.hole_noise_prob),
            "hole_block_size_min": int(cfg.hole_block_size_min),
            "hole_block_size": int(cfg.hole_block_size),
            "hole_white_prob": float(cfg.hole_white_prob),
            "dropout_prob": float(cfg.dropout_prob),
            "salt_pepper_prob": float(cfg.salt_pepper_prob),
            "gaussian_blur_enabled": bool(cfg.gaussian_blur_enabled),
            "gaussian_blur_ksize": int(cfg.gaussian_blur_ksize),
            "gaussian_blur_sigma": float(cfg.gaussian_blur_sigma),
            "gaussian_blur_order": "after_depth_noise",
            "gaussian_blur_space": "policy_depth_normalized",
            "near_clip_m": float(cfg.near_clip_m),
            "far_clip_m": float(cfg.far_clip_m),
        },
        "depth_camera_randomization_config": {
            "enabled": bool(getattr(args, "enable_depth_camera_randomization", False)),
            "pos_rand_m": float(getattr(args, "depth_camera_pos_rand_m", 0.01)),
            "rot_rand_deg": float(getattr(args, "depth_camera_rot_rand_deg", 2.0)),
        },
    }


def _rng_uniform(rng: Any, low: float, high: float, size: int | tuple[int, ...]):
    if rng is None:
        return np.random.uniform(low, high, size=size)
    if hasattr(rng, "uniform"):
        return rng.uniform(low, high, size=size)
    return np.random.uniform(low, high, size=size)


def _rng_int_inclusive(rng: Any, low: int, high: int) -> int:
    low = int(low)
    high = int(high)
    if high < low:
        low, high = high, low
    if rng is not None and hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))
    if rng is not None and hasattr(rng, "randint"):
        return int(rng.randint(low, high + 1))
    return int(np.random.randint(low, high + 1))


def jitter_camera_pose(
    pos: Any,
    rot: Any,
    rng: Any = None,
    *,
    enabled: bool = False,
    pos_range_m: float = 0.01,
    rot_range_deg: float = 2.0,
) -> tuple[list[float], list[float]]:
    pos_out = np.asarray(pos, dtype=np.float32).reshape(3).copy()
    rot_out = np.asarray(rot, dtype=np.float32).reshape(3).copy()
    if enabled:
        pos_out += np.asarray(_rng_uniform(rng, -float(pos_range_m), float(pos_range_m), 3), dtype=np.float32)
        rot_out += np.asarray(
            _rng_uniform(rng, -math.radians(float(rot_range_deg)), math.radians(float(rot_range_deg)), 3),
            dtype=np.float32,
        )
    return [float(x) for x in pos_out], [float(x) for x in rot_out]


def _config_from_mapping(cfg: DepthNoiseConfig | Mapping[str, Any]) -> DepthNoiseConfig:
    if isinstance(cfg, DepthNoiseConfig):
        return cfg
    return DepthNoiseConfig(**dict(cfg))


def _is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch"


def _torch_modules():
    import torch
    import torch.nn.functional as F

    return torch, F


def _sanitize_gaussian_blur_ksize(ksize: int) -> int:
    ksize = int(ksize)
    if ksize <= 0:
        return 0
    if ksize % 2 == 0:
        ksize += 1
    return max(1, ksize)


def _effective_gaussian_sigma(ksize: int, sigma: float) -> float:
    sigma = float(sigma)
    if sigma > 0.0:
        return sigma
    # OpenCV's common automatic sigma approximation when sigmaX=0.
    return 0.3 * ((float(ksize) - 1.0) * 0.5 - 1.0) + 0.8


def _apply_policy_depth_gaussian_blur_numpy(depth: np.ndarray, cfg: DepthNoiseConfig) -> np.ndarray:
    ksize = _sanitize_gaussian_blur_ksize(cfg.gaussian_blur_ksize)
    if not bool(cfg.gaussian_blur_enabled) or ksize <= 1 or depth.ndim < 2:
        return depth
    near = float(cfg.near_clip_m)
    far = float(cfg.far_clip_m)
    denom = max(far - near, 1.0e-6)
    policy_depth = np.zeros_like(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= near)
    policy_depth[valid] = np.clip((depth[valid] - near) / denom, 0.0, 1.0)
    sigma = float(cfg.gaussian_blur_sigma)
    try:
        import cv2  # type: ignore

        blurred = cv2.GaussianBlur(policy_depth, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    except Exception:
        sigma_eff = _effective_gaussian_sigma(ksize, sigma)
        radius = ksize // 2
        xs = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(xs * xs) / (2.0 * sigma_eff * sigma_eff)).astype(np.float32)
        kernel /= max(float(kernel.sum()), 1.0e-12)
        padded_x = np.pad(policy_depth, ((0, 0), (radius, radius)), mode="reflect")
        tmp = np.zeros_like(policy_depth, dtype=np.float32)
        for idx, weight in enumerate(kernel):
            tmp += float(weight) * padded_x[:, idx : idx + policy_depth.shape[-1]]
        padded_y = np.pad(tmp, ((radius, radius), (0, 0)), mode="reflect")
        blurred = np.zeros_like(policy_depth, dtype=np.float32)
        for idx, weight in enumerate(kernel):
            blurred += float(weight) * padded_y[idx : idx + policy_depth.shape[-2], :]
    return np.clip(near + np.asarray(blurred, dtype=np.float32) * denom, 0.0, far).astype(np.float32)


def _apply_policy_depth_gaussian_blur_torch(depth: Any, cfg: DepthNoiseConfig):
    ksize = _sanitize_gaussian_blur_ksize(cfg.gaussian_blur_ksize)
    if not bool(cfg.gaussian_blur_enabled) or ksize <= 1 or depth.ndim < 2:
        return depth
    torch, F = _torch_modules()
    if F is None:
        return depth
    near = float(cfg.near_clip_m)
    far = float(cfg.far_clip_m)
    denom = max(far - near, 1.0e-6)
    valid = torch.isfinite(depth) & (depth >= near)
    policy_depth = torch.where(valid, torch.clamp((depth - near) / denom, 0.0, 1.0), torch.zeros_like(depth))
    sigma = _effective_gaussian_sigma(ksize, float(cfg.gaussian_blur_sigma))
    radius = ksize // 2
    xs = torch.arange(-radius, radius + 1, device=depth.device, dtype=depth.dtype)
    kernel_1d = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / torch.clamp(kernel_1d.sum(), min=torch.finfo(depth.dtype).eps)
    original_shape = policy_depth.shape
    image_4d = policy_depth.reshape(-1, 1, policy_depth.shape[-2], policy_depth.shape[-1])
    pad_mode = "reflect" if image_4d.shape[-2] > radius and image_4d.shape[-1] > radius else "replicate"
    kernel_x = kernel_1d.reshape(1, 1, 1, ksize)
    kernel_y = kernel_1d.reshape(1, 1, ksize, 1)
    image_4d = F.pad(image_4d, (radius, radius, 0, 0), mode=pad_mode)
    image_4d = F.conv2d(image_4d, kernel_x)
    image_4d = F.pad(image_4d, (0, 0, radius, radius), mode=pad_mode)
    image_4d = F.conv2d(image_4d, kernel_y)
    blurred = image_4d.reshape(original_shape)
    return torch.clamp(near + blurred * denom, 0.0, far)


def apply_depth_noise(depth_m: Any, rng: Any, cfg: DepthNoiseConfig | Mapping[str, Any], valid_mask: Any = None):
    cfg = _config_from_mapping(cfg)
    if not cfg.enabled or cfg.noise_prob <= 0.0:
        return depth_m
    if cfg.env_selected is False:
        return depth_m
    if _is_torch_tensor(depth_m):
        return _apply_depth_noise_torch(depth_m, cfg, valid_mask)
    return _apply_depth_noise_numpy(depth_m, rng, cfg, valid_mask)


def _apply_depth_noise_numpy(depth_m: Any, rng: Any, cfg: DepthNoiseConfig, valid_mask: Any = None):
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    if cfg.env_selected is None and (
        rng.random() if rng is not None and hasattr(rng, "random") else np.random.random()
    ) >= cfg.noise_prob:
        return depth
    valid = np.isfinite(depth) & (depth >= cfg.near_clip_m) & (depth <= cfg.far_clip_m)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return depth
    if cfg.gaussian_std_m > 0.0:
        scale = cfg.gaussian_std_m * (1.0 + cfg.gaussian_distance_factor * depth)
        noise = (rng.normal(0.0, 1.0, depth.shape) if rng is not None and hasattr(rng, "normal") else np.random.normal(0.0, 1.0, depth.shape))
        depth[valid] += (noise.astype(np.float32) * scale.astype(np.float32))[valid]
    if cfg.edge_noise_prob > 0.0:
        grad_y, grad_x = np.gradient(depth)
        edge = np.sqrt(grad_x * grad_x + grad_y * grad_y) > cfg.edge_gradient_threshold_m
        if cfg.edge_dilation_kernel_size > 1:
            edge = _dilate_bool_numpy(edge, int(cfg.edge_dilation_kernel_size))
        random_edge = (
            rng.random(depth.shape) if rng is not None and hasattr(rng, "random") else np.random.random(depth.shape)
        ) < cfg.edge_noise_prob
        depth[edge & random_edge & valid] = 0.0
    if cfg.hole_noise_prob > 0.0:
        max_block = max(0, int(cfg.hole_block_size))
        min_block = min(max_block, max(0, int(cfg.hole_block_size_min)))
        block = _rng_int_inclusive(rng, min_block, max_block) if max_block > 0 else 0
        if block > 0:
            sparse_shape = (
                max((depth.shape[-2] + block - 1) // block, 1),
                max((depth.shape[-1] + block - 1) // block, 1),
            )
            sparse = (
                rng.random(sparse_shape) if rng is not None and hasattr(rng, "random") else np.random.random(sparse_shape)
            ) < cfg.hole_noise_prob
            sparse_white = (
                rng.random(sparse_shape) if rng is not None and hasattr(rng, "random") else np.random.random(sparse_shape)
            ) < min(1.0, max(0.0, float(cfg.hole_white_prob)))
            holes = np.repeat(np.repeat(sparse, block, axis=0), block, axis=1)[: depth.shape[-2], : depth.shape[-1]]
            white_holes = np.repeat(np.repeat(sparse_white, block, axis=0), block, axis=1)[
                : depth.shape[-2], : depth.shape[-1]
            ]
            depth[holes & white_holes & valid] = cfg.far_clip_m
            depth[holes & ~white_holes & valid] = 0.0
    if cfg.dropout_prob > 0.0:
        drop = (
            rng.random(depth.shape) if rng is not None and hasattr(rng, "random") else np.random.random(depth.shape)
        ) < cfg.dropout_prob
        depth[drop & valid] = 0.0
    if cfg.salt_pepper_prob > 0.0:
        rand = rng.random(depth.shape) if rng is not None and hasattr(rng, "random") else np.random.random(depth.shape)
        depth[(rand < cfg.salt_pepper_prob * 0.5) & valid] = cfg.far_clip_m
        depth[((rand >= cfg.salt_pepper_prob * 0.5) & (rand < cfg.salt_pepper_prob)) & valid] = 0.0
    depth = np.clip(depth, 0.0, cfg.far_clip_m).astype(np.float32)
    depth = _apply_policy_depth_gaussian_blur_numpy(depth, cfg)
    return np.clip(depth, 0.0, cfg.far_clip_m).astype(np.float32)


def _dilate_bool_numpy(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = max(0, kernel_size // 2)
    padded = np.pad(mask.astype(bool), ((pad, pad), (pad, pad)), mode="edge")
    out = np.zeros_like(mask, dtype=bool)
    for y in range(kernel_size):
        for x in range(kernel_size):
            out |= padded[y : y + mask.shape[0], x : x + mask.shape[1]]
    return out


def _apply_depth_noise_torch(depth_m: Any, cfg: DepthNoiseConfig, valid_mask: Any = None):
    torch, F = _torch_modules()
    depth = depth_m.clone()
    if cfg.env_selected is None:
        if depth.ndim >= 3:
            apply_shape = (*depth.shape[:-2], 1, 1)
        else:
            apply_shape = (1,) * depth.ndim
        apply_sample = torch.rand(apply_shape, device=depth.device) < float(cfg.noise_prob)
        if not bool(torch.any(apply_sample).detach().cpu().item()):
            return depth
    else:
        apply_sample = torch.ones(
            (*depth.shape[:-2], 1, 1) if depth.ndim >= 3 else (1,) * depth.ndim,
            dtype=torch.bool,
            device=depth.device,
        )
    valid = torch.isfinite(depth) & (depth >= float(cfg.near_clip_m)) & (depth <= float(cfg.far_clip_m))
    valid = valid & apply_sample
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=depth.device, dtype=torch.bool)
    if not bool(torch.any(valid).detach().cpu().item()):
        return depth
    if cfg.gaussian_std_m > 0.0:
        adaptive_std = float(cfg.gaussian_std_m) * (1.0 + float(cfg.gaussian_distance_factor) * depth)
        noise = torch.randn_like(depth) * adaptive_std
        depth = torch.where(valid, depth + noise, depth)
    if cfg.edge_noise_prob > 0.0 and depth.ndim >= 2:
        grad_x = torch.zeros_like(depth)
        grad_y = torch.zeros_like(depth)
        grad_x[..., :, 1:-1] = (depth[..., :, 2:] - depth[..., :, :-2]) * 0.5
        grad_x[..., :, 0] = depth[..., :, 1] - depth[..., :, 0]
        grad_x[..., :, -1] = depth[..., :, -1] - depth[..., :, -2]
        grad_y[..., 1:-1, :] = (depth[..., 2:, :] - depth[..., :-2, :]) * 0.5
        grad_y[..., 0, :] = depth[..., 1, :] - depth[..., 0, :]
        grad_y[..., -1, :] = depth[..., -1, :] - depth[..., -2, :]
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y) > float(cfg.edge_gradient_threshold_m)
        if cfg.edge_dilation_kernel_size > 1 and F is not None:
            original_shape = edge.shape
            edge_4d = edge.reshape(-1, 1, edge.shape[-2], edge.shape[-1]).float()
            edge = F.max_pool2d(
                edge_4d,
                kernel_size=int(cfg.edge_dilation_kernel_size),
                stride=1,
                padding=int(cfg.edge_dilation_kernel_size) // 2,
            ).reshape(original_shape).bool()
        edge_drop = torch.rand_like(depth) < float(cfg.edge_noise_prob)
        depth = torch.where(edge & edge_drop & valid, torch.zeros_like(depth), depth)
    if cfg.hole_noise_prob > 0.0 and depth.ndim >= 2 and F is not None:
        max_block = max(0, int(cfg.hole_block_size))
        min_block = min(max_block, max(0, int(cfg.hole_block_size_min)))
        block = (
            int(torch.randint(min_block, max_block + 1, (1,), device=depth.device).detach().cpu().item())
            if max_block > 0
            else 0
        )
        if block > 0:
            batch_shape = depth.shape[:-2]
            h, w = depth.shape[-2:]
            sparse_h, sparse_w = max((h + block - 1) // block, 1), max((w + block - 1) // block, 1)
            sparse = torch.rand((*batch_shape, sparse_h, sparse_w), device=depth.device) < float(cfg.hole_noise_prob)
            sparse_white = (
                torch.rand((*batch_shape, sparse_h, sparse_w), device=depth.device)
                < min(1.0, max(0.0, float(cfg.hole_white_prob)))
            )
            sparse_4d = sparse.reshape(-1, 1, sparse_h, sparse_w).float()
            holes = F.interpolate(sparse_4d, size=(h, w), mode="nearest").reshape(depth.shape).bool()
            sparse_white_4d = sparse_white.reshape(-1, 1, sparse_h, sparse_w).float()
            white_holes = F.interpolate(sparse_white_4d, size=(h, w), mode="nearest").reshape(depth.shape).bool()
            depth = torch.where(holes & white_holes & valid, torch.full_like(depth, float(cfg.far_clip_m)), depth)
            depth = torch.where(holes & ~white_holes & valid, torch.zeros_like(depth), depth)
    if cfg.dropout_prob > 0.0:
        dropout = torch.rand_like(depth) < float(cfg.dropout_prob)
        depth = torch.where(dropout & valid, torch.zeros_like(depth), depth)
    if cfg.salt_pepper_prob > 0.0:
        rand = torch.rand_like(depth)
        salt = rand < float(cfg.salt_pepper_prob) * 0.5
        pepper = (rand >= float(cfg.salt_pepper_prob) * 0.5) & (rand < float(cfg.salt_pepper_prob))
        depth = torch.where(salt & valid, torch.full_like(depth, float(cfg.far_clip_m)), depth)
        depth = torch.where(pepper & valid, torch.zeros_like(depth), depth)
    depth = torch.clamp(depth, 0.0, float(cfg.far_clip_m))
    depth = _apply_policy_depth_gaussian_blur_torch(depth, cfg)
    return torch.clamp(depth, 0.0, float(cfg.far_clip_m))
