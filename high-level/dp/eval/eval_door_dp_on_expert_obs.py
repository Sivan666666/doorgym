#!/usr/bin/env python3
"""Evaluate a Door DP checkpoint on recorded expert observations.

This script does not start Isaac Gym. It loads one raw .npz episode, feeds
recorded observation histories into the diffusion policy, and compares the
predicted action chunk against the recorded expert actions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DP_ROOT = SCRIPT_DIR.parent
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from door_dp_common import (  # noqa: E402
    ACTION_NAMES,
    DEFAULT_END_SIGNAL_POSITIVE_PHASES,
    DoorDPPolicyController,
    RAW_END_SIGNAL_KEY,
    RAW_FRONT_CAMERA_POSE_KEY,
    RAW_WRIST_CAMERA_POSE_KEY,
    image_to_three_channel_uint8,
    make_end_signal_from_phase_ids,
    make_interaction_state_targets,
    normalize_vision_mode,
    raw_image_keys_for_vision_mode,
)
from play.play_door_policy import auto_wrap_official_lerobot_checkpoint  # noqa: E402


@dataclass
class Metrics:
    action_dim: int = 0
    checks: int = 0
    passed: int = 0
    rows: int = 0
    rows_all_passed: int = 0
    vx_abs: float = 0.0
    yaw_abs: float = 0.0
    pos_l2: float = 0.0
    quat_deg: float = 0.0
    gripper_abs: float = 0.0
    joint_l2: float = 0.0
    joint_max_abs: float = 0.0
    value_abs: float = 0.0
    value_sq: float = 0.0
    normalized_value_abs: float = 0.0
    values: int = 0
    per_dim_abs: list[float] = field(default_factory=list)

    def update(
        self,
        expert: np.ndarray,
        pred: np.ndarray,
        args: argparse.Namespace,
        *,
        normalized_expert: np.ndarray | None = None,
        normalized_pred: np.ndarray | None = None,
    ) -> None:
        expert = np.asarray(expert, dtype=np.float32)
        pred = np.asarray(pred, dtype=np.float32)
        if expert.shape != pred.shape:
            raise ValueError(f"Shape mismatch: expert={expert.shape}, pred={pred.shape}")
        self.action_dim = int(expert.shape[-1])
        abs_error = np.abs(pred - expert)
        self.value_abs += float(abs_error.sum())
        self.value_sq += float(np.square(pred - expert).sum())
        self.values += int(abs_error.size)
        per_dim = abs_error.reshape(-1, abs_error.shape[-1]).sum(axis=0)
        if not self.per_dim_abs:
            self.per_dim_abs = [0.0] * int(per_dim.shape[0])
        for idx, value in enumerate(per_dim):
            self.per_dim_abs[idx] += float(value)
        if normalized_expert is not None and normalized_pred is not None:
            self.normalized_value_abs += float(
                np.abs(np.asarray(normalized_pred) - np.asarray(normalized_expert)).sum()
            )
        for exp_row, pred_row in zip(expert, pred):
            vx_err = abs(float(pred_row[0] - exp_row[0]))
            yaw_err = abs(float(pred_row[1] - exp_row[1]))
            if self.action_dim == 9:
                joint_error = np.abs(pred_row[2:8] - exp_row[2:8])
                joint_l2 = float(np.linalg.norm(joint_error))
                joint_max_abs = float(np.max(joint_error))
                gripper_err = abs(float(pred_row[8] - exp_row[8]))
                pos_err = 0.0
                quat_err = 0.0
                passes = [
                    vx_err <= args.vx_tol,
                    yaw_err <= args.yaw_tol,
                    joint_max_abs <= args.joint_tol,
                    gripper_err <= args.gripper_tol,
                ]
                self.joint_l2 += joint_l2
                self.joint_max_abs += joint_max_abs
            elif self.action_dim >= 10:
                pos_err = float(np.linalg.norm(pred_row[2:5] - exp_row[2:5]))
                quat_err = quat_angle_deg(pred_row[5:9], exp_row[5:9])
                gripper_err = abs(float(pred_row[9] - exp_row[9]))
                passes = [
                    vx_err <= args.vx_tol,
                    yaw_err <= args.yaw_tol,
                    pos_err <= args.pos_tol,
                    quat_err <= args.quat_deg_tol,
                    gripper_err <= args.gripper_tol,
                ]
            else:
                raise ValueError(f"Unsupported action dimension for expert comparison: {self.action_dim}")
            self.passed += int(sum(passes))
            self.checks += len(passes)
            self.rows += 1
            self.rows_all_passed += int(all(passes))
            self.vx_abs += vx_err
            self.yaw_abs += yaw_err
            self.pos_l2 += pos_err
            self.quat_deg += quat_err
            self.gripper_abs += gripper_err

    def merge(self, other: "Metrics") -> None:
        if self.action_dim == 0:
            self.action_dim = other.action_dim
        elif other.action_dim not in (0, self.action_dim):
            raise ValueError(f"Cannot merge action_dim={other.action_dim} into action_dim={self.action_dim} metrics.")
        self.checks += other.checks
        self.passed += other.passed
        self.rows += other.rows
        self.rows_all_passed += other.rows_all_passed
        self.vx_abs += other.vx_abs
        self.yaw_abs += other.yaw_abs
        self.pos_l2 += other.pos_l2
        self.quat_deg += other.quat_deg
        self.gripper_abs += other.gripper_abs
        self.joint_l2 += other.joint_l2
        self.joint_max_abs += other.joint_max_abs
        self.value_abs += other.value_abs
        self.value_sq += other.value_sq
        self.normalized_value_abs += other.normalized_value_abs
        self.values += other.values
        if other.per_dim_abs:
            if not self.per_dim_abs:
                self.per_dim_abs = [0.0] * len(other.per_dim_abs)
            for idx, value in enumerate(other.per_dim_abs):
                self.per_dim_abs[idx] += float(value)

    def to_dict(self) -> dict:
        rows = max(1, self.rows)
        values = max(1, self.values)
        dim_rows = max(1, self.values // max(1, len(self.per_dim_abs)))
        return {
            "action_dim": int(self.action_dim),
            "rows": int(self.rows),
            "motion_l1": self.value_abs / values,
            "motion_rmse": float(np.sqrt(self.value_sq / values)),
            "normalized_motion_l1": self.normalized_value_abs / values,
            "per_dim_mae": [value / dim_rows for value in self.per_dim_abs],
            "vx_mae": self.vx_abs / rows,
            "yaw_mae": self.yaw_abs / rows,
            "position_l2_mean": self.pos_l2 / rows,
            "quaternion_deg_mean": self.quat_deg / rows,
            "gripper_mae": self.gripper_abs / rows,
            "joint_l2_mean": self.joint_l2 / rows if self.action_dim == 9 else None,
            "joint_max_abs_mean": self.joint_max_abs / rows if self.action_dim == 9 else None,
            "threshold_check_accuracy": self.passed / max(1, self.checks),
            "all_action_threshold_accuracy": self.rows_all_passed / rows,
        }

    def summary(self) -> str:
        if self.rows <= 0:
            return "n=0"
        common = (
            f"n={self.rows} "
            f"motion_l1={self.value_abs/max(1,self.values):.5f} "
            f"norm_l1={self.normalized_value_abs/max(1,self.values):.5f} "
            f"accuracy={100.0 * self.passed / max(1, self.checks):.2f}% "
            f"all_action_acc={100.0 * self.rows_all_passed / max(1, self.rows):.2f}% "
            f"vx_mae={self.vx_abs / self.rows:.4f} "
            f"yaw_mae={self.yaw_abs / self.rows:.4f} "
        )
        if self.action_dim == 9:
            return (
                common
                + f"joint_l2={self.joint_l2 / self.rows:.4f} "
                + f"joint_max={self.joint_max_abs / self.rows:.4f} "
                + f"grip_mae={self.gripper_abs / self.rows:.4f}"
            )
        return (
            common
            + f"pos_l2={self.pos_l2 / self.rows:.4f} "
            + f"quat_deg={self.quat_deg / self.rows:.2f} "
            + f"grip_mae={self.gripper_abs / self.rows:.4f}"
        )


@dataclass
class InteractionMetrics:
    rows: int = 0
    contact_tp: int = 0
    contact_fp: int = 0
    contact_fn: int = 0
    contact_tn: int = 0
    handle_abs: float = 0.0
    door_abs: float = 0.0
    handle_sq: float = 0.0
    door_sq: float = 0.0
    contact_probability_sum: float = 0.0
    contact_target_sum: float = 0.0

    def update(self, target: np.ndarray, pred: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float32).reshape(3)
        pred = np.asarray(pred, dtype=np.float32).reshape(3)
        gt_contact = bool(target[0] >= 0.5)
        pred_contact = bool(pred[0] >= 0.5)
        self.contact_tp += int(gt_contact and pred_contact)
        self.contact_fp += int(not gt_contact and pred_contact)
        self.contact_fn += int(gt_contact and not pred_contact)
        self.contact_tn += int(not gt_contact and not pred_contact)
        self.handle_abs += abs(float(pred[1] - target[1]))
        self.door_abs += abs(float(pred[2] - target[2]))
        self.handle_sq += float((pred[1] - target[1]) ** 2)
        self.door_sq += float((pred[2] - target[2]) ** 2)
        self.contact_probability_sum += float(pred[0])
        self.contact_target_sum += float(target[0])
        self.rows += 1

    def merge(self, other: "InteractionMetrics") -> None:
        for name in (
            "rows", "contact_tp", "contact_fp", "contact_fn", "contact_tn",
            "handle_abs", "door_abs", "handle_sq", "door_sq",
            "contact_probability_sum", "contact_target_sum",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> dict:
        precision = self.contact_tp / max(1, self.contact_tp + self.contact_fp)
        recall = self.contact_tp / max(1, self.contact_tp + self.contact_fn)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        return {
            "rows": int(self.rows),
            "contact_precision": precision,
            "contact_recall": recall,
            "contact_f1": f1,
            "contact_accuracy": (self.contact_tp + self.contact_tn) / max(1, self.rows),
            "contact_predicted_positive_ratio": (self.contact_tp + self.contact_fp) / max(1, self.rows),
            "contact_target_positive_ratio": (self.contact_tp + self.contact_fn) / max(1, self.rows),
            "contact_probability_mean": self.contact_probability_sum / max(1, self.rows),
            "handle_progress_mae": self.handle_abs / max(1, self.rows),
            "handle_progress_rmse": float(np.sqrt(self.handle_sq / max(1, self.rows))),
            "door_progress_mae": self.door_abs / max(1, self.rows),
            "door_progress_rmse": float(np.sqrt(self.door_sq / max(1, self.rows))),
        }

    def summary(self) -> str:
        precision = self.contact_tp / max(1, self.contact_tp + self.contact_fp)
        recall = self.contact_tp / max(1, self.contact_tp + self.contact_fn)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        return (
            f"n={self.rows} contact_precision={precision:.4f} contact_recall={recall:.4f} "
            f"contact_f1={f1:.4f} handle_mae={self.handle_abs/max(1,self.rows):.4f} "
            f"door_mae={self.door_abs/max(1,self.rows):.4f}"
        )


@dataclass
class EndSignalMetrics:
    rows: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    bce_sum: float = 0.0
    positive_probability_sum: float = 0.0
    negative_probability_sum: float = 0.0
    positives: int = 0
    negatives: int = 0

    def update(self, target: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> None:
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        if target.shape != pred.shape:
            raise ValueError(f"End-signal shape mismatch: target={target.shape}, pred={pred.shape}")
        pred = np.clip(pred, 1.0e-7, 1.0 - 1.0e-7)
        positive = target >= 0.5
        predicted = pred >= float(threshold)
        self.tp += int(np.count_nonzero(positive & predicted))
        self.fp += int(np.count_nonzero(~positive & predicted))
        self.fn += int(np.count_nonzero(positive & ~predicted))
        self.tn += int(np.count_nonzero(~positive & ~predicted))
        self.bce_sum += float((-(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred))).sum())
        self.positive_probability_sum += float(pred[positive].sum())
        self.negative_probability_sum += float(pred[~positive].sum())
        self.positives += int(np.count_nonzero(positive))
        self.negatives += int(np.count_nonzero(~positive))
        self.rows += int(target.size)

    def merge(self, other: "EndSignalMetrics") -> None:
        for name in (
            "rows", "tp", "fp", "fn", "tn", "bce_sum",
            "positive_probability_sum", "negative_probability_sum", "positives", "negatives",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> dict:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        return {
            "rows": int(self.rows),
            "bce": self.bce_sum / max(1, self.rows),
            "accuracy": (self.tp + self.tn) / max(1, self.rows),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "target_positive_ratio": self.positives / max(1, self.rows),
            "predicted_positive_ratio": (self.tp + self.fp) / max(1, self.rows),
            "positive_probability_mean": self.positive_probability_sum / max(1, self.positives),
            "negative_probability_mean": self.negative_probability_sum / max(1, self.negatives),
        }

    def summary(self) -> str:
        values = self.to_dict()
        return " ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in values.items()
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Door DP checkpoint on expert raw observations and compare predicted/expert actions."
    )
    raw_group = parser.add_mutually_exclusive_group(required=True)
    raw_group.add_argument("--raw_episode", type=str, help="Path to one raw Door DP episode_*.npz file.")
    raw_group.add_argument("--raw_root", type=str, help="Directory containing raw episode_*.npz files.")
    parser.add_argument("--episode_glob", type=str, default="episode_*.npz")
    parser.add_argument("--episode_stride", type=int, default=1, help="Select every Nth episode under --raw_root.")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None, help="Optional structured diagnostic report.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Door DP checkpoint, usually model_latest.pt.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--num_inference_steps",
        "--dp_inference_steps",
        dest="num_inference_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--noise_scheduler_type",
        "--dp_noise_scheduler_type",
        dest="noise_scheduler_type",
        type=str.upper,
        choices=["DDIM", "DDPM"],
        default="DDIM",
    )
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--compare_horizon", type=int, default=None, help="Actions per queried step to compare.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rgb", action="store_true", help="Require RGB+mask raw data/checkpoint.")
    parser.add_argument("--depth_only", action="store_true", help="Require wrist/front depth-only raw data/checkpoint.")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="Specific expert steps to evaluate.")
    parser.add_argument("--start", type=int, default=None, help="First expert step when --steps is omitted.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive final expert step when --steps is omitted.")
    parser.add_argument("--stride", type=int, default=25, help="Step interval when --steps is omitted.")
    parser.add_argument("--report_every", type=int, default=300, help="Print segment accuracy every N expert steps.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Number of eval points sampled per diffusion batch.")
    parser.add_argument("--print_each", action="store_true", help="Print first-action comparison for every queried step.")
    parser.add_argument("--vx_tol", type=float, default=0.03)
    parser.add_argument("--yaw_tol", type=float, default=0.03)
    parser.add_argument("--pos_tol", type=float, default=0.03)
    parser.add_argument("--quat_deg_tol", type=float, default=10.0)
    parser.add_argument("--gripper_tol", type=float, default=0.20)
    parser.add_argument("--joint_tol", type=float, default=0.15, help="Maximum absolute q1..q6 error for joint9 checkpoints.")
    parser.add_argument("--end_signal_threshold", type=float, default=0.5)
    return parser.parse_args()


def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def raw_vision_mode(data) -> str:
    if "vision_mode" in data.files:
        return normalize_vision_mode(scalar_str(data["vision_mode"]))
    if "wrist_rgb" in data.files or "front_rgb" in data.files:
        return "rgb"
    return "depth"


def action_frame(data) -> str:
    for key in ("action_frame", "action_pose_frame", "target_pose_frame"):
        if key in data.files:
            return scalar_str(data[key]).lower()
    return "world"


def ikpush_state_version(data) -> str:
    if "ikpush_state_version" in data.files:
        return scalar_str(data["ikpush_state_version"])
    return "legacy"


def controller_mode(data) -> str:
    for key in ("door_dp_mode", "controller_mode"):
        if key in data.files:
            return scalar_str(data[key])
    return "legacy"


def quat_angle_deg(q1, q2) -> float:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def build_eval_steps(args: argparse.Namespace, total_frames: int, obs_horizon: int, pred_horizon: int) -> list[int]:
    max_center = total_frames - 1
    if args.compare_horizon is None:
        max_center = total_frames - max(1, pred_horizon)
    else:
        max_center = total_frames - max(1, int(args.compare_horizon))
    if max_center < 0:
        raise ValueError("Episode is shorter than the requested comparison horizon.")
    if args.steps:
        steps = [int(x) for x in args.steps]
    else:
        start = int(args.start) if args.start is not None else max(0, obs_horizon - 1)
        end = int(args.end) if args.end is not None else max_center + 1
        steps = list(range(start, min(end, max_center + 1), max(1, int(args.stride))))
    steps = [s for s in steps if 0 <= s <= max_center]
    if not steps:
        raise ValueError("No valid eval steps selected.")
    return steps


def validate_inputs(data, controller: DoorDPPolicyController, expected_vision_mode: str) -> tuple[str, str, list[str]]:
    data_vision = raw_vision_mode(data)
    if data_vision != expected_vision_mode:
        raise ValueError(f"Raw episode vision_mode={data_vision!r}, but script expected {expected_vision_mode!r}.")
    if controller.vision_mode != expected_vision_mode:
        raise ValueError(
            f"Checkpoint vision_mode={controller.vision_mode!r}, but script expected {expected_vision_mode!r}."
        )
    raw_frame = action_frame(data)
    ckpt_frame = str(getattr(controller, "action_frame", "world")).lower()
    # Early Joint9 recordings inherited the EE pose-frame metadata even though
    # their actual action payload is [vx, vyaw, q1..q6, gripper].  For Joint9,
    # validate the payload/format rather than rejecting that stale sidecar
    # value.  EE10 recordings still require an exact frame match.
    action_dim = int(np.asarray(data["action"]).shape[-1])
    raw_action_format = scalar_str(data["action_format"]).lower() if "action_format" in data.files else ""
    joint9_payload = action_dim == 9 and (not raw_action_format or "joint" in raw_action_format)
    if ckpt_frame != raw_frame and not (joint9_payload and ckpt_frame == "joint_command"):
        raise ValueError(f"Checkpoint action_frame={ckpt_frame!r}, raw episode action_frame={raw_frame!r}.")
    raw_state_version = ikpush_state_version(data)
    ckpt_state_version = str(controller.config.get("ikpush_state_version", "legacy"))
    raw_state_format = scalar_str(data["state_format"]) if "state_format" in data.files else ""
    joint9_state_metadata_match = joint9_payload and raw_state_format == ckpt_state_version
    if ckpt_state_version != raw_state_version and not joint9_state_metadata_match:
        raise ValueError(
            f"Checkpoint ikpush_state_version={ckpt_state_version!r}, "
            f"raw episode ikpush_state_version={raw_state_version!r}."
        )
    if joint9_state_metadata_match:
        raw_state_version = raw_state_format
    raw_controller_mode = controller_mode(data)
    ckpt_controller_mode = str(controller.config.get("door_dp_mode", controller.config.get("controller_mode", "legacy")))
    if raw_controller_mode != ckpt_controller_mode:
        raise ValueError(
            f"Checkpoint door_dp_mode={ckpt_controller_mode!r}, raw episode door_dp_mode={raw_controller_mode!r}."
        )
    image_keys = raw_image_keys_for_vision_mode(expected_vision_mode)
    missing = [key for key in image_keys if key not in data.files]
    if missing:
        raise KeyError(f"Raw episode is missing image fields: {missing}")
    return raw_frame, raw_state_version, image_keys


def preload_episode_arrays(data, image_keys: list[str]) -> dict[str, np.ndarray]:
    keys = ["state", "action"] + list(image_keys)
    if RAW_FRONT_CAMERA_POSE_KEY in data.files and RAW_WRIST_CAMERA_POSE_KEY in data.files:
        keys += [RAW_FRONT_CAMERA_POSE_KEY, RAW_WRIST_CAMERA_POSE_KEY]
    for key in (
        "gripper_handle_contact_both",
        "replay_door_dof_pos",
        RAW_END_SIGNAL_KEY,
        "subtask_index",
        "phase_names",
        "door_asset_name",
    ):
        if key in data.files:
            keys.append(key)
    return {key: np.asarray(data[key]) for key in keys}


def resolve_raw_paths(args: argparse.Namespace) -> list[Path]:
    if args.raw_episode:
        return [Path(args.raw_episode).expanduser().resolve()]
    raw_root = Path(args.raw_root).expanduser().resolve()
    paths = sorted(raw_root.glob(str(args.episode_glob)))
    paths = paths[:: max(1, int(args.episode_stride))]
    if args.max_episodes is not None:
        paths = paths[: max(0, int(args.max_episodes))]
    if not paths:
        raise FileNotFoundError(f"No raw episodes matched {raw_root / args.episode_glob}.")
    return paths


def end_signal_targets_from_episode(episode: dict[str, np.ndarray]) -> np.ndarray:
    if RAW_END_SIGNAL_KEY in episode:
        return np.asarray(episode[RAW_END_SIGNAL_KEY], dtype=np.float32).reshape(-1, 1)
    missing = [key for key in ("subtask_index", "phase_names") if key not in episode]
    if missing:
        raise ValueError(f"End-signal offline metrics require raw fields {missing}.")
    return make_end_signal_from_phase_ids(
        episode["subtask_index"],
        episode["phase_names"],
        positive_phases=DEFAULT_END_SIGNAL_POSITIVE_PHASES,
    )


def normalized_actions(controller: DoorDPPolicyController, actions: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(actions, dtype=torch.float32, device=controller.device)
    normalized = controller.backend.normalizer.apply(tensor, "action", inverse=False)
    return normalized.detach().cpu().numpy().astype(np.float32)


def episode_memory_mb(episode: dict[str, np.ndarray]) -> float:
    return sum(float(value.nbytes) for value in episode.values()) / (1024.0 * 1024.0)


def make_controller_item(controller: DoorDPPolicyController, state: np.ndarray, episode, image_keys: list[str], idx: int):
    if bool(getattr(controller, "plucker_conditioning", False)):
        if RAW_FRONT_CAMERA_POSE_KEY not in episode or RAW_WRIST_CAMERA_POSE_KEY not in episode:
            raise ValueError("Plücker checkpoint eval requires raw episode camera pose arrays.")
        front_pose = episode[RAW_FRONT_CAMERA_POSE_KEY][idx]
        wrist_pose = episode[RAW_WRIST_CAMERA_POSE_KEY][idx]
    else:
        front_pose = None
        wrist_pose = None
    if controller.vision_mode == "depth_only":
        wrist_depth = image_to_three_channel_uint8(episode[image_keys[0]][idx])
        front_depth = image_to_three_channel_uint8(episode[image_keys[1]][idx])
        dummy_mask = np.zeros_like(wrist_depth)
        return controller._make_item(
            state.astype(np.float32),
            dummy_mask,
            wrist_depth,
            None,
            front_depth,
            front_pose,
            wrist_pose,
        )
    return controller._make_item(
        state.astype(np.float32),
        episode[image_keys[0]][idx].astype(np.uint8),
        episode[image_keys[1]][idx].astype(np.uint8),
        episode[image_keys[2]][idx].astype(np.uint8),
        episode[image_keys[3]][idx].astype(np.uint8),
        front_pose,
        wrist_pose,
    )


def build_obs_cache(controller: DoorDPPolicyController, episode: dict[str, np.ndarray], image_keys: list[str]):
    cache = []
    total = int(episode["state"].shape[0])
    for idx in range(total):
        cache.append(make_controller_item(controller, episode["state"][idx], episode, image_keys, idx))
    return cache


def obs_window_from_cache(obs_cache, step: int, obs_horizon: int):
    step = int(step)
    obs_horizon = int(obs_horizon)
    first = max(0, step - obs_horizon + 1)
    items = list(obs_cache[first : step + 1])
    if len(items) < obs_horizon:
        items = [obs_cache[0]] * (obs_horizon - len(items)) + items
    return items[-obs_horizon:]


def initial_noise_for_steps(controller: DoorDPPolicyController, steps: list[int], seed: int) -> torch.Tensor:
    rows = []
    for step in steps:
        generator = torch.Generator(device=controller.device)
        generator.manual_seed(int(seed) + int(step))
        rows.append(
            torch.randn(
                controller.pred_horizon,
                controller.action_dim,
                device=controller.device,
                generator=generator,
            )
        )
    return torch.stack(rows, dim=0)


@torch.no_grad()
def predict_action_chunks_batched(
    controller: DoorDPPolicyController,
    obs_cache,
    steps: list[int],
    seed: int,
    compare_horizon: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    windows = [obs_window_from_cache(obs_cache, step, controller.obs_horizon) for step in steps]
    noise = initial_noise_for_steps(controller, steps, seed)
    action = controller.predict_action_chunks_from_windows(windows, noise=noise)
    action = action.detach().cpu().numpy().astype(np.float32)
    motion_dim = int(controller.motion_action_dim)
    end_probability = None
    if bool(controller.end_signal_prediction):
        if action.shape[-1] <= motion_dim:
            raise RuntimeError("End-signal checkpoint did not append end probabilities to the predicted chunk.")
        end_probability = action[:, :compare_horizon, motion_dim].copy()
    motion_action = action[:, :compare_horizon, :motion_dim].copy()
    interaction = getattr(controller.backend, "last_interaction_state", None)
    interaction = None if interaction is None else np.asarray(interaction, dtype=np.float32).copy()
    return motion_action, end_probability, interaction


def reset_controller_on_expert_window(controller: DoorDPPolicyController, data, image_keys: list[str], step: int) -> None:
    controller.obs_buffer.clear()
    controller.action_queue.clear()
    first = max(0, int(step) - controller.obs_horizon + 1)
    for idx in range(first, int(step) + 1):
        if bool(getattr(controller, "plucker_conditioning", False)):
            if RAW_FRONT_CAMERA_POSE_KEY not in data.files or RAW_WRIST_CAMERA_POSE_KEY not in data.files:
                raise ValueError("Plücker checkpoint eval requires raw episode camera pose arrays.")
            front_pose = data[RAW_FRONT_CAMERA_POSE_KEY][idx]
            wrist_pose = data[RAW_WRIST_CAMERA_POSE_KEY][idx]
        else:
            front_pose = None
            wrist_pose = None
        if controller.vision_mode == "depth_only":
            wrist_depth = image_to_three_channel_uint8(data[image_keys[0]][idx])
            front_depth = image_to_three_channel_uint8(data[image_keys[1]][idx])
            controller.append_observation(
                data["state"][idx].astype(np.float32),
                np.zeros_like(wrist_depth),
                wrist_depth,
                None,
                front_depth,
                front_pose,
                wrist_pose,
            )
        else:
            controller.append_observation(
                data["state"][idx].astype(np.float32),
                data[image_keys[0]][idx].astype(np.uint8),
                data[image_keys[1]][idx].astype(np.uint8),
                data[image_keys[2]][idx].astype(np.uint8),
                data[image_keys[3]][idx].astype(np.uint8),
                front_pose,
                wrist_pose,
            )


def predict_action_chunk(
    controller: DoorDPPolicyController,
    data,
    image_keys: list[str],
    step: int,
    seed: int,
) -> np.ndarray:
    reset_controller_on_expert_window(controller, data, image_keys, step)
    noise = initial_noise_for_steps(controller, [step], seed)
    controller.sample_action_chunk(noise=noise)
    chunk = np.asarray(list(controller.action_queue), dtype=np.float32)
    controller.action_queue.clear()
    return chunk


def print_step_detail(step: int, expert: np.ndarray, pred: np.ndarray) -> None:
    exp0 = expert[0]
    pred0 = pred[0]
    if exp0.shape[0] == 9:
        print(
            f"step={step} expert(vx,yaw,q,grip)=({exp0[0]:.4f},{exp0[1]:.4f},"
            f"{np.round(exp0[2:8], 4).tolist()},{exp0[8]:.4f}) "
            f"pred=({pred0[0]:.4f},{pred0[1]:.4f},"
            f"{np.round(pred0[2:8], 4).tolist()},{pred0[8]:.4f})",
            flush=True,
        )
        return
    print(
        f"step={step} "
        f"expert(vx,yaw,target,grip)=({exp0[0]:.4f},{exp0[1]:.4f},"
        f"{np.round(exp0[2:5], 4).tolist()},{exp0[9]:.4f}) "
        f"pred=({pred0[0]:.4f},{pred0[1]:.4f},"
        f"{np.round(pred0[2:5], 4).tolist()},{pred0[9]:.4f})",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    raw_paths = resolve_raw_paths(args)
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if args.rgb and args.depth_only:
        raise ValueError("--rgb and --depth_only are mutually exclusive.")
    expected_vision_mode = "rgb" if args.rgb else ("depth_only" if args.depth_only else "depth")
    args.rl_device = args.device
    args.dp_inference_steps = args.num_inference_steps
    args.dp_noise_scheduler_type = args.noise_scheduler_type
    ckpt_path = Path(auto_wrap_official_lerobot_checkpoint(ckpt_path, args)).expanduser().resolve()
    controller = DoorDPPolicyController(
        ckpt_path,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        action_horizon=args.action_horizon,
        noise_scheduler_type=args.noise_scheduler_type,
    )
    total = Metrics()
    interaction_total = InteractionMetrics()
    end_total = EndSignalMetrics()
    episode_reports: list[dict] = []
    eval_t0 = time.perf_counter()
    batch_size = max(1, int(args.eval_batch_size))
    interaction_enabled = bool(getattr(controller.backend.config, "interaction_state_conditioning", False))
    end_enabled = bool(controller.end_signal_prediction)

    print(
        f"checkpoint={ckpt_path}\n"
        f"episodes={len(raw_paths)} vision_mode={expected_vision_mode} "
        f"obs_horizon={controller.obs_horizon} pred_horizon={controller.pred_horizon} "
        f"action_horizon={controller.action_horizon} end_signal={end_enabled} "
        f"interaction_state={interaction_enabled}",
        flush=True,
    )

    for episode_idx, raw_path in enumerate(raw_paths):
        data = np.load(raw_path, allow_pickle=True)
        raw_frame, raw_state_version, image_keys = validate_inputs(data, controller, expected_vision_mode)
        t0 = time.perf_counter()
        episode = preload_episode_arrays(data, image_keys)
        data.close()
        obs_cache = build_obs_cache(controller, episode, image_keys)
        actions = np.asarray(episode["action"], dtype=np.float32)
        compare_horizon = int(args.compare_horizon or controller.action_horizon)
        compare_horizon = max(1, min(compare_horizon, controller.pred_horizon, controller.action_horizon))
        steps = build_eval_steps(args, actions.shape[0], controller.obs_horizon, compare_horizon)

        interaction_targets = None
        if interaction_enabled:
            missing = [
                key for key in ("gripper_handle_contact_both", "replay_door_dof_pos") if key not in episode
            ]
            if missing:
                raise ValueError(f"Interaction-state offline metrics require raw fields {missing}.")
            targets = make_interaction_state_targets(
                episode["gripper_handle_contact_both"],
                episode["replay_door_dof_pos"],
                contact_min_consecutive_frames=3,
                handle_unlock_delta_rad=np.deg2rad(40.0),
                door_goal_delta_rad=np.deg2rad(90.0),
            )
            interaction_targets = np.concatenate(list(targets.values()), axis=1)
        end_targets = end_signal_targets_from_episode(episode) if end_enabled else None

        episode_motion = Metrics()
        episode_interaction = InteractionMetrics()
        episode_end = EndSignalMetrics()
        for batch_start in range(0, len(steps), batch_size):
            batch_steps = steps[batch_start : batch_start + batch_size]
            batch_pred, batch_end, batch_interaction = predict_action_chunks_batched(
                controller, obs_cache, batch_steps, args.seed + episode_idx * 100000, compare_horizon
            )
            for local_idx, step in enumerate(batch_steps):
                pred = batch_pred[local_idx, :, : actions.shape[1]]
                expert = actions[step : step + compare_horizon]
                pred_norm = normalized_actions(controller, pred)
                expert_norm = normalized_actions(controller, expert)
                current = Metrics()
                current.update(
                    expert,
                    pred,
                    args,
                    normalized_expert=expert_norm,
                    normalized_pred=pred_norm,
                )
                episode_motion.merge(current)
                if args.print_each or args.steps:
                    print_step_detail(step, expert, pred)
                if interaction_targets is not None:
                    if batch_interaction is None:
                        raise RuntimeError("Interaction-enabled checkpoint did not expose interaction predictions.")
                    episode_interaction.update(interaction_targets[step], batch_interaction[local_idx])
                if end_targets is not None:
                    if batch_end is None:
                        raise RuntimeError("End-signal checkpoint did not expose end probabilities.")
                    episode_end.update(
                        end_targets[step : step + compare_horizon, 0],
                        batch_end[local_idx],
                        threshold=args.end_signal_threshold,
                    )

        total.merge(episode_motion)
        interaction_total.merge(episode_interaction)
        end_total.merge(episode_end)
        door_name = scalar_str(episode.get("door_asset_name", "unknown"))
        report = {
            "episode": raw_path.name,
            "door": door_name,
            "frames": int(actions.shape[0]),
            "query_steps": len(steps),
            "motion": episode_motion.to_dict(),
            "end_signal": episode_end.to_dict() if end_enabled else None,
            "interaction_state": episode_interaction.to_dict() if interaction_enabled else None,
        }
        episode_reports.append(report)
        print(
            f"[{episode_idx + 1}/{len(raw_paths)}] {raw_path.name} door={door_name} "
            f"load_cache={time.perf_counter() - t0:.2f}s MOTION {episode_motion.summary()}",
            flush=True,
        )
        if end_enabled:
            print(f"  END {episode_end.summary()}", flush=True)
        if interaction_enabled:
            print(f"  INTERACTION {episode_interaction.summary()}", flush=True)
        del obs_cache, episode
        if controller.device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "checkpoint": str(ckpt_path),
        "raw_paths": [str(path) for path in raw_paths],
        "episodes": len(raw_paths),
        "stride": int(args.stride),
        "compare_horizon": int(args.compare_horizon or controller.action_horizon),
        "motion": total.to_dict(),
        "end_signal": end_total.to_dict() if end_enabled else None,
        "interaction_state": interaction_total.to_dict() if interaction_enabled else None,
        "per_episode": episode_reports,
    }
    print(f"TOTAL MOTION {total.summary()} elapsed={time.perf_counter() - eval_t0:.2f}s", flush=True)
    if end_enabled:
        print(f"TOTAL END {end_total.summary()}", flush=True)
    if interaction_enabled:
        print(f"TOTAL INTERACTION {interaction_total.summary()}", flush=True)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote diagnostic report: {output_path}", flush=True)


if __name__ == "__main__":
    main()
