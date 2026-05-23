#!/usr/bin/env python3
"""Evaluate a Door DP checkpoint on recorded expert observations.

This script does not start Isaac Gym. It loads one raw .npz episode, feeds
recorded observation histories into the diffusion policy, and compares the
predicted action chunk against the recorded expert actions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from door_dp_common import (  # noqa: E402
    ACTION_NAMES,
    DoorDPPolicyController,
    normalize_vision_mode,
    raw_image_keys_for_vision_mode,
)


@dataclass
class Metrics:
    checks: int = 0
    passed: int = 0
    rows: int = 0
    rows_all_passed: int = 0
    vx_abs: float = 0.0
    yaw_abs: float = 0.0
    pos_l2: float = 0.0
    quat_deg: float = 0.0
    gripper_abs: float = 0.0

    def update(self, expert: np.ndarray, pred: np.ndarray, args: argparse.Namespace) -> None:
        expert = np.asarray(expert, dtype=np.float32)
        pred = np.asarray(pred, dtype=np.float32)
        if expert.shape != pred.shape:
            raise ValueError(f"Shape mismatch: expert={expert.shape}, pred={pred.shape}")
        for exp_row, pred_row in zip(expert, pred):
            vx_err = abs(float(pred_row[0] - exp_row[0]))
            yaw_err = abs(float(pred_row[1] - exp_row[1]))
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
        self.checks += other.checks
        self.passed += other.passed
        self.rows += other.rows
        self.rows_all_passed += other.rows_all_passed
        self.vx_abs += other.vx_abs
        self.yaw_abs += other.yaw_abs
        self.pos_l2 += other.pos_l2
        self.quat_deg += other.quat_deg
        self.gripper_abs += other.gripper_abs

    def summary(self) -> str:
        if self.rows <= 0:
            return "n=0"
        return (
            f"n={self.rows} "
            f"accuracy={100.0 * self.passed / max(1, self.checks):.2f}% "
            f"all_action_acc={100.0 * self.rows_all_passed / max(1, self.rows):.2f}% "
            f"vx_mae={self.vx_abs / self.rows:.4f} "
            f"yaw_mae={self.yaw_abs / self.rows:.4f} "
            f"pos_l2={self.pos_l2 / self.rows:.4f} "
            f"quat_deg={self.quat_deg / self.rows:.2f} "
            f"grip_mae={self.gripper_abs / self.rows:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Door DP checkpoint on expert raw observations and compare predicted/expert actions."
    )
    parser.add_argument("--raw_episode", type=str, required=True, help="Path to one raw Door DP episode_*.npz file.")
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
    if ckpt_frame != raw_frame:
        raise ValueError(f"Checkpoint action_frame={ckpt_frame!r}, raw episode action_frame={raw_frame!r}.")
    raw_state_version = ikpush_state_version(data)
    ckpt_state_version = str(controller.config.get("ikpush_state_version", "legacy"))
    if ckpt_state_version != raw_state_version:
        raise ValueError(
            f"Checkpoint ikpush_state_version={ckpt_state_version!r}, "
            f"raw episode ikpush_state_version={raw_state_version!r}."
        )
    image_keys = raw_image_keys_for_vision_mode(expected_vision_mode)
    missing = [key for key in image_keys if key not in data.files]
    if missing:
        raise KeyError(f"Raw episode is missing image fields: {missing}")
    return raw_frame, raw_state_version, image_keys


def preload_episode_arrays(data, image_keys: list[str]) -> dict[str, np.ndarray]:
    keys = ["state", "action"] + list(image_keys)
    return {key: np.asarray(data[key]) for key in keys}


def episode_memory_mb(episode: dict[str, np.ndarray]) -> float:
    return sum(float(value.nbytes) for value in episode.values()) / (1024.0 * 1024.0)


def build_obs_cache(controller: DoorDPPolicyController, episode: dict[str, np.ndarray], image_keys: list[str]):
    cache = []
    total = int(episode["state"].shape[0])
    for idx in range(total):
        cache.append(
            controller._make_item(
                episode["state"][idx].astype(np.float32),
                episode[image_keys[0]][idx].astype(np.uint8),
                episode[image_keys[1]][idx].astype(np.uint8),
                episode[image_keys[2]][idx].astype(np.uint8),
                episode[image_keys[3]][idx].astype(np.uint8),
            )
        )
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
) -> np.ndarray:
    windows = [obs_window_from_cache(obs_cache, step, controller.obs_horizon) for step in steps]
    noise = initial_noise_for_steps(controller, steps, seed)
    action = controller.predict_action_chunks_from_windows(windows, noise=noise)
    action = action.detach().cpu().numpy().astype(np.float32)
    return action[:, :compare_horizon]


def reset_controller_on_expert_window(controller: DoorDPPolicyController, data, image_keys: list[str], step: int) -> None:
    controller.obs_buffer.clear()
    controller.action_queue.clear()
    first = max(0, int(step) - controller.obs_horizon + 1)
    for idx in range(first, int(step) + 1):
        controller.append_observation(
            data["state"][idx].astype(np.float32),
            data[image_keys[0]][idx].astype(np.uint8),
            data[image_keys[1]][idx].astype(np.uint8),
            data[image_keys[2]][idx].astype(np.uint8),
            data[image_keys[3]][idx].astype(np.uint8),
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
    raw_path = Path(args.raw_episode).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    expected_vision_mode = "rgb" if args.rgb else "depth"
    data = np.load(raw_path, allow_pickle=True)
    controller = DoorDPPolicyController(
        ckpt_path,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        action_horizon=args.action_horizon,
        noise_scheduler_type=args.noise_scheduler_type,
    )
    raw_frame, raw_state_version, image_keys = validate_inputs(data, controller, expected_vision_mode)
    t0 = time.perf_counter()
    episode = preload_episode_arrays(data, image_keys)
    data.close()
    print(f"Loaded raw arrays into memory: {episode_memory_mb(episode):.1f} MiB in {time.perf_counter() - t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    obs_cache = build_obs_cache(controller, episode, image_keys)
    print(f"Cached {len(obs_cache)} observation tensors on {controller.device} in {time.perf_counter() - t0:.2f}s", flush=True)

    actions = episode["action"].astype(np.float32)
    compare_horizon = int(args.compare_horizon or controller.action_horizon)
    compare_horizon = max(1, min(compare_horizon, controller.pred_horizon, controller.action_horizon))
    steps = build_eval_steps(args, actions.shape[0], controller.obs_horizon, compare_horizon)

    print(
        f"raw_episode={raw_path}\n"
        f"checkpoint={ckpt_path}\n"
        f"vision_mode={expected_vision_mode} action_frame={raw_frame} ikpush_state_version={raw_state_version} "
        f"obs_horizon={controller.obs_horizon} pred_horizon={controller.pred_horizon} "
        f"action_horizon={controller.action_horizon} compare_horizon={compare_horizon}\n"
        f"eval_steps={len(steps)} stride={args.stride if not args.steps else 'explicit'} "
        f"report_every={args.report_every} eval_batch_size={max(1, int(args.eval_batch_size))}",
        flush=True,
    )

    total = Metrics()
    segment = Metrics()
    segment_start = steps[0]
    next_report = segment_start + max(1, int(args.report_every))
    eval_t0 = time.perf_counter()
    batch_size = max(1, int(args.eval_batch_size))

    for batch_start in range(0, len(steps), batch_size):
        batch_steps = steps[batch_start : batch_start + batch_size]
        batch_pred = predict_action_chunks_batched(controller, obs_cache, batch_steps, args.seed, compare_horizon)
        for local_idx, step in enumerate(batch_steps):
            pred = batch_pred[local_idx]
            expert = actions[step : step + compare_horizon]
            current = Metrics()
            current.update(expert, pred, args)
            total.merge(current)
            segment.merge(current)
            if args.print_each or args.steps:
                print_step_detail(step, expert, pred)
            if step >= next_report:
                print(f"[{segment_start:05d}-{step:05d}] {segment.summary()}", flush=True)
                segment = Metrics()
                segment_start = step + 1
                while next_report <= step:
                    next_report += max(1, int(args.report_every))

    if segment.rows > 0:
        print(f"[{segment_start:05d}-{steps[-1]:05d}] {segment.summary()}", flush=True)
    print(f"TOTAL {total.summary()} elapsed={time.perf_counter() - eval_t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
