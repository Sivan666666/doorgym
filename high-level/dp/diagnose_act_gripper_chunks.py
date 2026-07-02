#!/usr/bin/env python3
"""Diagnose ACT gripper chunks on recorded Door-DP observations.

This is an offline diagnostic: it does not start Isaac Gym.  It loads one raw
episode, feeds recorded observation windows into a checkpoint, and plots the
predicted 100-step gripper chunk against the expert future gripper sequence.

The main question it answers is:

    "Does a single predicted chunk already contain close-open-close?"

Example:

  conda run --no-capture-output -n b1z1_lerobot python \
    high-level/dp/diagnose_act_gripper_chunks.py \
    --raw_episode high-level/data/door_dp_raw/a2w_state10_wc4_pitch_depthnoise_camerarand_noblur_100/episode_000000.npz \
    --checkpoint high-level/dp/logs/door-auto-wrapped/leroact_a2w_gating_keyframe_w3_r3_sample30_chunk100_exec50_bs16_0630_2234/050000/model_latest.pt \
    --device cuda:0 \
    --depth_only \
    --use_keyframes \
    --out_dir high-level/dp/result/gripper_chunk_diagnostics/gating_w3r3_50k_ep000000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency in the training env.
    plt = None

try:
    import cv2
except Exception:  # pragma: no cover - optional fallback.
    cv2 = None


SCRIPT_DIR = Path(__file__).resolve().parent
DP_ROOT = SCRIPT_DIR
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from door_dp_common import DoorDPPolicyController, normalize_vision_mode  # noqa: E402
from eval.eval_door_dp_on_expert_obs import (  # noqa: E402
    build_obs_cache,
    initial_noise_for_steps,
    obs_window_from_cache,
    preload_episode_arrays,
    raw_vision_mode,
    validate_inputs,
)
from play.play_door_policy import auto_wrap_official_lerobot_checkpoint  # noqa: E402


GRIPPER_OPEN = -1.5707963267948966


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and score ACT predicted gripper chunks on a raw Door-DP episode."
    )
    parser.add_argument("--raw_episode", type=str, required=True, help="Path to episode_*.npz.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Door checkpoint or LeRobot checkpoint.")
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
    parser.add_argument("--chunk_horizon", type=int, default=100, help="How many predicted chunk steps to analyze.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="Explicit observation indices.")
    parser.add_argument(
        "--use_keyframes",
        action="store_true",
        help="Use keyframe_indices from the raw episode when --steps is omitted.",
    )
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="high-level/dp/result/gripper_chunk_diagnostics/latest")
    parser.add_argument(
        "--close_threshold",
        type=float,
        default=-0.45,
        help="Predicted gripper values above this are considered meaningfully closed.",
    )
    parser.add_argument(
        "--reopen_delta",
        type=float,
        default=0.15,
        help="After closing, opening by this amount and closing again flags close-open-close.",
    )
    parser.add_argument(
        "--turn_delta",
        type=float,
        default=0.03,
        help="Ignore tiny gripper deltas below this when counting direction reversals.",
    )
    return parser.parse_args()


def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def build_steps(args: argparse.Namespace, data, total_frames: int, chunk_horizon: int) -> list[int]:
    max_step = max(0, int(total_frames) - max(1, int(chunk_horizon)))
    if args.steps:
        steps = [int(x) for x in args.steps]
    elif args.use_keyframes and "keyframe_indices" in data.files:
        steps = [int(x) for x in np.asarray(data["keyframe_indices"]).reshape(-1)]
        if "keyframe_names" in data.files:
            names = [str(x) for x in np.asarray(data["keyframe_names"]).reshape(-1)]
            print("Using raw keyframes:", flush=True)
            for idx, name in zip(steps, names):
                print(f"  {idx:04d}: {name}", flush=True)
    else:
        start = 0 if args.start is None else int(args.start)
        end = max_step + 1 if args.end is None else int(args.end)
        steps = list(range(start, min(end, max_step + 1), max(1, int(args.stride))))
    steps = [s for s in steps if 0 <= int(s) <= max_step]
    if len(steps) > int(args.max_steps):
        steps = steps[: int(args.max_steps)]
    if not steps:
        raise ValueError("No valid diagnostic steps selected.")
    return steps


def meaningful_turns(values: np.ndarray, eps: float) -> tuple[int, list[int]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    delta = np.diff(values)
    signs = np.zeros_like(delta, dtype=np.int8)
    signs[delta > float(eps)] = 1
    signs[delta < -float(eps)] = -1
    nonzero_idx = np.flatnonzero(signs)
    if len(nonzero_idx) <= 1:
        return 0, []
    turn_indices = []
    last_sign = int(signs[nonzero_idx[0]])
    for idx in nonzero_idx[1:]:
        sign = int(signs[idx])
        if sign != last_sign:
            turn_indices.append(int(idx + 1))
            last_sign = sign
    return len(turn_indices), turn_indices


def close_open_close_score(values: np.ndarray, close_threshold: float, reopen_delta: float) -> dict:
    """Detect a simple close-open-close pattern in one gripper chunk.

    The gripper convention here is open ~= -1.57 and more closed is larger.
    """
    g = np.asarray(values, dtype=np.float32).reshape(-1)
    closed_idxs = np.flatnonzero(g >= float(close_threshold))
    if len(closed_idxs) == 0:
        return {
            "close_open_close": False,
            "first_close_idx": -1,
            "reopen_idx": -1,
            "reclose_idx": -1,
            "first_close_value": float("nan"),
            "reopen_value": float("nan"),
            "reclose_value": float("nan"),
            "max_reopen_after_close": 0.0,
        }
    first_close = int(closed_idxs[0])
    closed_value = float(g[first_close])
    after_close = g[first_close + 1 :]
    if len(after_close) == 0:
        return {
            "close_open_close": False,
            "first_close_idx": first_close,
            "reopen_idx": -1,
            "reclose_idx": -1,
            "first_close_value": closed_value,
            "reopen_value": float("nan"),
            "reclose_value": float("nan"),
            "max_reopen_after_close": 0.0,
        }
    running_max_before = closed_value
    reopen_idx = -1
    reopen_value = float("nan")
    max_reopen = 0.0
    for local_idx, value in enumerate(after_close, start=first_close + 1):
        value = float(value)
        running_max_before = max(running_max_before, value)
        reopen = running_max_before - value
        max_reopen = max(max_reopen, reopen)
        if reopen >= float(reopen_delta):
            reopen_idx = int(local_idx)
            reopen_value = value
            break
    if reopen_idx < 0:
        return {
            "close_open_close": False,
            "first_close_idx": first_close,
            "reopen_idx": -1,
            "reclose_idx": -1,
            "first_close_value": closed_value,
            "reopen_value": float("nan"),
            "reclose_value": float("nan"),
            "max_reopen_after_close": float(max_reopen),
        }
    reclose_idx = -1
    reclose_value = float("nan")
    for idx in range(reopen_idx + 1, len(g)):
        value = float(g[idx])
        if value - reopen_value >= float(reopen_delta):
            reclose_idx = int(idx)
            reclose_value = value
            break
    return {
        "close_open_close": bool(reclose_idx >= 0),
        "first_close_idx": first_close,
        "reopen_idx": reopen_idx,
        "reclose_idx": reclose_idx,
        "first_close_value": closed_value,
        "reopen_value": reopen_value,
        "reclose_value": reclose_value,
        "max_reopen_after_close": float(max_reopen),
    }


def plot_chunk(
    out_path: Path,
    step: int,
    pred_gripper: np.ndarray,
    expert_gripper: np.ndarray,
    stats: dict,
    keyframe_name: str | None = None,
) -> None:
    if plt is None:
        plot_chunk_cv2(out_path, step, pred_gripper, expert_gripper, stats, keyframe_name)
        return
    x_pred = np.arange(len(pred_gripper), dtype=np.int32)
    x_exp = np.arange(len(expert_gripper), dtype=np.int32)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=140)
    ax.plot(x_pred, pred_gripper, label="predicted chunk gripper", linewidth=2.0)
    ax.plot(x_exp, expert_gripper, label="expert future gripper", linewidth=1.5, alpha=0.75)
    ax.axhline(GRIPPER_OPEN, linestyle="--", linewidth=1.0, color="gray", label="open ~= -1.57")
    ax.axhline(stats["close_threshold"], linestyle=":", linewidth=1.0, color="tab:red", label="close threshold")
    for field, color, label in (
        ("first_close_idx", "tab:green", "first close"),
        ("reopen_idx", "tab:orange", "reopen"),
        ("reclose_idx", "tab:red", "reclose"),
    ):
        idx = int(stats.get(field, -1))
        if idx >= 0:
            ax.axvline(idx, color=color, alpha=0.55, linewidth=1.1)
            ax.text(idx + 0.5, ax.get_ylim()[1], label, color=color, fontsize=8, va="top")
    title = f"step={step}"
    if keyframe_name:
        title += f" | {keyframe_name}"
    title += f" | close-open-close={stats['close_open_close']} turns={stats['turn_count']}"
    ax.set_title(title)
    ax.set_xlabel("chunk timestep")
    ax.set_ylabel("gripper action")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_chunk_cv2(
    out_path: Path,
    step: int,
    pred_gripper: np.ndarray,
    expert_gripper: np.ndarray,
    stats: dict,
    keyframe_name: str | None = None,
) -> None:
    if cv2 is None:
        return
    width, height = 1200, 580
    margin_l, margin_r, margin_t, margin_b = 78, 30, 74, 58
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    pred = np.asarray(pred_gripper, dtype=np.float32).reshape(-1)
    expert = np.asarray(expert_gripper, dtype=np.float32).reshape(-1)
    all_y = np.concatenate([pred, expert, np.asarray([GRIPPER_OPEN, stats["close_threshold"]], dtype=np.float32)])
    y_min = float(np.min(all_y) - 0.08)
    y_max = float(np.max(all_y) + 0.08)
    y_span = max(1e-6, y_max - y_min)
    x_max = max(1, max(len(pred), len(expert)) - 1)

    def xy(idx: int, value: float) -> tuple[int, int]:
        x = margin_l + int(round(float(idx) / x_max * (width - margin_l - margin_r)))
        y = margin_t + int(round((y_max - float(value)) / y_span * (height - margin_t - margin_b)))
        return x, y

    # Axes/grid.
    cv2.rectangle(img, (margin_l, margin_t), (width - margin_r, height - margin_b), (225, 225, 225), 1)
    for frac in np.linspace(0, 1, 6):
        y = margin_t + int(round(frac * (height - margin_t - margin_b)))
        cv2.line(img, (margin_l, y), (width - margin_r, y), (238, 238, 238), 1)
    for value, color, label in (
        (GRIPPER_OPEN, (130, 130, 130), "open -1.57"),
        (float(stats["close_threshold"]), (40, 40, 210), "close threshold"),
    ):
        y = xy(0, value)[1]
        cv2.line(img, (margin_l, y), (width - margin_r, y), color, 1, cv2.LINE_AA)
        cv2.putText(img, label, (margin_l + 8, max(margin_t + 14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1)

    def draw_curve(values: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        points = np.asarray([xy(i, float(v)) for i, v in enumerate(values)], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(img, [points], False, color, thickness, cv2.LINE_AA)

    draw_curve(expert, (220, 130, 20), 2)
    draw_curve(pred, (30, 90, 220), 3)

    for field, color, label in (
        ("first_close_idx", (30, 150, 30), "first close"),
        ("reopen_idx", (0, 150, 220), "reopen"),
        ("reclose_idx", (40, 40, 220), "reclose"),
    ):
        idx = int(stats.get(field, -1))
        if idx >= 0:
            x = xy(idx, y_min)[0]
            cv2.line(img, (x, margin_t), (x, height - margin_b), color, 1, cv2.LINE_AA)
            cv2.putText(img, label, (x + 4, margin_t + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    title = f"step={step}"
    if keyframe_name:
        title += f" | {keyframe_name}"
    title += f" | close-open-close={stats['close_open_close']} turns={stats['turn_count']}"
    cv2.putText(img, title[:150], (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, "blue=pred, orange=expert", (margin_l, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(img, "gripper action", (10, margin_t - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 60), 1)
    cv2.putText(img, "chunk timestep", (width // 2 - 70, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.imwrite(str(out_path), img)


@torch.no_grad()
def predict_chunks(controller: DoorDPPolicyController, obs_cache, steps: list[int], seed: int, horizon: int) -> np.ndarray:
    windows = [obs_window_from_cache(obs_cache, step, controller.obs_horizon) for step in steps]
    noise = initial_noise_for_steps(controller, steps, seed)
    action = controller.predict_action_chunks_from_windows(windows, noise=noise)
    action = action.detach().cpu().numpy().astype(np.float32)
    return action[:, :horizon]


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_episode).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rgb and args.depth_only:
        raise ValueError("--rgb and --depth_only are mutually exclusive.")
    data = np.load(raw_path, allow_pickle=True)
    raw_mode = raw_vision_mode(data)
    expected_vision_mode = "rgb" if args.rgb else ("depth_only" if args.depth_only else normalize_vision_mode(raw_mode))

    # Match the wrapping path used by play/eval scripts.
    args.rl_device = args.device
    args.dp_inference_steps = args.num_inference_steps
    args.dp_noise_scheduler_type = args.noise_scheduler_type
    wrapped_ckpt = Path(auto_wrap_official_lerobot_checkpoint(ckpt_path, args)).expanduser().resolve()
    controller = DoorDPPolicyController(
        wrapped_ckpt,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        action_horizon=args.action_horizon,
        noise_scheduler_type=args.noise_scheduler_type,
    )
    _raw_frame, _raw_state_version, image_keys = validate_inputs(data, controller, expected_vision_mode)
    chunk_horizon = max(1, min(int(args.chunk_horizon), int(controller.pred_horizon)))
    steps = build_steps(args, data, int(data["action"].shape[0]), chunk_horizon)

    keyframe_names_by_idx = {}
    if "keyframe_indices" in data.files and "keyframe_names" in data.files:
        for idx, name in zip(np.asarray(data["keyframe_indices"]).reshape(-1), np.asarray(data["keyframe_names"]).reshape(-1)):
            keyframe_names_by_idx[int(idx)] = str(name)

    t0 = time.perf_counter()
    episode = preload_episode_arrays(data, image_keys)
    data.close()
    obs_cache = build_obs_cache(controller, episode, image_keys)
    chunks = predict_chunks(controller, obs_cache, steps, args.seed, chunk_horizon)
    actions = episode["action"].astype(np.float32)
    print(
        f"Predicted {len(steps)} chunks from {raw_path.name} with {wrapped_ckpt} "
        f"in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    rows = []
    for row_idx, step in enumerate(steps):
        pred = chunks[row_idx]
        expert = actions[step : step + chunk_horizon]
        pred_gripper = pred[:, 9]
        expert_gripper = expert[:, 9]
        turn_count, turn_indices = meaningful_turns(pred_gripper, args.turn_delta)
        coc = close_open_close_score(pred_gripper, args.close_threshold, args.reopen_delta)
        stats = {
            "raw_episode": str(raw_path),
            "checkpoint": str(wrapped_ckpt),
            "step": int(step),
            "keyframe_name": keyframe_names_by_idx.get(int(step), ""),
            "horizon": int(chunk_horizon),
            "pred_first": float(pred_gripper[0]),
            "pred_last": float(pred_gripper[-1]),
            "pred_min": float(np.min(pred_gripper)),
            "pred_max": float(np.max(pred_gripper)),
            "pred_range": float(np.max(pred_gripper) - np.min(pred_gripper)),
            "expert_first": float(expert_gripper[0]),
            "expert_last": float(expert_gripper[-1]),
            "expert_min": float(np.min(expert_gripper)),
            "expert_max": float(np.max(expert_gripper)),
            "turn_count": int(turn_count),
            "turn_indices": json.dumps(turn_indices, ensure_ascii=False),
            "close_threshold": float(args.close_threshold),
            "reopen_delta": float(args.reopen_delta),
            **coc,
        }
        rows.append(stats)
        plot_chunk(
            out_dir / f"step_{int(step):04d}_gripper_chunk.png",
            int(step),
            pred_gripper,
            expert_gripper,
            stats,
            keyframe_names_by_idx.get(int(step)),
        )

    csv_path = out_dir / "gripper_chunk_summary.csv"
    json_path = out_dir / "gripper_chunk_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    flagged = [r for r in rows if bool(r["close_open_close"])]
    print(f"Saved plots and summaries to {out_dir}", flush=True)
    print(
        f"close-open-close flagged: {len(flagged)}/{len(rows)}; "
        f"mean turns={np.mean([r['turn_count'] for r in rows]):.2f}",
        flush=True,
    )
    if flagged:
        print("Flagged steps:", ", ".join(str(r["step"]) for r in flagged), flush=True)


if __name__ == "__main__":
    main()
