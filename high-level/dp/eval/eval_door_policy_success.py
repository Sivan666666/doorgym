#!/usr/bin/env python3
"""Batch success-rate evaluation for Door ACT/DP/pi0.5 policy play.

This script is a thin wrapper around play_door_policy.py. It runs the normal
Isaac Gym play loop in batches, writes one policy JSONL log per batch, then
computes per-env success from the maximum observed door hinge angle.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DP_ROOT = SCRIPT_DIR.parent
HIGH_LEVEL_ROOT = DP_ROOT.parent
REPO_ROOT = HIGH_LEVEL_ROOT.parent
PLAY_SCRIPT = DP_ROOT / "play" / "play_door_policy.py"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from depth_camera_aug import add_depth_aug_args, add_depth_aug_command_args

PRINT_LOCK = Lock()
PROGRESS_RENDERER: "InlineProgress | None" = None


@dataclass(frozen=True)
class BatchSpec:
    batch_idx: int
    batch_envs: int
    remaining_before: int
    log_path: Path
    stdout_path: Path


@dataclass(frozen=True)
class BatchRunResult:
    spec: BatchSpec
    cmd: list[str]
    returncode: int
    elapsed_s: float


class InlineProgress:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._states: dict[int, str] = {}
        self._active = False
        self._last_len = 0

    def update(self, batch_idx: int, text: str) -> None:
        if not self.enabled:
            return
        with PRINT_LOCK:
            self._states[int(batch_idx)] = text
            self._render_locked()

    def clear(self) -> None:
        if not self.enabled:
            return
        with PRINT_LOCK:
            self._clear_locked()

    def remove(self, batch_idx: int) -> None:
        if not self.enabled:
            return
        with PRINT_LOCK:
            self._states.pop(int(batch_idx), None)
            self._render_locked()

    def _clear_locked(self) -> None:
        if not self._active:
            return
        sys.stdout.write("\r" + " " * max(1, self._last_len) + "\r")
        sys.stdout.flush()
        self._active = False

    def _render_locked(self) -> None:
        if not self._states:
            self._clear_locked()
            return
        line = " | ".join(self._states[idx] for idx in sorted(self._states))
        columns = max(40, shutil.get_terminal_size(fallback=(160, 24)).columns)
        if len(line) >= columns:
            line = line[: max(0, columns - 4)] + "..."
        sys.stdout.write("\r\x1b[2K" + line)
        sys.stdout.flush()
        self._active = True
        self._last_len = len(line)


def safe_print(message: str, *, end: str = "\n") -> None:
    with PRINT_LOCK:
        if PROGRESS_RENDERER is not None:
            PROGRESS_RENDERER._clear_locked()
        print(message, end=end, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Door policy success rate by repeatedly running play.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Door policy checkpoint (.pt manifest or directory).")
    parser.add_argument("--yaml", "--door_cfg", dest="door_cfg", required=True, type=str, help="Door YAML config.")
    parser.add_argument("--mode", choices=["ikpush", "ikpull", "push", "pull"], default="ikpush")
    parser.add_argument(
        "--robot_body",
        "--robot",
        dest="robot_body",
        choices=["b1z1", "a2wz1"],
        default="b1z1",
        help="Robot play script to evaluate. a2wz1 currently supports --mode ikpush.",
    )
    parser.add_argument("--num_envs", type=int, default=16, help="Number of envs per play run.")
    parser.add_argument("--total_trials", type=int, default=64, help="Total policy-controlled attempts to run.")
    parser.add_argument(
        "--parallel_batches",
        type=int,
        default=1,
        help="Number of play batches to run concurrently. Each batch is a separate Isaac Gym subprocess.",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--pass_open_angle_deg", type=float, default=80.0)
    parser.add_argument("--success_metric", choices=["auto", "signed", "abs"], default="auto")
    parser.add_argument("--door_motion_sign", type=float, default=-1.0)
    parser.add_argument("--base_seed", type=int, default=None, help="If set, run seed is base_seed + batch_index.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--dp_inference_steps", type=int, default=10)
    parser.add_argument("--dp_noise_scheduler_type", type=str.upper, choices=["DDIM", "DDPM"], default="DDIM")
    parser.add_argument("--dp_action_horizon", type=int, default=None)
    parser.add_argument(
        "--dp_temporal_ensemble",
        action="store_true",
        help="Forward NX-style action chunk overlap fusion to play_door_policy.py.",
    )
    parser.add_argument("--dp_temporal_prefetch_actions", type=int, default=3)
    parser.add_argument("--dp_temporal_old_weight", type=float, default=0.3)
    parser.add_argument("--dp_temporal_new_weight", type=float, default=0.7)
    parser.add_argument("--dp_end_signal_monitor", action="store_true")
    parser.add_argument("--dp_end_signal_threshold", type=float, default=0.8)
    parser.add_argument("--dp_end_signal_consecutive_steps", type=int, default=10)
    parser.add_argument(
        "--dp_fps",
        type=int,
        default=25,
        help="Policy observation/action update rate forwarded to the float_ik play script.",
    )
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--depth_only", dest="depth_only", action="store_true", default=True, help="Use wrist/front depth only, no mask image inputs.")
    parser.add_argument("--no_depth_only", dest="depth_only", action="store_false", help="Use legacy depth+mask image inputs.")
    add_depth_aug_args(parser)
    parser.add_argument("--camera_display_scale", type=int, default=1)
    parser.add_argument("--run_root", type=str, default=None, help="Directory for logs and summary JSON.")
    parser.add_argument("--stream_output", action="store_true", help="Stream each play subprocess output to this terminal.")
    parser.add_argument("--progress_interval", type=float, default=5.0, help="Seconds between per-batch progress updates.")
    parser.add_argument("--no_progress", action="store_true", help="Disable per-batch progress updates.")
    parser.add_argument("--print_policy_steps", action="store_true", help="Do not pass --no_dp_print to play.")
    parser.add_argument(
        "--dp_log_interval",
        type=int,
        default=25,
        help=(
            "Forwarded to play_door_policy.py. Use 1 for dense per-step logs. "
            "When --save_failure_rollouts is set, this is automatically clamped to <= --failure_snapshot_interval."
        ),
    )
    parser.add_argument(
        "--save_failure_rollouts",
        action="store_true",
        help=(
            "After evaluation, export failed env trajectories from the per-step JSONL logs into "
            "failure rollout .npz + metadata/diagnosis JSON files."
        ),
    )
    parser.add_argument(
        "--failure_rollout_root",
        type=str,
        default=None,
        help="Directory for exported failed rollout bundles. Defaults to <run_root>/failure_rollouts.",
    )
    parser.add_argument(
        "--failure_snapshot_interval",
        type=int,
        default=1,
        help="Keep every Nth logged frame when exporting failure rollouts.",
    )
    parser.add_argument(
        "--failure_save_camera_obs",
        action="store_true",
        help="Reserve camera observation fields in exported metadata when available in logs.",
    )
    parser.add_argument(
        "--failure_save_privileged_state",
        action="store_true",
        default=True,
        help="Export privileged state derived from policy JSONL logs.",
    )
    parser.add_argument(
        "--no_failure_save_privileged_state",
        dest="failure_save_privileged_state",
        action="store_false",
        help="Do not export privileged state fields.",
    )
    parser.add_argument(
        "--failure_save_events",
        action="store_true",
        default=True,
        help="Export event/contact/progress fields when present in policy JSONL logs.",
    )
    parser.add_argument(
        "--no_failure_save_events",
        dest="failure_save_events",
        action="store_false",
        help="Do not export event fields.",
    )
    parser.add_argument(
        "play_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the underlying play script after --.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def success_metric_for_mode(mode: str, metric: str) -> str:
    if metric != "auto":
        return metric
    # Use absolute hinge angle by default.  Different door assets encode their
    # hinge opening direction with different signs, and some assets (e.g. wc4)
    # flip args.door_motion_sign at runtime through per-door cfg multipliers.
    # The JSONL policy log stores raw hinge DOF, not the runtime signed
    # door_open_deg printed to stdout, so signed evaluation can count a
    # correctly-opened door as a failure.  Keep explicit --success_metric signed
    # available for direction-sensitive debugging.
    return "abs"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc


def door_open_deg(record: dict[str, Any], metric: str, door_motion_sign: float) -> float | None:
    door = record.get("door") or {}
    dof = door.get("dof")
    if not dof:
        return None
    hinge_rad = float(dof[0])
    hinge_deg = math.degrees(hinge_rad)
    if metric == "abs":
        return abs(hinge_deg)
    return float(door_motion_sign) * hinge_deg


def scan_log_progress(
    log_path: Path,
    threshold_deg: float,
    metric: str,
    door_motion_sign: float,
) -> dict[str, Any]:
    progress: dict[str, Any] = {
        "records": 0,
        "max_step": None,
        "env_ids": set(),
        "success_env_ids": set(),
        "max_open_deg_by_env": {},
    }
    if not log_path.exists():
        return progress
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            progress["records"] += 1
            try:
                step = int(record.get("step", -1))
            except (TypeError, ValueError):
                step = -1
            if step >= 0:
                current = progress["max_step"]
                progress["max_step"] = step if current is None else max(int(current), step)
            try:
                env_id = int(record.get("controlled_env_id", -1))
            except (TypeError, ValueError):
                env_id = -1
            if env_id >= 0:
                progress["env_ids"].add(env_id)
            open_deg = door_open_deg(record, metric=metric, door_motion_sign=door_motion_sign)
            if open_deg is None or env_id < 0:
                continue
            max_by_env = progress["max_open_deg_by_env"]
            max_by_env[env_id] = max(float(max_by_env.get(env_id, float("-inf"))), float(open_deg))
            if open_deg >= threshold_deg:
                progress["success_env_ids"].add(env_id)
    return progress


def format_progress_bar(current: int, total: int, width: int = 12) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    filled = int(round(width * current / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def format_batch_progress(spec: BatchSpec, steps: int, progress: dict[str, Any], *, done: bool, elapsed_s: float | None) -> str:
    max_step = progress.get("max_step")
    current_step = 0 if max_step is None else int(max_step)
    bar = format_progress_bar(current_step, int(steps))
    state = "D" if done else ("L" if max_step is None else "R")
    max_open = progress.get("max_open_deg_by_env") or {}
    best_open = None if not max_open else max(float(v) for v in max_open.values())
    elapsed = "" if elapsed_s is None else f" {elapsed_s:.0f}s"
    best = "" if best_open is None else f" best={best_open:.1f}deg"
    return (
        f"b{spec.batch_idx:04d} {state} {bar} "
        f"{current_step}/{int(steps)} "
        f"env={len(progress.get('env_ids', set()))}/{spec.batch_envs} "
        f"ok={len(progress.get('success_env_ids', set()))}/{spec.batch_envs}"
        f"{best}{elapsed}"
    )


def update_progress_line(spec: BatchSpec, steps: int, progress: dict[str, Any], *, done: bool, elapsed_s: float | None) -> None:
    if PROGRESS_RENDERER is None:
        return
    PROGRESS_RENDERER.update(
        spec.batch_idx,
        format_batch_progress(spec, steps, progress, done=done, elapsed_s=elapsed_s),
    )


def remove_progress_line(spec: BatchSpec) -> None:
    if PROGRESS_RENDERER is None:
        return
    PROGRESS_RENDERER.remove(spec.batch_idx)


def tail_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_chars))
        return f.read().decode("utf-8", errors="replace")


def summarize_log(
    log_path: Path,
    num_envs: int,
    threshold_deg: float,
    metric: str,
    door_motion_sign: float,
) -> list[dict[str, Any]]:
    stats = {
        env_id: {
            "env_id": env_id,
            "records": 0,
            "max_open_deg": float("-inf"),
            "first_success_step": None,
            "success": False,
            "end_triggered": False,
            "first_end_trigger_step": None,
            "open_deg_at_end_trigger": None,
            "phase_at_end_trigger": None,
            "interaction_records": 0,
            "interaction_contact_sum": 0.0,
            "interaction_handle_sum": 0.0,
            "interaction_door_sum": 0.0,
            "interaction_contact_max": None,
            "interaction_handle_max": None,
            "interaction_door_max": None,
        }
        for env_id in range(int(num_envs))
    }
    for record in read_jsonl(log_path):
        env_id = int(record.get("controlled_env_id", -1))
        if env_id not in stats:
            continue
        item = stats[env_id]
        item["records"] += 1
        open_deg = door_open_deg(record, metric=metric, door_motion_sign=door_motion_sign)
        end_record = record.get("end_signal") or {}
        interaction = record.get("interaction_state") or {}
        if interaction:
            contact = float(interaction.get("contact_probability", 0.0))
            handle = float(interaction.get("handle_progress", 0.0))
            door_progress = float(interaction.get("door_progress", 0.0))
            item["interaction_records"] += 1
            item["interaction_contact_sum"] += contact
            item["interaction_handle_sum"] += handle
            item["interaction_door_sum"] += door_progress
            item["interaction_contact_max"] = max(
                contact,
                contact if item["interaction_contact_max"] is None else float(item["interaction_contact_max"]),
            )
            item["interaction_handle_max"] = max(
                handle,
                handle if item["interaction_handle_max"] is None else float(item["interaction_handle_max"]),
            )
            item["interaction_door_max"] = max(
                door_progress,
                door_progress if item["interaction_door_max"] is None else float(item["interaction_door_max"]),
            )
        if bool(end_record.get("triggered", False)) and item["first_end_trigger_step"] is None:
            raw_trigger_step = end_record.get("first_trigger_step", record.get("step", -1))
            item["end_triggered"] = True
            item["first_end_trigger_step"] = int(raw_trigger_step)
            item["open_deg_at_end_trigger"] = None if open_deg is None else float(open_deg)
            item["phase_at_end_trigger"] = end_record.get("phase_at_trigger")
        if open_deg is None:
            continue
        item["max_open_deg"] = max(float(item["max_open_deg"]), float(open_deg))
        if open_deg >= threshold_deg and item["first_success_step"] is None:
            item["first_success_step"] = int(record.get("step", -1))
            item["success"] = True
    out = []
    for env_id in range(int(num_envs)):
        item = dict(stats[env_id])
        if item["max_open_deg"] == float("-inf"):
            item["max_open_deg"] = None
        success_step = item.get("first_success_step")
        trigger_step = item.get("first_end_trigger_step")
        item["end_trigger_delay_after_first_task_success"] = (
            None if success_step is None or trigger_step is None else int(trigger_step) - int(success_step)
        )
        interaction_count = int(item.pop("interaction_records"))
        for name in ("contact", "handle", "door"):
            total_value = float(item.pop(f"interaction_{name}_sum"))
            item[f"interaction_{name}_mean"] = (
                None if interaction_count == 0 else total_value / interaction_count
            )
        item["interaction_records"] = interaction_count
        out.append(item)
    return out


def build_play_command(args: argparse.Namespace, batch_envs: int, batch_idx: int, log_path: Path) -> list[str]:
    checkpoint = resolve_path(args.checkpoint)
    door_cfg = resolve_path(args.door_cfg)
    cmd = [
        sys.executable,
        str(PLAY_SCRIPT),
        "--mode",
        args.mode,
        "--robot_body",
        args.robot_body,
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        str(batch_envs),
        "--steps",
        str(args.steps),
        "--rl_device",
        args.rl_device,
        "--sim_device",
        args.sim_device,
        "--dp_control_all_envs",
        "--dp_inference_steps",
        str(args.dp_inference_steps),
        "--dp_noise_scheduler_type",
        args.dp_noise_scheduler_type,
        "--dp_log_path",
        str(log_path),
        "--dp_log_interval",
        str(args.dp_log_interval),
        "--no_show_seg",
    ]
    if args.dp_action_horizon is not None:
        cmd += ["--dp_action_horizon", str(args.dp_action_horizon)]
    if args.dp_temporal_ensemble:
        cmd.append("--dp_temporal_ensemble")
        cmd += ["--dp_temporal_prefetch_actions", str(args.dp_temporal_prefetch_actions)]
        cmd += ["--dp_temporal_old_weight", str(args.dp_temporal_old_weight)]
        cmd += ["--dp_temporal_new_weight", str(args.dp_temporal_new_weight)]
    if args.dp_end_signal_monitor:
        cmd.append("--dp_end_signal_monitor")
        cmd += ["--dp_end_signal_threshold", str(args.dp_end_signal_threshold)]
        cmd += ["--dp_end_signal_consecutive_steps", str(args.dp_end_signal_consecutive_steps)]
    if args.save_failure_rollouts:
        cmd.append("--dp_log_replay_snapshot")
    if args.rgb:
        cmd.append("--rgb")
    elif args.depth_only:
        cmd.append("--depth_only")
    if args.headless:
        cmd.append("--headless")
    if args.graphics_device_id is not None:
        cmd += ["--graphics_device_id", str(args.graphics_device_id)]
    if not args.print_policy_steps:
        cmd.append("--no_dp_print")
    cmd += ["--camera_display_scale", str(args.camera_display_scale)]
    add_depth_aug_command_args(cmd, args)
    cmd.append("--")
    cmd += [
        "--door_cfg",
        str(door_cfg),
        "--pass_open_angle_deg",
        str(args.pass_open_angle_deg),
        "--dp_fps",
        str(int(args.dp_fps)),
    ]
    if args.base_seed is not None:
        cmd += ["--seed", str(int(args.base_seed) + int(batch_idx))]
    extra = args.play_args[1:] if args.play_args[:1] == ["--"] else args.play_args
    cmd += extra
    return cmd


def _as_float_list(value: Any, length: int | None = None) -> list[float]:
    if value is None:
        out: list[float] = []
    elif isinstance(value, (list, tuple)):
        out = [float(x) for x in value]
    else:
        out = [float(value)]
    if length is not None:
        if len(out) < length:
            out.extend([float("nan")] * (length - len(out)))
        elif len(out) > length:
            out = out[:length]
    return out


def _records_for_env(log_path: Path, env_id: int, stride: int) -> list[dict[str, Any]]:
    stride = max(1, int(stride))
    records = [record for record in read_jsonl(log_path) if int(record.get("controlled_env_id", -1)) == int(env_id)]
    if stride <= 1:
        return records
    return [record for idx, record in enumerate(records) if idx % stride == 0]


def _collect_failure_npz_arrays(records: list[dict[str, Any]], *, save_privileged: bool, save_events: bool) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    if not records:
        return arrays

    arrays["step"] = np.asarray([int(r.get("step", -1)) for r in records], dtype=np.int32)
    arrays["controlled_env_id"] = np.asarray([int(r.get("controlled_env_id", -1)) for r in records], dtype=np.int32)
    arrays["state"] = np.asarray([_as_float_list(r.get("state"), 10) for r in records], dtype=np.float32)
    arrays["action"] = np.asarray([_as_float_list(r.get("dp_action_raw"), 10) for r in records], dtype=np.float32)
    arrays["applied_action"] = np.asarray([_as_float_list(r.get("applied_action"), 10) for r in records], dtype=np.float32)
    arrays["phase_name"] = np.asarray([str(r.get("phase_name") or "") for r in records])

    snapshot_keys = (
        "replay_root_state",
        "replay_dof_pos",
        "replay_dof_vel",
        "replay_ee_pos",
        "replay_ee_quat",
        "replay_door_root_state",
        "replay_box_root_state",
        "replay_door_dof_pos",
        "replay_door_dof_vel",
        "replay_door_open_stage",
        "front_camera_pose_base",
        "wrist_camera_pose_base",
    )
    for key in snapshot_keys:
        values = []
        found = False
        max_len = 0
        for record in records:
            snapshot = record.get("sim_snapshot") or {}
            value = snapshot.get(key)
            if value is not None:
                found = True
            value_list = _as_float_list(value, None)
            values.append(value_list)
            max_len = max(max_len, len(value_list))
        if found:
            value = np.asarray(
                [v + [float("nan")] * (max_len - len(v)) for v in values],
                dtype=np.float32,
            )
            arrays[f"sim_snapshot_{key}"] = value
            arrays[key] = value

    if save_privileged:
        arrays["privileged_base_xy"] = np.asarray(
            [_as_float_list((r.get("base") or {}).get("xy"), 2) for r in records],
            dtype=np.float32,
        )
        arrays["privileged_base_height"] = np.asarray(
            [float((r.get("base") or {}).get("height", float("nan"))) for r in records],
            dtype=np.float32,
        )
        arrays["privileged_base_lin_vel"] = np.asarray(
            [_as_float_list((r.get("base") or {}).get("lin_vel"), 3) for r in records],
            dtype=np.float32,
        )
        arrays["privileged_base_ang_vel"] = np.asarray(
            [_as_float_list((r.get("base") or {}).get("ang_vel"), 3) for r in records],
            dtype=np.float32,
        )
        arrays["privileged_ee_target_pos_world"] = np.asarray(
            [_as_float_list((r.get("ee") or {}).get("target_pos_world"), 3) for r in records],
            dtype=np.float32,
        )
        arrays["privileged_ee_actual_pos_world"] = np.asarray(
            [_as_float_list((r.get("ee") or {}).get("actual_pos_world"), 3) for r in records],
            dtype=np.float32,
        )
        arrays["privileged_ee_pos_error"] = np.asarray(
            [_as_float_list((r.get("ee") or {}).get("pos_error"), 3) for r in records],
            dtype=np.float32,
        )
        arrays["privileged_gripper_target"] = np.asarray(
            [_as_float_list((r.get("gripper") or {}).get("target"), None) for r in records],
            dtype=np.float32,
        )
        arrays["privileged_gripper_actual_pos"] = np.asarray(
            [_as_float_list((r.get("gripper") or {}).get("actual_pos"), None) for r in records],
            dtype=np.float32,
        )
        arrays["privileged_door_dof"] = np.asarray(
            [_as_float_list((r.get("door") or {}).get("dof"), 2) for r in records],
            dtype=np.float32,
        )

    if save_events:
        door_dof = arrays.get("privileged_door_dof")
        if door_dof is None:
            door_dof = np.asarray([_as_float_list((r.get("door") or {}).get("dof"), 2) for r in records], dtype=np.float32)
        door_abs = np.abs(np.nan_to_num(door_dof[:, 0], nan=0.0))
        handle_abs = np.abs(np.nan_to_num(door_dof[:, 1], nan=0.0))
        arrays["events_door_progress"] = door_abs.astype(np.float32)
        arrays["events_handle_progress"] = handle_abs.astype(np.float32)
        arrays["events_door_progress_delta"] = np.diff(door_abs, prepend=door_abs[:1]).astype(np.float32)
        arrays["events_handle_progress_delta"] = np.diff(handle_abs, prepend=handle_abs[:1]).astype(np.float32)

        extra_keys = (
            "gripper_handle_contact_score",
            "gripper_handle_contact_count",
            "gripper_handle_contact_any",
            "gripper_handle_contact_both",
        )
        for key in extra_keys:
            values = []
            found = False
            for record in records:
                extra = record.get("extra") or {}
                value = extra.get(key, record.get(key))
                if value is not None:
                    found = True
                values.append(_as_float_list(value, None))
            if found:
                max_len = max(1, max(len(v) for v in values))
                arrays[f"events_{key}"] = np.asarray(
                    [v + [float("nan")] * (max_len - len(v)) for v in values],
                    dtype=np.float32,
                )
    return arrays


def diagnose_failure_arrays(arrays: dict[str, Any], threshold_deg: float) -> dict[str, Any]:
    step = np.asarray(arrays.get("step", []), dtype=np.int32)
    door_dof = np.asarray(arrays.get("privileged_door_dof", np.zeros((len(step), 2), dtype=np.float32)), dtype=np.float32)
    ee_err = np.asarray(arrays.get("privileged_ee_pos_error", np.zeros((len(step), 3), dtype=np.float32)), dtype=np.float32)
    action = np.asarray(arrays.get("action", np.zeros((len(step), 10), dtype=np.float32)), dtype=np.float32)
    if len(step) == 0:
        return {
            "failure_type": "progress_stagnation",
            "t_dev": 0,
            "t_fail": 0,
            "recoverability": "unrecoverable",
            "candidate_recovery_family": "none",
            "evidence": ["empty failure rollout"],
        }

    door_abs_deg = np.abs(np.nan_to_num(door_dof[:, 0], nan=0.0)) * 180.0 / math.pi
    handle_abs_deg = np.abs(np.nan_to_num(door_dof[:, 1], nan=0.0)) * 180.0 / math.pi
    ee_err_norm = np.linalg.norm(np.nan_to_num(ee_err, nan=0.0), axis=1) if ee_err.ndim == 2 else np.zeros(len(step))
    gripper = action[:, 9] if action.ndim == 2 and action.shape[1] >= 10 else np.zeros(len(step))

    max_door = float(np.nanmax(door_abs_deg)) if len(door_abs_deg) else 0.0
    max_handle = float(np.nanmax(handle_abs_deg)) if len(handle_abs_deg) else 0.0
    max_ee_err = float(np.nanmax(ee_err_norm)) if len(ee_err_norm) else 0.0
    close_candidates = np.flatnonzero(gripper > -0.8)
    first_close_idx = int(close_candidates[0]) if len(close_candidates) else max(0, len(step) // 2)
    progress_candidates = np.flatnonzero(door_abs_deg > max(2.0, 0.05 * float(threshold_deg)))
    handle_candidates = np.flatnonzero(handle_abs_deg > 10.0)

    evidence: list[str] = [
        f"max_door_open_deg={max_door:.2f}",
        f"max_handle_or_secondary_dof_deg={max_handle:.2f}",
        f"max_ee_position_error_m={max_ee_err:.3f}",
    ]
    if len(close_candidates):
        evidence.append(f"gripper_close_like_action_first_step={int(step[first_close_idx])}")
    else:
        evidence.append("no_clear_gripper_close_action_detected")

    if max_door < max(5.0, 0.15 * float(threshold_deg)):
        if max_handle >= 20.0:
            failure_type = "insufficient_interaction"
            candidate_family = "maintain_grasp_then_push"
            evidence.append("handle/secondary DOF moved but hinge did not open enough")
            t_dev_idx = int(handle_candidates[0]) if len(handle_candidates) else first_close_idx
        elif max_ee_err > 0.08:
            failure_type = "geometric_misalignment"
            candidate_family = "retreat_realign_reapproach"
            evidence.append("large EE tracking/target error while door progress stayed low")
            err_candidates = np.flatnonzero(ee_err_norm > 0.08)
            t_dev_idx = int(err_candidates[0]) if len(err_candidates) else first_close_idx
        else:
            failure_type = "contact_establishment_failure"
            candidate_family = "reopen_retreat_reapproach"
            evidence.append("door hinge stayed near zero after gripper close")
            t_dev_idx = first_close_idx
    elif max_door < float(threshold_deg):
        failure_type = "progress_stagnation"
        candidate_family = "continue_push_with_reposition"
        evidence.append("door moved partially but did not reach success threshold")
        t_dev_idx = int(progress_candidates[0]) if len(progress_candidates) else first_close_idx
    else:
        failure_type = "temporal_coordination_failure"
        candidate_family = "phase_timing_adjustment"
        evidence.append("rollout marked failed despite reaching threshold; check metric/sign/timeout")
        t_dev_idx = first_close_idx

    stagnation_window = 25
    t_fail_idx = len(step) - 1
    if len(door_abs_deg) > stagnation_window:
        # Failure confirmation must follow the earliest deviation by a full
        # observation window. Otherwise a quiet window immediately before
        # t_dev incorrectly collapses t_fail onto t_dev.
        first_confirm_idx = max(t_dev_idx + stagnation_window, stagnation_window)
        for idx in range(first_confirm_idx, len(door_abs_deg)):
            recent = door_abs_deg[max(0, idx - stagnation_window) : idx + 1]
            if float(np.max(recent) - np.min(recent)) < 1.0 and float(np.max(recent)) < float(threshold_deg):
                t_fail_idx = idx
                break

    recoverability = "reactive_recoverable"
    if max_ee_err > 0.5:
        recoverability = "unrecoverable"
        evidence.append("EE error is extremely large; likely unsafe or infeasible without reset")
    elif int(step[t_dev_idx]) < int(step[t_fail_idx]):
        recoverability = "preventive_recoverable" if failure_type in {"geometric_misalignment", "progress_stagnation"} else "reactive_recoverable"

    return {
        "failure_type": failure_type,
        "t_dev": int(step[t_dev_idx]),
        "t_fail": int(step[t_fail_idx]),
        "recoverability": recoverability,
        "candidate_recovery_family": candidate_family,
        "evidence": evidence,
        "metrics": {
            "max_door_open_deg": max_door,
            "max_handle_or_secondary_dof_deg": max_handle,
            "max_ee_position_error_m": max_ee_err,
            "success_threshold_deg": float(threshold_deg),
        },
    }


def export_failure_rollout_bundle(
    *,
    records: list[dict[str, Any]],
    out_dir: Path,
    trial: dict[str, Any],
    args: argparse.Namespace,
    metric: str,
    cmd: list[str] | None,
) -> dict[str, Any] | None:
    if not records:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = _collect_failure_npz_arrays(
        records,
        save_privileged=bool(args.failure_save_privileged_state),
        save_events=bool(args.failure_save_events),
    )
    arrays["success_metrics_success"] = np.asarray([0], dtype=np.int8)
    arrays["success_metrics_max_open_deg"] = np.asarray([float(trial.get("max_open_deg") or 0.0)], dtype=np.float32)
    arrays["success_metrics_first_success_step"] = np.asarray([-1], dtype=np.int32)

    npz_path = out_dir / "failure_rollout.npz"
    np.savez_compressed(npz_path, **arrays)
    diagnosis = diagnose_failure_arrays(arrays, float(args.pass_open_angle_deg))
    diagnosis_path = out_dir / "diagnosis.json"
    with diagnosis_path.open("w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2, sort_keys=True)

    metadata = {
        "schema_version": 1,
        "source": {
            "log_path": str(trial.get("log_path", "")),
            "stdout_path": str(trial.get("stdout_path", "")),
            "batch": int(trial.get("batch", -1)),
            "env_id": int(trial.get("env_id", -1)),
            "global_trial": int(trial.get("global_trial", -1)),
        },
        "eval": {
            "checkpoint": str(resolve_path(args.checkpoint)),
            "door_cfg": str(resolve_path(args.door_cfg)),
            "mode": args.mode,
            "robot_body": args.robot_body,
            "base_seed": args.base_seed,
            "success_metric": metric,
            "pass_open_angle_deg": float(args.pass_open_angle_deg),
            "dp_action_horizon": args.dp_action_horizon,
            "dp_fps": int(args.dp_fps),
            "steps": int(args.steps),
        },
        "failure_export": {
            "snapshot_interval": int(args.failure_snapshot_interval),
            "save_camera_obs_requested": bool(args.failure_save_camera_obs),
            "camera_obs_available": False,
            "save_privileged_state": bool(args.failure_save_privileged_state),
            "save_events": bool(args.failure_save_events),
            "records": int(len(records)),
            "sim_snapshot_available": "sim_snapshot_replay_root_state" in arrays,
            "npz_path": str(npz_path),
            "diagnosis_path": str(diagnosis_path),
            "note": (
                "This bundle is exported from policy JSONL logs. It contains dense actions, "
                "privileged/event signals, and replay-style sim_snapshot fields when the play loop "
                "was run with --dp_log_replay_snapshot."
            ),
        },
        "play_command": cmd or [],
        "success_metrics": {
            "success": False,
            "max_open_deg": None if trial.get("max_open_deg") is None else float(trial.get("max_open_deg")),
            "first_success_step": trial.get("first_success_step"),
        },
    }
    randomization = next(
        (record.get("ikpush_randomization") for record in records if record.get("ikpush_randomization")),
        None,
    )
    if randomization is not None:
        metadata["ikpush_randomization"] = randomization
    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return {
        "global_trial": int(trial.get("global_trial", -1)),
        "batch": int(trial.get("batch", -1)),
        "env_id": int(trial.get("env_id", -1)),
        "dir": str(out_dir),
        "npz_path": str(npz_path),
        "metadata_path": str(metadata_path),
        "diagnosis_path": str(diagnosis_path),
        "diagnosis": diagnosis,
    }


def export_failed_rollouts(
    *,
    args: argparse.Namespace,
    run_root: Path,
    all_trials: list[dict[str, Any]],
    batch_logs: dict[int, dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    if not bool(args.save_failure_rollouts):
        return []
    failure_root = Path(args.failure_rollout_root).expanduser() if args.failure_rollout_root else run_root / "failure_rollouts"
    if not failure_root.is_absolute():
        failure_root = (Path.cwd() / failure_root).resolve()
    failure_root.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    for trial in all_trials:
        if bool(trial.get("success")):
            continue
        log_path = Path(str(trial.get("log_path", ""))).expanduser()
        if not log_path.is_file():
            continue
        env_id = int(trial.get("env_id", -1))
        records = _records_for_env(log_path, env_id, int(args.failure_snapshot_interval))
        out_dir = failure_root / f"failure_trial_{int(trial.get('global_trial', -1)):06d}_batch{int(trial.get('batch', -1)):04d}_env{env_id:02d}"
        batch_info = batch_logs.get(int(trial.get("batch", -1)), {})
        item = export_failure_rollout_bundle(
            records=records,
            out_dir=out_dir,
            trial=trial,
            args=args,
            metric=metric,
            cmd=batch_info.get("command"),
        )
        if item is not None:
            exported.append(item)
    manifest_path = failure_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "failure_rollout_count": len(exported),
                "failure_rollouts": exported,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    safe_print(f"Failure rollouts exported: {len(exported)} -> {manifest_path}")
    return exported


def pump_subprocess_output(proc: subprocess.Popen, stdout_path: Path, stream_output: bool, batch_idx: int) -> None:
    if proc.stdout is None:
        return
    with stdout_path.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if stream_output:
                safe_print(f"[batch {batch_idx:04d}] {line}", end="")


def run_batch_job(args: argparse.Namespace, spec: BatchSpec, metric: str) -> BatchRunResult:
    cmd = build_play_command(args, spec.batch_envs, spec.batch_idx, spec.log_path)
    safe_print(
        f"[batch {spec.batch_idx:04d}] start envs={spec.batch_envs} "
        f"remaining_before={spec.remaining_before} stdout={spec.stdout_path}"
    )
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    stdout_thread = Thread(
        target=pump_subprocess_output,
        args=(proc, spec.stdout_path, bool(args.stream_output), spec.batch_idx),
        daemon=True,
    )
    stdout_thread.start()

    progress_interval = max(0.5, float(args.progress_interval))
    last_progress_time = 0.0
    while True:
        returncode = proc.poll()
        now = time.monotonic()
        if not args.no_progress and now - last_progress_time >= progress_interval:
            progress = scan_log_progress(
                spec.log_path,
                threshold_deg=float(args.pass_open_angle_deg),
                metric=metric,
                door_motion_sign=float(args.door_motion_sign),
            )
            update_progress_line(spec, int(args.steps), progress, done=False, elapsed_s=now - start)
            last_progress_time = now
        if returncode is not None:
            break
        time.sleep(min(1.0, progress_interval))

    stdout_thread.join(timeout=5.0)
    elapsed = time.monotonic() - start
    if not args.no_progress:
        progress = scan_log_progress(
            spec.log_path,
            threshold_deg=float(args.pass_open_angle_deg),
            metric=metric,
            door_motion_sign=float(args.door_motion_sign),
        )
        update_progress_line(spec, int(args.steps), progress, done=True, elapsed_s=elapsed)
        remove_progress_line(spec)
    return BatchRunResult(spec=spec, cmd=cmd, returncode=int(returncode), elapsed_s=elapsed)


def main() -> None:
    global PROGRESS_RENDERER
    args = parse_args()
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if args.total_trials <= 0:
        raise ValueError("--total_trials must be positive.")
    if args.parallel_batches <= 0:
        raise ValueError("--parallel_batches must be positive.")
    if args.progress_interval <= 0:
        raise ValueError("--progress_interval must be positive.")
    if args.dp_log_interval <= 0:
        raise ValueError("--dp_log_interval must be positive.")
    if args.failure_snapshot_interval <= 0:
        raise ValueError("--failure_snapshot_interval must be positive.")
    if args.save_failure_rollouts:
        args.dp_log_interval = min(int(args.dp_log_interval), int(args.failure_snapshot_interval))
    if args.rgb:
        args.depth_only = False
    if args.dp_fps <= 0:
        raise ValueError("--dp_fps must be positive.")
    if args.steps is None:
        args.steps = 4300 if args.mode == "ikpull" else (2405 if args.mode == "ikpush" else 2500)
    PROGRESS_RENDERER = InlineProgress(enabled=not args.no_progress)

    metric = success_metric_for_mode(args.mode, args.success_metric)
    run_root = Path(args.run_root).expanduser() if args.run_root else HIGH_LEVEL_ROOT / "logs" / "door-policy-success" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not run_root.is_absolute():
        run_root = (Path.cwd() / run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    remaining = int(args.total_trials)
    batch_idx = 0
    specs: list[BatchSpec] = []
    while remaining > 0:
        batch_envs = min(int(args.num_envs), remaining)
        specs.append(
            BatchSpec(
                batch_idx=batch_idx,
                batch_envs=batch_envs,
                remaining_before=remaining,
                log_path=run_root / f"batch_{batch_idx:04d}.jsonl",
                stdout_path=run_root / f"batch_{batch_idx:04d}.out",
            )
        )
        remaining -= batch_envs
        batch_idx += 1

    safe_print(
        f"Door policy success eval: checkpoint={resolve_path(args.checkpoint)} door_cfg={resolve_path(args.door_cfg)}\n"
        f"mode={args.mode} robot_body={args.robot_body} num_envs={args.num_envs} total_trials={args.total_trials} "
        f"steps={args.steps} threshold={args.pass_open_angle_deg}deg metric={metric} "
        f"vision_mode={'rgb' if args.rgb else ('depth_only' if args.depth_only else 'depth')} "
        f"dp_action_horizon={args.dp_action_horizon} dp_temporal_ensemble={args.dp_temporal_ensemble} "
        f"dp_fps={args.dp_fps:g} dp_log_interval={args.dp_log_interval} "
        f"parallel_batches={args.parallel_batches} headless={args.headless} run_root={run_root}"
    )
    if args.parallel_batches > 1:
        safe_print(
            "Warning: --parallel_batches launches multiple full Isaac Gym play subprocesses. "
            "Each subprocess loads its own simulator, cameras, and policy, so GPU memory and render contention can increase quickly."
        )

    batch_trials: dict[int, list[dict[str, Any]]] = {}
    batch_logs: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=int(args.parallel_batches)) as executor:
        futures = {executor.submit(run_batch_job, args, spec, metric): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            result = future.result()
            remove_progress_line(spec)
            if result.returncode != 0:
                progress = scan_log_progress(
                    spec.log_path,
                    threshold_deg=float(args.pass_open_angle_deg),
                    metric=metric,
                    door_motion_sign=float(args.door_motion_sign),
                )
                max_step = progress.get("max_step")
                completed_steps = max_step is not None and int(max_step) >= max(0, int(args.steps) - 1)
                completed_envs = len(progress.get("env_ids", set())) >= int(spec.batch_envs)
                if completed_steps and completed_envs:
                    safe_print(
                        f"[batch {spec.batch_idx:04d}] warning: play subprocess exited with code "
                        f"{result.returncode}, but policy log reached step {max_step}/{int(args.steps)} "
                        f"for {len(progress.get('env_ids', set()))}/{spec.batch_envs} envs. "
                        "Treating this as a completed batch; Isaac Gym can segfault during shutdown."
                    )
                else:
                    tail = tail_text(spec.stdout_path)
                    if tail:
                        safe_print(tail)
                    raise RuntimeError(f"Play batch {spec.batch_idx} failed with exit code {result.returncode}.")
            if not spec.log_path.exists():
                tail = tail_text(spec.stdout_path)
                if tail:
                    safe_print(tail)
                raise FileNotFoundError(f"Expected policy log was not written: {spec.log_path}")
            batch_stats = summarize_log(
                spec.log_path,
                num_envs=spec.batch_envs,
                threshold_deg=float(args.pass_open_angle_deg),
                metric=metric,
                door_motion_sign=float(args.door_motion_sign),
            )
            for item in batch_stats:
                item["batch"] = spec.batch_idx
                item["batch_trial"] = int(item["env_id"])
                item["log_path"] = str(spec.log_path)
                item["stdout_path"] = str(spec.stdout_path)
            batch_trials[spec.batch_idx] = batch_stats
            successes = sum(1 for item in batch_stats if item["success"])
            batch_logs[spec.batch_idx] = {
                "batch": spec.batch_idx,
                "envs": spec.batch_envs,
                "log_path": str(spec.log_path),
                "stdout_path": str(spec.stdout_path),
                "elapsed_s": result.elapsed_s,
                "successes": successes,
                "command": result.cmd,
            }
            safe_print(
                f"[batch {spec.batch_idx:04d}] success={successes}/{spec.batch_envs} "
                f"elapsed={result.elapsed_s:.1f}s "
                f"max_open_deg={[None if x['max_open_deg'] is None else round(float(x['max_open_deg']), 1) for x in batch_stats]}"
            )

    all_trials: list[dict[str, Any]] = []
    for idx in sorted(batch_trials):
        for item in batch_trials[idx]:
            item["global_trial"] = len(all_trials)
            all_trials.append(item)

    total_successes = sum(1 for item in all_trials if item["success"])
    total = len(all_trials)
    success_rate = total_successes / max(1, total)
    end_triggered_trials = [item for item in all_trials if item.get("end_triggered")]
    end_false_trigger_count = sum(
        1
        for item in end_triggered_trials
        if item.get("first_success_step") is None
        or int(item["first_end_trigger_step"]) < int(item["first_success_step"])
    )
    end_missing_trigger_count = sum(
        1 for item in all_trials if item.get("success") and not item.get("end_triggered")
    )
    end_trigger_steps = [int(item["first_end_trigger_step"]) for item in end_triggered_trials]
    end_trigger_delays = [
        int(item["end_trigger_delay_after_first_task_success"])
        for item in end_triggered_trials
        if item.get("end_trigger_delay_after_first_task_success") is not None
    ]
    interaction_trials = [item for item in all_trials if int(item.get("interaction_records", 0)) > 0]
    failure_rollouts = export_failed_rollouts(
        args=args,
        run_root=run_root,
        all_trials=all_trials,
        batch_logs=batch_logs,
        metric=metric,
    )
    summary = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "door_cfg": str(resolve_path(args.door_cfg)),
        "mode": args.mode,
        "robot_body": args.robot_body,
        "num_envs": int(args.num_envs),
        "total_trials": int(args.total_trials),
        "parallel_batches": int(args.parallel_batches),
        "steps": int(args.steps),
        "pass_open_angle_deg": float(args.pass_open_angle_deg),
        "success_metric": metric,
        "door_motion_sign": float(args.door_motion_sign),
        "base_seed": args.base_seed,
        "headless": bool(args.headless),
        "graphics_device_id": args.graphics_device_id,
        "dp_inference_steps": int(args.dp_inference_steps),
        "dp_log_interval": int(args.dp_log_interval),
        "dp_noise_scheduler_type": args.dp_noise_scheduler_type,
        "dp_action_horizon": None if args.dp_action_horizon is None else int(args.dp_action_horizon),
        "dp_temporal_ensemble": bool(args.dp_temporal_ensemble),
        "dp_temporal_prefetch_actions": int(args.dp_temporal_prefetch_actions),
        "dp_temporal_old_weight": float(args.dp_temporal_old_weight),
        "dp_temporal_new_weight": float(args.dp_temporal_new_weight),
        "dp_end_signal_monitor": bool(args.dp_end_signal_monitor),
        "dp_end_signal_threshold": float(args.dp_end_signal_threshold),
        "dp_end_signal_consecutive_steps": int(args.dp_end_signal_consecutive_steps),
        "dp_fps": int(args.dp_fps),
        "rgb": bool(args.rgb),
        "depth_only": bool(args.depth_only),
        "depth_noise_enabled": bool(args.enable_depth_noise),
        "depth_noise_prob": float(args.depth_noise_prob),
        "depth_gaussian_std_m": float(args.depth_gaussian_std_m),
        "depth_edge_noise_prob": float(args.depth_edge_noise_prob),
        "depth_hole_noise_prob": float(args.depth_hole_noise_prob),
        "depth_dropout_prob": float(args.depth_dropout_prob),
        "depth_camera_randomization": bool(args.enable_depth_camera_randomization),
        "depth_camera_pos_rand_m": float(args.depth_camera_pos_rand_m),
        "depth_camera_rot_rand_deg": float(args.depth_camera_rot_rand_deg),
        "successes": total_successes,
        "trials": total,
        "success_rate": success_rate,
        "end_trigger_rate": len(end_triggered_trials) / max(1, total),
        "end_false_trigger_count": int(end_false_trigger_count),
        "end_missing_trigger_count": int(end_missing_trigger_count),
        "end_trigger_steps": end_trigger_steps,
        "end_trigger_delay_after_first_task_success": end_trigger_delays,
        "interaction_state_summary": {
            "trials_with_predictions": len(interaction_trials),
            "contact_mean": (
                None
                if not interaction_trials
                else float(np.mean([item["interaction_contact_mean"] for item in interaction_trials]))
            ),
            "handle_progress_mean": (
                None
                if not interaction_trials
                else float(np.mean([item["interaction_handle_mean"] for item in interaction_trials]))
            ),
            "door_progress_mean": (
                None
                if not interaction_trials
                else float(np.mean([item["interaction_door_mean"] for item in interaction_trials]))
            ),
        },
        "failure_rollout_root": (
            str(Path(args.failure_rollout_root).expanduser())
            if args.failure_rollout_root
            else (str(run_root / "failure_rollouts") if args.save_failure_rollouts else None)
        ),
        "failure_rollouts": failure_rollouts,
        "batch_logs": [batch_logs[idx] for idx in sorted(batch_logs)],
        "trials_detail": all_trials,
    }
    summary_path = run_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    safe_print(f"SUCCESS_RATE {total_successes}/{total} = {success_rate:.4f}")
    safe_print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
