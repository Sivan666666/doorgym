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


SCRIPT_DIR = Path(__file__).resolve().parent
DP_ROOT = SCRIPT_DIR.parent
HIGH_LEVEL_ROOT = DP_ROOT.parent
REPO_ROOT = HIGH_LEVEL_ROOT.parent
PLAY_SCRIPT = DP_ROOT / "play" / "play_door_policy.py"
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
    parser.add_argument("--mode", choices=["ikpush", "push", "pull"], default="ikpush")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of envs per play run.")
    parser.add_argument("--total_trials", type=int, default=64, help="Total policy-controlled attempts to run.")
    parser.add_argument(
        "--parallel_batches",
        type=int,
        default=1,
        help="Number of play batches to run concurrently. Each batch is a separate Isaac Gym subprocess.",
    )
    parser.add_argument("--steps", type=int, default=2500)
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
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--camera_display_scale", type=int, default=5)
    parser.add_argument("--run_root", type=str, default=None, help="Directory for logs and summary JSON.")
    parser.add_argument("--stream_output", action="store_true", help="Stream each play subprocess output to this terminal.")
    parser.add_argument("--progress_interval", type=float, default=5.0, help="Seconds between per-batch progress updates.")
    parser.add_argument("--no_progress", action="store_true", help="Disable per-batch progress updates.")
    parser.add_argument("--print_policy_steps", action="store_true", help="Do not pass --no_dp_print to play.")
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
    return "abs" if mode == "pull" else "signed"


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
        "--no_show_seg",
    ]
    if args.dp_action_horizon is not None:
        cmd += ["--dp_action_horizon", str(args.dp_action_horizon)]
    if args.rgb:
        cmd.append("--rgb")
    if args.headless:
        cmd.append("--headless")
    if args.graphics_device_id is not None:
        cmd += ["--graphics_device_id", str(args.graphics_device_id)]
    if not args.print_policy_steps:
        cmd.append("--no_dp_print")
    cmd += ["--camera_display_scale", str(args.camera_display_scale)]
    cmd.append("--")
    cmd += [
        "--door_cfg",
        str(door_cfg),
        "--pass_open_angle_deg",
        str(args.pass_open_angle_deg),
    ]
    if args.base_seed is not None:
        cmd += ["--seed", str(int(args.base_seed) + int(batch_idx))]
    extra = args.play_args[1:] if args.play_args[:1] == ["--"] else args.play_args
    cmd += extra
    return cmd


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
        f"mode={args.mode} num_envs={args.num_envs} total_trials={args.total_trials} "
        f"steps={args.steps} threshold={args.pass_open_angle_deg}deg metric={metric} "
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
    summary = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "door_cfg": str(resolve_path(args.door_cfg)),
        "mode": args.mode,
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
        "dp_noise_scheduler_type": args.dp_noise_scheduler_type,
        "successes": total_successes,
        "trials": total,
        "success_rate": success_rate,
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
