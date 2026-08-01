#!/usr/bin/env python3
"""Audit scripted pull/push raw datasets without loading camera arrays.

The report focuses on phase balance, contact, articulation progress, release
timing, base motion, and action smoothness.  These are the fields needed to
separate data-volume effects from trajectory-quality effects.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_spec(text: str) -> tuple[str, Path, int | None]:
    parts = text.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected LABEL=RAW_ROOT[:LIMIT]")
    label, value = parts
    limit = None
    if ":" in value:
        value, limit_text = value.rsplit(":", 1)
        limit = int(limit_text)
    return label, Path(value), limit


def first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else None


def longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask.astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def finite_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "p10": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def audit(label: str, root: Path, limit: int | None) -> dict:
    paths = sorted(root.glob("episode_*.npz"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz files under {root}")

    phase_frames: Counter[str] = Counter()
    phase_episode_lengths: dict[str, list[int]] = defaultdict(list)
    metrics: dict[str, list[float]] = defaultdict(list)
    total_frames = 0
    metadata_values: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        with np.load(path, allow_pickle=True) as episode:
            state = episode["state"].astype(np.float64)
            action = episode["action"].astype(np.float64)
            phase_ids = episode["subtask_index"].reshape(-1).astype(np.int64)
            phase_names = [str(item) for item in episode["phase_names"].tolist()]
            door_q = episode["replay_door_dof_pos"].astype(np.float64)
            door_qd = episode["replay_door_dof_vel"].astype(np.float64)
            root_state = episode["replay_root_state"].astype(np.float64)
            contact_any = episode["gripper_handle_contact_any"].reshape(-1) > 0.5
            contact_both = episode["gripper_handle_contact_both"].reshape(-1) > 0.5

            frames = len(state)
            total_frames += frames
            names = np.asarray([
                phase_names[idx] if 0 <= idx < len(phase_names) else f"unknown_{idx}"
                for idx in phase_ids
            ])
            for phase in sorted(set(names.tolist())):
                count = int(np.sum(names == phase))
                phase_frames[phase] += count
                phase_episode_lengths[phase].append(count)

            door_closed = float(episode["door_closed_angle"])
            handle_closed = float(episode["handle_closed_angle"])
            door_delta = np.abs(door_q[:, 0] - door_closed)
            handle_delta = np.abs(door_q[:, 1] - handle_closed)
            door_deg = np.rad2deg(door_delta)
            handle_deg = np.rad2deg(handle_delta)

            release = np.flatnonzero(names == "release_handle")
            release_idx = int(release[0]) if release.size else None
            contact_loss_idx = None
            if release_idx is not None:
                lost_after_release = np.flatnonzero(~contact_both[release_idx:])
                if lost_after_release.size:
                    contact_loss_idx = release_idx + int(lost_after_release[0])
            pull_push_mask = np.isin(names, ["pull_door", "push_door"])
            interaction_mask = np.isin(
                names,
                ["grasp", "close_gripper", "rotate_handle", "pull_door", "push_door", "release_handle"],
            )

            action_delta = np.linalg.norm(np.diff(action, axis=0), axis=1)
            state_action_error = np.linalg.norm(action - state, axis=1)
            interaction_delta_mask = interaction_mask[1:]

            metrics["frames"].append(frames)
            metrics["max_door_deg"].append(float(door_deg.max()))
            metrics["final_door_deg"].append(float(door_deg[-1]))
            metrics["max_handle_deg"].append(float(handle_deg.max()))
            metrics["first_handle_40_frame"].append(float(first_true(handle_deg >= 40.0) or math.nan))
            metrics["first_door_60_frame"].append(float(first_true(door_deg >= 60.0) or math.nan))
            metrics["contact_any_fraction"].append(float(contact_any.mean()))
            metrics["contact_both_fraction"].append(float(contact_both.mean()))
            metrics["contact_both_longest_run"].append(float(longest_true_run(contact_both)))
            metrics["interaction_contact_both_fraction"].append(
                float(contact_both[interaction_mask].mean()) if interaction_mask.any() else math.nan
            )
            metrics["pull_push_contact_both_fraction"].append(
                float(contact_both[pull_push_mask].mean()) if pull_push_mask.any() else math.nan
            )
            metrics["action_delta_mean"].append(float(action_delta.mean()))
            metrics["action_delta_p99"].append(float(np.quantile(action_delta, 0.99)))
            metrics["interaction_action_delta_mean"].append(
                float(action_delta[interaction_delta_mask].mean()) if interaction_delta_mask.any() else math.nan
            )
            metrics["state_action_error_mean"].append(float(state_action_error.mean()))
            metrics["base_displacement_xy"].append(float(np.linalg.norm(root_state[-1, :2] - root_state[0, :2])))
            metrics["base_delta_x"].append(float(root_state[-1, 0] - root_state[0, 0]))
            metrics["base_delta_y"].append(float(root_state[-1, 1] - root_state[0, 1]))
            metrics["release_frame"].append(float(release_idx) if release_idx is not None else math.nan)
            metrics["release_door_deg"].append(float(door_deg[release_idx]) if release_idx is not None else math.nan)
            metrics["release_door_speed_deg_s"].append(
                float(abs(np.rad2deg(door_qd[release_idx, 0]))) if release_idx is not None else math.nan
            )
            metrics["post_release_extra_open_deg"].append(
                float(door_deg[release_idx:].max() - door_deg[release_idx]) if release_idx is not None else math.nan
            )
            metrics["contact_loss_frame"].append(
                float(contact_loss_idx) if contact_loss_idx is not None else math.nan
            )
            metrics["contact_loss_door_deg"].append(
                float(door_deg[contact_loss_idx]) if contact_loss_idx is not None else math.nan
            )
            metrics["contact_loss_door_speed_deg_s"].append(
                float(abs(np.rad2deg(door_qd[contact_loss_idx, 0])))
                if contact_loss_idx is not None else math.nan
            )
            metrics["post_contact_loss_extra_open_deg"].append(
                float(door_deg[contact_loss_idx:].max() - door_deg[contact_loss_idx])
                if contact_loss_idx is not None else math.nan
            )

            for key in [
                "door_dp_mode", "controller_mode", "state_format", "action_format", "vision_mode",
                "depth_noise_enabled", "door_open_resistance", "camera_intrinsics_mode",
            ]:
                if key in episode:
                    metadata_values[key].add(str(episode[key].item()))

    return {
        "label": label,
        "root": str(root.resolve()),
        "episodes": len(paths),
        "total_frames": total_frames,
        "phase_frames": dict(sorted(phase_frames.items())),
        "phase_fraction": {key: value / total_frames for key, value in sorted(phase_frames.items())},
        "phase_length_per_episode": {
            key: finite_summary(value) for key, value in sorted(phase_episode_lengths.items())
        },
        "metrics": {key: finite_summary(value) for key, value in sorted(metrics.items())},
        "metadata_values": {key: sorted(value) for key, value in sorted(metadata_values.items())},
    }


def markdown(reports: list[dict]) -> str:
    metric_names = [
        "frames", "max_door_deg", "max_handle_deg", "release_door_deg",
        "release_door_speed_deg_s", "post_release_extra_open_deg",
        "contact_loss_door_deg", "contact_loss_door_speed_deg_s",
        "post_contact_loss_extra_open_deg",
        "contact_both_fraction", "pull_push_contact_both_fraction",
        "action_delta_mean", "interaction_action_delta_mean", "base_displacement_xy",
    ]
    lines = ["# Pull/Push raw dataset audit", "", "## Summary", ""]
    lines.append("| Dataset | Episodes | Frames | Mean frames | Max door (deg) | Max handle (deg) | Release phase door (deg) | Contact-loss door (deg) | Contact-loss speed (deg/s) | Post-contact-loss open (deg) | Both-contact (%) | Pull/push both-contact (%) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for report in reports:
        m = report["metrics"]
        fmt = lambda name, scale=1.0: "—" if m[name]["mean"] is None else f"{m[name]['mean'] * scale:.2f}"
        lines.append(
            f"| {report['label']} | {report['episodes']} | {report['total_frames']} | "
            f"{fmt('frames')} | {fmt('max_door_deg')} | {fmt('max_handle_deg')} | "
            f"{fmt('release_door_deg')} | {fmt('contact_loss_door_deg')} | "
            f"{fmt('contact_loss_door_speed_deg_s')} | {fmt('post_contact_loss_extra_open_deg')} | "
            f"{fmt('contact_both_fraction', 100)} | "
            f"{fmt('pull_push_contact_both_fraction', 100)} |"
        )
    lines.extend(["", "## Phase fractions", ""])
    phases = sorted({phase for report in reports for phase in report["phase_fraction"]})
    lines.append("| Dataset | " + " | ".join(phases) + " |")
    lines.append("|---|" + "---:|" * len(phases))
    for report in reports:
        values = [f"{100 * report['phase_fraction'].get(phase, 0.0):.2f}%" for phase in phases]
        lines.append(f"| {report['label']} | " + " | ".join(values) + " |")
    lines.extend(["", "## Full metric means", ""])
    lines.append("| Metric | " + " | ".join(report["label"] for report in reports) + " |")
    lines.append("|---|" + "---:|" * len(reports))
    all_metrics = sorted({metric for report in reports for metric in report["metrics"]})
    for metric in all_metrics:
        values = []
        for report in reports:
            value = report["metrics"].get(metric, {}).get("mean")
            values.append("—" if value is None else f"{value:.6g}")
        lines.append(f"| {metric} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=parse_spec)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_markdown", type=Path, required=True)
    args = parser.parse_args()
    reports = [audit(label, root, limit) for label, root, limit in args.dataset]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(markdown(reports), encoding="utf-8")
    print(markdown(reports), end="")


if __name__ == "__main__":
    main()
