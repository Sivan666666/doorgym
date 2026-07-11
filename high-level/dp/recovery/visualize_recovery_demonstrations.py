#!/usr/bin/env python3
"""Create contact sheets and metric plots for verified recovery raw episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_IDS = "0,4,5,10,15,24,44,64,84,104"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize verified Door ACT recovery demonstrations.")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--candidate_ids", default=DEFAULT_IDS)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def scalar(data, key, default=None):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value


def first_phase_index(subtask: np.ndarray, phase_names: list[str], names: tuple[str, ...], fallback: int) -> int:
    wanted = {phase_names.index(name) for name in names if name in phase_names}
    if wanted:
        indices = np.flatnonzero(np.isin(subtask, list(wanted)))
        if indices.size:
            return int(indices[0])
    return int(fallback)


def episode_summary(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        frame_count = int(data["state"].shape[0])
        phases = [str(x) for x in np.asarray(data["phase_names"]).tolist()]
        subtask = np.asarray(data["subtask_index"], dtype=np.int64).reshape(-1)
        stage_indices = {
            "start": 0,
            "grasp": first_phase_index(subtask, phases, ("grasp", "close_gripper"), frame_count // 4),
            "rotate": first_phase_index(subtask, phases, ("rotate_handle",), frame_count // 2),
            "push": first_phase_index(subtask, phases, ("push_door",), 3 * frame_count // 4),
            "final": frame_count - 1,
        }
        source = str(scalar(data, "recovery_source_failure_rollout", ""))
        return {
            "path": str(path.resolve()),
            "episode": path.name,
            "candidate_id": int(scalar(data, "recovery_candidate_id", -1)),
            "failure_source": os.path.basename(os.path.dirname(source)),
            "failure_type": str(scalar(data, "recovery_failure_type", "unknown")),
            "t_dev": int(scalar(data, "recovery_t_dev", -1)),
            "t_fail": int(scalar(data, "recovery_t_fail", -1)),
            "t_branch": int(scalar(data, "recovery_t_branch", -1)),
            "parameters": json.loads(str(scalar(data, "recovery_parameters_json", "{}"))),
            "frames": frame_count,
            "stage_indices": stage_indices,
        }


def discover(raw_root: Path, candidate_ids: list[int]) -> list[dict]:
    wanted = set(candidate_ids)
    found: dict[int, dict] = {}
    for path in sorted(raw_root.glob("episode_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            candidate_id = int(scalar(data, "recovery_candidate_id", -1))
        if candidate_id in wanted:
            found[candidate_id] = episode_summary(path)
    missing = [cid for cid in candidate_ids if cid not in found]
    if missing:
        raise FileNotFoundError(f"Missing recovery candidate ids under {raw_root}: {missing}")
    return [found[cid] for cid in candidate_ids]


def row_label(item: dict) -> str:
    short_type = "geom" if item["failure_type"] == "geometric_misalignment" else "interaction"
    return (
        f"C{item['candidate_id']} | {short_type}\n"
        f"branch={item['t_branch']} | {item['failure_source'].split('_batch')[0]}"
    )


def plot_depth_contact_sheet(items: list[dict], image_key: str, out_path: Path, view_name: str) -> None:
    stages = ("start", "grasp", "rotate", "push", "final")
    fig, axes = plt.subplots(len(items), len(stages), figsize=(15, 2.35 * len(items)), squeeze=False)
    for row, item in enumerate(items):
        with np.load(item["path"], allow_pickle=True) as data:
            images = np.asarray(data[image_key])
            for col, stage in enumerate(stages):
                idx = min(int(item["stage_indices"][stage]), images.shape[0] - 1)
                ax = axes[row, col]
                ax.imshow(images[idx], cmap="turbo", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(stage)
                if col == 0:
                    ax.set_ylabel(row_label(item), rotation=0, ha="right", va="center", fontsize=8)
                ax.text(
                    0.02,
                    0.96,
                    f"f={idx}",
                    transform=ax.transAxes,
                    va="top",
                    color="white",
                    fontsize=7,
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
                )
    fig.suptitle(f"Verified recovery demonstrations — {view_name} depth", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(items: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(16, 18), squeeze=False)
    for ax, item in zip(axes.reshape(-1), items):
        with np.load(item["path"], allow_pickle=True) as data:
            door = np.asarray(data["replay_door_dof_pos"], dtype=np.float32)
            door_deg = np.abs(np.rad2deg(door[:, 0]))
            handle_deg = np.abs(np.rad2deg(door[:, 1])) if door.shape[1] > 1 else np.zeros_like(door_deg)
            contact = np.asarray(data["gripper_handle_contact_both"], dtype=np.float32).reshape(-1)
            subtask = np.asarray(data["subtask_index"], dtype=np.int64).reshape(-1)
            phase_names = [str(x) for x in np.asarray(data["phase_names"]).tolist()]
        x = np.arange(len(door_deg))
        ax.plot(x, door_deg, label="door hinge", linewidth=1.8)
        ax.plot(x, handle_deg, label="handle", linewidth=1.2)
        ax.fill_between(x, 0, contact * 12.0, color="tab:green", alpha=0.25, label="both contact")
        ax.axhline(80.0, color="tab:red", linestyle="--", linewidth=0.8, label="80 deg success")
        for phase in ("grasp", "close_gripper", "rotate_handle", "push_door"):
            if phase not in phase_names:
                continue
            indices = np.flatnonzero(subtask == phase_names.index(phase))
            if indices.size:
                ax.axvline(int(indices[0]), color="0.55", linestyle=":", linewidth=0.7)
        p = item["parameters"]
        ax.set_title(
            f"C{item['candidate_id']} {item['failure_type']} | branch={item['t_branch']} | "
            f"lat={p.get('base_lateral_adjust_m', 0):+.2f}, z={p.get('reapproach_z_offset_m', 0):+.2f}",
            fontsize=9,
        )
        ax.set_xlabel("recorded frame @25Hz")
        ax.set_ylabel("degrees")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Verified recovery progress and contact", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_replay_script(items: list[dict], out_path: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {Path(__file__).resolve().parents[3]}",
        "",
    ]
    for item in items:
        lines.extend(
            [
                f"echo '===== Candidate {item['candidate_id']} | {item['failure_type']} | branch {item['t_branch']} ====='",
                "conda run --no-capture-output -n b1z1 python \\",
                "  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \\",
                f"  --raw_episode {item['path']} \\",
                "  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \\",
                "  --door_asset_name wc4 \\",
                "  --mode ikpush \\",
                "  --replay_mode state \\",
                "  --start_step 0 \\",
                f"  --steps {item['frames']} \\",
                "  --stride 1 \\",
                "  --rl_device cuda:0 \\",
                "  --sim_device cuda:0 \\",
                "  --graphics_device_id 0 \\",
                "  --real_time \\",
                "  --no_show_seg \\",
                "  --log_interval 25",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(0o755)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_ids = [int(token.strip()) for token in args.candidate_ids.split(",") if token.strip()]
    items = discover(raw_root, candidate_ids)
    plot_depth_contact_sheet(items, "front_masked_depth", out_dir / "recovery_front_depth_stages.png", "front")
    plot_depth_contact_sheet(items, "wrist_masked_depth", out_dir / "recovery_wrist_depth_stages.png", "wrist")
    plot_metrics(items, out_dir / "recovery_metrics.png")
    with (out_dir / "selected_recovery_demonstrations.json").open("w", encoding="utf-8") as f:
        json.dump({"raw_root": str(raw_root), "selected": items}, f, indent=2, sort_keys=True)
    write_replay_script(items, out_dir / "replay_selected_recovery.sh")
    print(f"Recovery visualization written to {out_dir} ({len(items)} demonstrations)", flush=True)


if __name__ == "__main__":
    main()
