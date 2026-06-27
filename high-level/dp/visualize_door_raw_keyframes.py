#!/usr/bin/env python
"""Visualize Door DP raw keyframes as a timeline and camera contact sheet."""

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .door_dp_common import (
        DEFAULT_A2W_PHASE_NAMES,
        extract_motion_keyframes_from_raw_arrays,
        sliding_mean_displacement,
    )
except ImportError:
    from door_dp_common import (
        DEFAULT_A2W_PHASE_NAMES,
        extract_motion_keyframes_from_raw_arrays,
        sliding_mean_displacement,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", required=True, help="Directory containing episode_*.npz raw files.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Defaults to <raw_root>/keyframe_viz/episode_xxxxxx.",
    )
    parser.add_argument("--cols", type=int, default=3, help="Number of keyframe panels per contact-sheet row.")
    parser.add_argument("--thumb_width", type=int, default=240, help="Width of each camera thumbnail.")
    parser.add_argument("--max_label_chars", type=int, default=64, help="Truncate long rule labels in contact sheet.")
    return parser.parse_args()


def episode_path(raw_root, episode):
    path = Path(raw_root).expanduser() / f"episode_{int(episode):06d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def phase_names_for_data(data):
    if "phase_names" in data.files:
        return [str(x) for x in data["phase_names"].tolist()]
    return list(DEFAULT_A2W_PHASE_NAMES)


def phase_name_at(data, idx, phase_names):
    if "subtask_index" not in data.files:
        return "unknown"
    phase_ids = data["subtask_index"].reshape(-1)
    if idx < 0 or idx >= len(phase_ids):
        return "unknown"
    pid = int(phase_ids[idx])
    return phase_names[pid] if 0 <= pid < len(phase_names) else f"phase_{pid}"


def compute_metrics(data):
    n = int(data["state"].shape[0] if "state" in data.files else data["action"].shape[0])
    metrics = {}

    if "replay_root_state" in data.files:
        root = np.asarray(data["replay_root_state"], dtype=np.float64)
        if root.ndim == 2 and root.shape[1] >= 13:
            metrics["base speed change"] = sliding_mean_displacement(
                np.stack([root[:, 7], root[:, 8], root[:, 12]], axis=1),
                window=5,
            )

    if "replay_dof_pos" in data.files:
        dof_pos = np.asarray(data["replay_dof_pos"], dtype=np.float64)
        if dof_pos.ndim == 2 and dof_pos.shape[1] >= 7:
            metrics["arm joint motion"] = sliding_mean_displacement(dof_pos[:, -7:-1], window=5)
            metrics["gripper motion"] = sliding_mean_displacement(dof_pos[:, -1], window=3)

    if "replay_door_dof_vel" in data.files:
        door_vel = np.asarray(data["replay_door_dof_vel"], dtype=np.float64)
        if door_vel.ndim == 2 and door_vel.shape[1] >= 2:
            metrics["handle speed change"] = sliding_mean_displacement(door_vel[:, 1], window=5)
            metrics["door hinge speed change"] = sliding_mean_displacement(door_vel[:, 0], window=5)

    if "action" in data.files:
        action = np.asarray(data["action"], dtype=np.float64)
        if action.ndim == 2 and action.shape[1] >= 1:
            metrics["gripper target motion"] = sliding_mean_displacement(action[:, -1], window=3)

    # Keep a stable order and fill missing metrics with zeros so the plot layout is predictable.
    order = [
        "base speed change",
        "arm joint motion",
        "handle speed change",
        "door hinge speed change",
        "gripper target motion",
    ]
    return {name: metrics.get(name, np.zeros(n, dtype=np.float64)) for name in order}


def write_keyframe_table(path, indices, names, rules, data, phase_names):
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("idx,phase,name,rule\n")
        for idx, name, rule in zip(indices, names, rules):
            f.write(f"{idx},{phase_name_at(data, idx, phase_names)},{name},{rule}\n")


def plot_timeline(path, data, indices, names, phase_names):
    metrics = compute_metrics(data)
    n = len(next(iter(metrics.values()))) if metrics else int(data["state"].shape[0])
    x = np.arange(n)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(18, 11), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    phase_ids = data["subtask_index"].reshape(-1) if "subtask_index" in data.files else None
    phase_boundaries = []
    if phase_ids is not None and len(phase_ids) > 0:
        changes = np.nonzero(np.diff(phase_ids) != 0)[0] + 1
        phase_boundaries = [0, *changes.tolist(), len(phase_ids)]

    for ax, (label, metric) in zip(axes, metrics.items()):
        ax.plot(x, metric, linewidth=1.4)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        for idx in indices:
            ax.axvline(idx, color="crimson", alpha=0.35, linewidth=1.0)
        for boundary in phase_boundaries[1:-1]:
            ax.axvline(boundary, color="gray", alpha=0.25, linestyle="--", linewidth=0.8)

    top_ax = axes[0]
    ymax = top_ax.get_ylim()[1]
    for k, (idx, name) in enumerate(zip(indices, names), start=1):
        top_ax.text(
            idx,
            ymax,
            f"{k}:{idx}",
            rotation=90,
            fontsize=7,
            va="top",
            ha="center",
            color="crimson",
        )

    if phase_ids is not None and len(phase_boundaries) >= 2:
        bottom_ax = axes[-1]
        ymin, ymax = bottom_ax.get_ylim()
        for start, end in zip(phase_boundaries[:-1], phase_boundaries[1:]):
            mid = (start + end) * 0.5
            pid = int(phase_ids[start])
            pname = phase_names[pid] if 0 <= pid < len(phase_names) else str(pid)
            bottom_ax.text(mid, ymin, pname, fontsize=8, ha="center", va="bottom", alpha=0.8)

    axes[-1].set_xlabel("frame index")
    fig.suptitle("Door DP keyframes: dynamic metrics + extracted keyframe anchors", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_rgb_image(array, size):
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        image = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    else:
        arr = np.squeeze(arr)
        image = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    return image.resize(size, Image.Resampling.BILINEAR)


def wrap_short(text, width=36, max_chars=64):
    text = str(text)
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 3)] + "..."
    return textwrap.wrap(text, width=width) or [""]


def draw_panel(data, idx, number, name, rule, phase_name, thumb_width, max_label_chars):
    wrist = data["wrist_masked_depth"][idx] if "wrist_masked_depth" in data.files else None
    front = data["front_masked_depth"][idx] if "front_masked_depth" in data.files else None
    if wrist is None and "wrist_rgb" in data.files:
        wrist = data["wrist_rgb"][idx]
    if front is None and "front_rgb" in data.files:
        front = data["front_rgb"][idx]
    if wrist is None:
        wrist = np.zeros((480, 640), dtype=np.uint8)
    if front is None:
        front = np.zeros_like(wrist)

    thumb_height = int(round(thumb_width * 0.75))
    label_h = 88
    gap = 4
    panel_w = thumb_width * 2 + gap
    panel_h = thumb_height + label_h
    panel = Image.new("RGB", (panel_w, panel_h), (245, 245, 245))
    wrist_img = to_rgb_image(wrist, (thumb_width, thumb_height))
    front_img = to_rgb_image(front, (thumb_width, thumb_height))
    panel.paste(wrist_img, (0, label_h))
    panel.paste(front_img, (thumb_width + gap, label_h))

    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    draw.rectangle((0, 0, panel_w, label_h - 1), fill=(255, 255, 255), outline=(200, 200, 200))
    draw.text((6, 4), f"#{number}  frame={idx}  phase={phase_name}", fill=(0, 0, 0), font=font)
    draw.text((6, 22), str(name), fill=(140, 0, 0), font=small_font)
    y = 40
    for line in wrap_short(rule, width=60, max_chars=max_label_chars):
        draw.text((6, y), line, fill=(40, 40, 40), font=small_font)
        y += 14
    draw.text((6, label_h + 4), "wrist", fill=(255, 255, 255), font=small_font)
    draw.text((thumb_width + gap + 6, label_h + 4), "front", fill=(255, 255, 255), font=small_font)
    return panel


def make_contact_sheet(path, data, indices, names, rules, phase_names, cols=3, thumb_width=240, max_label_chars=64):
    panels = []
    for number, (idx, name, rule) in enumerate(zip(indices, names, rules), start=1):
        panels.append(
            draw_panel(
                data,
                idx,
                number,
                name,
                rule,
                phase_name_at(data, idx, phase_names),
                thumb_width,
                max_label_chars,
            )
        )
    if not panels:
        raise ValueError("No keyframes to visualize.")
    cols = max(1, int(cols))
    rows = int(math.ceil(len(panels) / cols))
    panel_w, panel_h = panels[0].size
    sheet = Image.new("RGB", (cols * panel_w, rows * panel_h), (230, 230, 230))
    for i, panel in enumerate(panels):
        row, col = divmod(i, cols)
        sheet.paste(panel, (col * panel_w, row * panel_h))
    sheet.save(path)


def main():
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser()
    path = episode_path(raw_root, args.episode)
    data = np.load(path, allow_pickle=True)
    phase_names = phase_names_for_data(data)
    indices, names, rules = extract_motion_keyframes_from_raw_arrays(data, phase_names=phase_names)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = raw_root / "keyframe_viz" / f"episode_{int(args.episode):06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = output_dir / "keyframe_timeline.png"
    sheet_path = output_dir / "keyframe_contact_sheet.png"
    csv_path = output_dir / "keyframes.csv"

    plot_timeline(timeline_path, data, indices, names, phase_names)
    make_contact_sheet(
        sheet_path,
        data,
        indices,
        names,
        rules,
        phase_names,
        cols=args.cols,
        thumb_width=args.thumb_width,
        max_label_chars=args.max_label_chars,
    )
    write_keyframe_table(csv_path, indices, names, rules, data, phase_names)

    print(f"episode: {path}")
    print(f"keyframes: {len(indices)}")
    print("indices:", indices)
    print(f"timeline: {timeline_path}")
    print(f"contact_sheet: {sheet_path}")
    print(f"table: {csv_path}")


if __name__ == "__main__":
    main()
