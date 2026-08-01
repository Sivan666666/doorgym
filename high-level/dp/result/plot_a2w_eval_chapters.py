#!/usr/bin/env python3
"""Generate the Chapter 3 and Chapter 7 evaluation figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FONT_FAMILY = "AR PL UMing CN"
BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#B279A2"
GRAY = "#79706E"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )


def annotate_bar(ax: plt.Axes, bar, value: float, *, fontsize: int = 8) -> None:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.5,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        rotation=90 if value >= 10 else 0,
    )


def plot_chapter3(output_path: Path) -> None:
    strategies = ["Baseline ACT", "Keyframe ACT", "Plücker ACT", "KF + Plücker", "π0.5"]
    wc4 = np.asarray([90.625, 67.1875, 92.1875, 76.5625, 48.4375])
    button = np.asarray([4.6875, 9.375, 20.3125, 29.6875, 48.4375])
    fire = np.asarray([87.5, np.nan, 95.3125, 95.3125, 90.625])

    fig, (ax_main, ax_ft) = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.55, 1]})

    x = np.arange(len(strategies))
    width = 0.24
    for offset, values, label, color in (
        (-width, wc4, "WC4（seen）", BLUE),
        (0.0, button, "Button Door（unseen）", ORANGE),
        (width, fire, "Fire Door（unseen）", GREEN),
    ):
        valid = ~np.isnan(values)
        bars = ax_main.bar(x[valid] + offset, values[valid], width, label=label, color=color, edgecolor="white")
        for bar, value in zip(bars, values[valid]):
            annotate_bar(ax_main, bar, float(value))

    missing_index = 1
    ax_main.text(
        x[missing_index] + width,
        3.0,
        "未测",
        ha="center",
        va="bottom",
        fontsize=8,
        color=GRAY,
        rotation=90,
    )
    ax_main.set_title("(a) 5-door 训练策略在 seen / unseen 门上的成功率")
    ax_main.set_ylabel("成功率（%）")
    ax_main.set_xticks(x, strategies, rotation=16, ha="right")
    ax_main.set_ylim(0, 108)
    ax_main.set_yticks(np.arange(0, 101, 20))
    ax_main.grid(axis="y", alpha=0.25)
    ax_main.legend(loc="upper center", ncols=3, frameon=False)

    finetune_steps = np.asarray([0, 5, 10, 20])
    baseline = np.asarray([87.5, 87.5, 81.25, 81.25])
    ax_ft.plot(finetune_steps, baseline, marker="o", linewidth=2.3, color=BLUE, label="Baseline ACT")
    ax_ft.plot([0, 20], [95.3125, 90.625], marker="s", linewidth=2.3, color=GREEN, label="Plücker ACT")
    ax_ft.plot([0, 20], [95.3125, 92.1875], marker="^", linewidth=2.3, color=PURPLE, label="KF + Plücker")
    for step, value in zip(finetune_steps, baseline):
        ax_ft.annotate(f"{value:.1f}%", (step, value), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8)
    for step, value in zip([0, 20], [95.3125, 90.625]):
        ax_ft.annotate(f"{value:.1f}%", (step, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    for step, value in zip([0, 20], [95.3125, 92.1875]):
        offset = 20 if step == 0 else 8
        ax_ft.annotate(f"{value:.1f}%", (step, value), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=8)
    ax_ft.set_title("(b) Fire Door 50 条数据 finetune")
    ax_ft.set_xlabel("Finetune steps（K）")
    ax_ft.set_ylabel("Fire Door 成功率（%）")
    ax_ft.set_xticks(finetune_steps)
    ax_ft.set_ylim(72, 101)
    ax_ft.grid(alpha=0.25)
    ax_ft.legend(frameon=False, loc="lower left")

    fig.suptitle("第3章主要结果：ACT 变体的 seen / unseen door 表现", fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        "每个点/柱为 64 次评测；统一 horizon=25、base seed=615455575。Keyframe ACT 的 Fire Door 未评测。",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_chapter7(output_path: Path) -> None:
    steps = np.asarray([0, 5, 20, 50])
    original_counts = np.asarray([59, 58, 57, 55])
    recovery_counts = np.asarray([59, 53, 49, 60])
    original = original_counts / 64.0 * 100.0
    recovery = recovery_counts / 64.0 * 100.0
    delta = recovery - original

    fig, (ax_curve, ax_delta) = plt.subplots(1, 2, figsize=(13.8, 5.8), gridspec_kw={"width_ratios": [1.55, 1]})

    ax_curve.plot(steps, original, marker="o", markersize=8, linewidth=2.5, color=BLUE, label="仅原始 250 条数据")
    ax_curve.plot(steps, recovery, marker="s", markersize=8, linewidth=2.5, color=RED, label="原始 80% + Recovery 20%")
    for x, value, count in zip(steps, original, original_counts):
        ax_curve.annotate(f"{count}/64\n{value:.1f}%", (x, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    for x, value, count in zip(steps, recovery, recovery_counts):
        y_offset = -29 if x != 50 else 8
        ax_curve.annotate(f"{count}/64\n{value:.1f}%", (x, value), xytext=(0, y_offset), textcoords="offset points", ha="center", fontsize=8)
    ax_curve.set_title("(a) WC4 成功率随 finetune 步数变化")
    ax_curve.set_xlabel("Finetune steps（K）")
    ax_curve.set_ylabel("成功率（%）")
    ax_curve.set_xticks(steps)
    ax_curve.set_ylim(70, 99)
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(frameon=False, loc="lower right")

    colors = [GRAY if abs(value) < 1e-9 else (GREEN if value > 0 else RED) for value in delta]
    bars = ax_delta.bar(steps, delta, width=[3.2, 3.2, 5.0, 5.0], color=colors, edgecolor="white")
    ax_delta.axhline(0, color="#333333", linewidth=1)
    for bar, value in zip(bars, delta):
        va = "bottom" if value >= 0 else "top"
        offset = 0.7 if value >= 0 else -0.7
        ax_delta.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.2f} pp", ha="center", va=va, fontsize=9)
    ax_delta.set_title("(b) Recovery 20% 相对原始数据对照")
    ax_delta.set_xlabel("Finetune steps（K）")
    ax_delta.set_ylabel("成功率差值（百分点）")
    ax_delta.set_xticks(steps)
    ax_delta.set_ylim(-16, 12)
    ax_delta.grid(axis="y", alpha=0.25)

    fig.suptitle("第7章主要结果：Recovery 20% 的先下降、后适应过程", fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        "相同 Plücker 50K 初始化；每个 checkpoint 在相同 64 个 WC4 随机环境上评测，horizon=25。",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    plot_chapter3(args.output_dir / "a2w_chapter3_seen_unseen_and_firedoor_finetune.png")
    plot_chapter7(args.output_dir / "a2w_chapter7_recovery_finetune.png")
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
