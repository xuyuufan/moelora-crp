"""Plot the H=20 MoE-LoRA training reward against epoch for three seeds."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SEEDS = (20260913, 20260923, 20260933)
RUN_DIR_TEMPLATE = "moe_lora_long_20260917_seed{seed}"

# Okabe-Ito colorblind-safe palette.
SEED_COLORS = ("#56B4E9", "#009E73", "#CC79A7")
MEAN_COLOR = "#D55E00"
GREY = "#777777"


def load_training_reward(seed: int) -> tuple[np.ndarray, np.ndarray]:
    metrics_path = PROJECT_ROOT / "output" / RUN_DIR_TEMPLATE.format(seed=seed) / "metrics.jsonl"
    epochs: list[int] = []
    rewards: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") != "epoch" or int(record["epoch"]) == 0:
                continue
            epochs.append(int(record["epoch"]))
            # The logged training cost is sampled relocations per rollout.
            # Reward is its negative, so higher (less negative) is better.
            rewards.append(-float(record["epoch_mean_sampled_relocations"]))

    if epochs != list(range(1, 6)):
        raise ValueError(f"Seed {seed}: expected training epochs 1..5, found {epochs}")
    return np.asarray(epochs), np.asarray(rewards)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    series = [load_training_reward(seed) for seed in SEEDS]
    epochs = series[0][0]
    reward_matrix = np.vstack([reward for _, reward in series])
    reward_mean = reward_matrix.mean(axis=0)
    reward_sd = reward_matrix.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(6.8, 4.0), layout="constrained")
    for idx, (seed, (_, reward)) in enumerate(zip(SEEDS, series)):
        ax.plot(
            epochs,
            reward,
            color=SEED_COLORS[idx],
            linewidth=1.2,
            marker="o",
            markersize=3.5,
            alpha=0.8,
            label=f"Seed {seed}",
        )

    ax.fill_between(
        epochs,
        reward_mean - reward_sd,
        reward_mean + reward_sd,
        color=MEAN_COLOR,
        alpha=0.16,
        linewidth=0,
        label=r"Mean $\pm$ 1 SD",
    )
    ax.plot(
        epochs,
        reward_mean,
        color=MEAN_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=4.5,
        label="Three-seed mean",
        zorder=5,
    )

    ax.axvspan(3, 5, color=GREY, alpha=0.07, zorder=0)
    ax.annotate(
        "Stable region",
        xy=(4.0, reward_mean[3]),
        xytext=(3.55, -118.25),
        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8},
        color=GREY,
        fontsize=8,
    )
    ax.set_title("MoE-LoRA training reward (H=20)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean sampled training reward (− relocations)")
    ax.set_xticks(epochs)
    ax.set_xlim(0.85, 5.15)
    ax.set_ylim(-120.0, -116.9)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.85)
    ax.legend(loc="lower right", ncol=2, frameon=False, handlelength=2.6, columnspacing=1.0)
    ax.text(
        0.02,
        0.965,
        "Training split; 1,024 instances per epoch; 8 rollouts per instance",
        transform=ax.transAxes,
        va="top",
        fontsize=7.8,
        color="#444444",
    )

    pdf_path = HERE / "fig_moe_lora_h20_training_reward_curve.pdf"
    png_path = HERE / "fig_moe_lora_h20_training_reward_curve.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"pdf={pdf_path}")
    print(f"png={png_path}")
    print("epoch_mean_rewards=" + ",".join(f"{value:.9f}" for value in reward_mean))
    print("epoch_sample_sd=" + ",".join(f"{value:.9f}" for value in reward_sd))
    print(f"late_epoch_reward_range={np.ptp(reward_mean[2:]):.9f}")


if __name__ == "__main__":
    main()
