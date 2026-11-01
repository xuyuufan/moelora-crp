#!/usr/bin/env python3
"""Plot the final MoE-LoRA sealed-test comparison from the saved summary."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MOE_METHODS = (
    "MoE_seed20260913",
    "MoE_seed20260923",
    "MoE_seed20260933",
)

OUR_COLOR = "#D55E00"       # Okabe-Ito vermillion
BASE_COLOR = "#7A7A7A"
LEVELING_COLOR = "#0072B2"  # Okabe-Ito blue
GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#262626"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "output/moe_lora_test_sealed_20260917/summary.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).resolve().parent / "fig_moe_sealed_test",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("split") != "test_sealed":
        raise ValueError("expected a test_sealed evaluation summary")

    methods = summary["methods"]
    comparisons = summary["comparisons_vs_base"]
    for name in (*MOE_METHODS, "MijnRL_base", "paper_leveling_L"):
        if name not in methods:
            raise KeyError(f"missing method in summary: {name}")

    moe_scores = np.asarray(
        [methods[name]["mean_relocations_per_container"] for name in MOE_METHODS],
        dtype=float,
    )
    moe_mean = float(np.mean(moe_scores))
    moe_std = float(np.std(moe_scores, ddof=1))
    base_score = float(methods["MijnRL_base"]["mean_relocations_per_container"])
    leveling_score = float(
        methods["paper_leveling_L"]["mean_relocations_per_container"]
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#555555",
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
        }
    )

    fig, (ax_perf, ax_delta) = plt.subplots(
        1,
        2,
        figsize=(6.75, 2.85),
        gridspec_kw={"width_ratios": (1.08, 1.0), "wspace": 0.42},
    )

    # Panel (a): primary objective. A dot plot avoids misleading truncated bars.
    labels = ("Leveling", "Frozen base", "MoE-LoRA")
    y = np.asarray((2.0, 1.0, 0.0))
    values = np.asarray((leveling_score, base_score, moe_mean))
    colors = (LEVELING_COLOR, BASE_COLOR, OUR_COLOR)
    x_min = min(values) - 0.025
    for yi, value, color in zip(y, values, colors, strict=True):
        ax_perf.hlines(yi, x_min, value, color=color, alpha=0.22, linewidth=2.0)
        ax_perf.scatter(
            value,
            yi,
            s=56,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )

    # Show the three independently trained models as small offset points.
    seed_offsets = np.asarray((-0.13, 0.0, 0.13))
    ax_perf.scatter(
        moe_scores,
        np.full(3, y[-1]) + seed_offsets,
        marker="D",
        s=17,
        facecolor="white",
        edgecolor=OUR_COLOR,
        linewidth=1.0,
        zorder=5,
        label="Individual seeds",
    )
    ax_perf.errorbar(
        moe_mean,
        y[-1],
        xerr=moe_std,
        fmt="none",
        ecolor=OUR_COLOR,
        elinewidth=1.5,
        capsize=3,
        zorder=3,
    )
    for yi, value in zip(y, values, strict=True):
        ax_perf.annotate(
            f"{value:.4f}",
            (value, yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            fontweight="bold" if yi == y[-1] else "normal",
        )
    ax_perf.set_yticks(y, labels)
    ax_perf.set_xlim(x_min, leveling_score + 0.065)
    ax_perf.set_xlabel("Mean relocations per container, $R/N$  (lower is better)")
    ax_perf.set_title("(a) Sealed-test performance", loc="left", pad=7)
    ax_perf.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax_perf.set_axisbelow(True)
    ax_perf.legend(loc="lower right", fontsize=7.5, handletextpad=0.4)

    # Panel (b): paired bootstrap confidence intervals against the frozen base.
    seed_labels = tuple(name.removeprefix("MoE_seed") for name in MOE_METHODS)
    delta_y = np.arange(len(MOE_METHODS))[::-1]
    for yi, name in zip(delta_y, MOE_METHODS, strict=True):
        comparison = comparisons[name]
        mean_delta = float(comparison["candidate_minus_baseline_mean_relocations"])
        ci = comparison["paired_bootstrap_mean_difference_ci"]
        lower, upper = float(ci["lower"]), float(ci["upper"])
        ax_delta.errorbar(
            mean_delta,
            yi,
            xerr=np.asarray([[mean_delta - lower], [upper - mean_delta]]),
            fmt="o",
            color=OUR_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.5,
            elinewidth=1.8,
            capsize=3,
            zorder=4,
        )
        wtl = comparison["win_tie_loss_vs_baseline"]
        ax_delta.text(
            -0.10,
            yi,
            f"W/T/L {wtl['wins']}/{wtl['ties']}/{wtl['losses']}",
            ha="right",
            va="center",
            fontsize=7.2,
            color="#555555",
        )

    ax_delta.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--", zorder=2)
    ax_delta.axvspan(-4.2, 0.0, color=OUR_COLOR, alpha=0.035, zorder=0)
    ax_delta.set_yticks(delta_y, [f"Seed {seed}" for seed in seed_labels])
    ax_delta.set_xlim(-4.2, 0.25)
    ax_delta.set_ylim(-0.65, 2.65)
    ax_delta.set_xlabel(
        r"Paired $\Delta$ relocations vs. frozen base  (lower is better)"
    )
    ax_delta.set_title("(b) Bootstrap 95% confidence intervals", loc="left", pad=7)
    ax_delta.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax_delta.set_axisbelow(True)

    fig.suptitle(
        "MoE-LoRA on the fixed large-scale synthetic target domain",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.035,
        r"500 held-out instances; H=20. MoE-LoRA: 2 experts, rank 8, $\alpha$=16.",
        ha="center",
        va="top",
        fontsize=7.8,
        color="#555555",
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_prefix.with_suffix(".pdf")
    png_path = args.output_prefix.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(
        json.dumps(
            {
                "source": str(args.summary),
                "pdf": str(pdf_path),
                "png": str(png_path),
                "moe_mean_r_per_n": moe_mean,
                "moe_sample_std_r_per_n": moe_std,
                "test_instances": 500,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
