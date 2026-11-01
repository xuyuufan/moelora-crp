#!/usr/bin/env python3
"""Plot Leveling versus final MoE-LoRA across Caserta lookahead horizons."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scrp.online_crp import (
    LevelingBaseline,
    OnlineCRPEnvironment,
    load_episode_manifest,
)


HORIZONS = (0, 5, 10, 20)
MOE_METHODS = (
    "MoE_seed20260913",
    "MoE_seed20260923",
    "MoE_seed20260933",
)
MOE_COLOR = "#D55E00"       # Okabe-Ito vermillion
LEVELING_COLOR = "#0072B2"  # Okabe-Ito blue
TEXT_COLOR = "#262626"
GRID_COLOR = "#D9D9D9"


def _leveling_mean_r_per_n(manifest: Path) -> float:
    """Re-evaluate Leveling exactly; it is invariant to the horizon value."""

    episodes = load_episode_manifest(manifest)
    ratios: list[float] = []
    for episode in episodes:
        environment = OnlineCRPEnvironment(episode, lookahead_horizon=0)
        while not environment.state.terminated:
            destination = LevelingBaseline.select_destination(
                environment.state, environment.legal_destinations()
            )
            environment.step(destination)
        ratios.append(environment.relocation_count / episode.num_containers)
    return float(statistics.fmean(ratios))


def build_figure(summary: Path, manifest: Path, output_prefix: Path) -> dict[str, object]:
    """Render a colorblind-safe single-column PDF and 300-DPI PNG."""

    payload = json.loads(summary.read_text(encoding="utf-8"))
    if payload.get("evaluation_horizons") != list(HORIZONS):
        raise ValueError("summary does not contain the fixed H=0/5/10/20 protocol")
    methods = payload.get("methods", {})
    if not isinstance(methods, dict) or any(name not in methods for name in MOE_METHODS):
        raise ValueError("summary is missing one or more final MoE seeds")

    seed_scores = np.asarray(
        [
            [
                methods[name]["horizons"][str(horizon)][
                    "mean_relocations_per_container"
                ]
                for horizon in HORIZONS
            ]
            for name in MOE_METHODS
        ],
        dtype=np.float64,
    )
    moe_mean = seed_scores.mean(axis=0)
    moe_std = seed_scores.std(axis=0, ddof=1)
    leveling = _leveling_mean_r_per_n(manifest)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
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

    x = np.arange(len(HORIZONS), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.65, alpha=0.75)

    # Small jitter exposes all three frozen seeds without adding extra methods
    # to the legend or obscuring the across-seed trend.
    jitter = np.asarray((-0.055, 0.0, 0.055))
    for seed_index, scores in enumerate(seed_scores):
        ax.scatter(
            x + jitter[seed_index],
            scores,
            s=15,
            color=MOE_COLOR,
            alpha=0.38,
            edgecolors="none",
            zorder=3,
        )

    moe_handle = ax.errorbar(
        x,
        moe_mean,
        yerr=moe_std,
        color=MOE_COLOR,
        marker="o",
        markersize=4.8,
        linewidth=2.0,
        capsize=2.5,
        capthick=1.0,
        label="MoE-LoRA (3-seed mean ± s.d.)",
        zorder=4,
    )
    leveling_handle = ax.axhline(
        leveling,
        color=LEVELING_COLOR,
        linestyle=(0, (5, 2.5)),
        linewidth=1.8,
        label="Leveling",
        zorder=2,
    )

    for index, (position, value) in enumerate(zip(x, moe_mean, strict=True)):
        offset = (0, 7) if index == 0 else (0, -11)
        vertical_alignment = "bottom" if index == 0 else "top"
        ax.annotate(
            f"{value:.3f}",
            (position, value),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=6.8,
            color=MOE_COLOR,
        )
    ax.annotate(
        f"{leveling:.3f}",
        (x[-1], leveling),
        xytext=(-3, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=LEVELING_COLOR,
    )

    ax.set_xticks(x, [str(horizon) for horizon in HORIZONS])
    ax.set_xlabel("Lookahead horizon H")
    ax.set_ylabel("Mean relocations per container (R/N)")
    ax.set_xlim(-0.22, len(HORIZONS) - 0.78)
    ax.set_ylim(0.69, 1.115)
    ax.text(
        0.02,
        0.04,
        "Lower is better",
        transform=ax.transAxes,
        fontsize=7,
        color="#666666",
    )
    ax.legend(
        handles=(moe_handle, leveling_handle),
        loc="upper right",
        handlelength=2.4,
    )
    fig.tight_layout(pad=0.45)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return {
        "summary": str(summary),
        "manifest": str(manifest),
        "leveling_mean_r_per_n": leveling,
        "moe_mean_r_per_n": moe_mean.tolist(),
        "moe_sample_std_r_per_n": moe_std.tolist(),
        "pdf": str(pdf_path),
        "png": str(png_path),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "output/moe_lora_caserta_20260918/summary.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "data/online_crp/caserta_table2_840.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).resolve().parent / "fig_moe_caserta_horizons",
    )
    args = parser.parse_args()
    result = build_figure(args.summary, args.manifest, args.output_prefix)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
