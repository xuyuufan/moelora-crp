"""Plot H=20 Caserta reward curves for the base policy and MoE-LoRA.

The figure uses the released 840-instance Caserta evaluation summary.  Reward
is defined as negative relocations per container (-R/N), so higher is better.
MoE-LoRA is summarized across the three formal seeds without selecting a seed
on the Caserta benchmark.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
GROUP_PATH = OUTPUT_DIR / "group_comparison.csv"
DATA_PATH = OUTPUT_DIR / "h20_base_vs_moe_reward_curve_data.csv"
FIGURE_PDF = OUTPUT_DIR / "fig_h20_base_vs_moe_reward_curve.pdf"
FIGURE_PNG = OUTPUT_DIR / "fig_h20_base_vs_moe_reward_curve.png"

CASERTA_SHA256 = "c28c975c8026a59a1b7995f8afe8a41e9e48b469835b9d2227a9bbe7a5371a12"
BASE_METHOD = "MijnRL_base"
MOE_METHODS = (
    "MoE_seed20260913",
    "MoE_seed20260923",
    "MoE_seed20260933",
)


def _load_summary() -> dict[str, object]:
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary.json must contain a JSON object")
    if payload.get("manifest_sha256") != CASERTA_SHA256:
        raise AssertionError("unexpected Caserta test-manifest hash")
    if int(payload.get("episode_count", -1)) != 840:
        raise AssertionError("the formal Caserta evaluation must contain 840 episodes")
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        raise AssertionError("summary.json has no method results")
    expected = {BASE_METHOD, *MOE_METHODS}
    if set(methods) != expected:
        raise AssertionError(f"unexpected methods: {sorted(methods)}")
    return payload


def _group_order() -> list[dict[str, object]]:
    with GROUP_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 21:
        raise AssertionError("the Caserta benchmark must contain 21 parameter groups")
    result: list[dict[str, object]] = []
    for row in rows:
        episodes = int(row["episodes"])
        if episodes != 40:
            raise AssertionError(f"group {row['group']} does not contain 40 episodes")
        result.append(
            {
                "group": row["group"],
                "initial_height": int(row["initial_height"]),
                "num_stacks": int(row["num_stacks"]),
                "num_containers": int(row["num_containers"]),
                "episodes": episodes,
            }
        )
    return result


def _h20(methods: dict[str, object], method: str) -> dict[str, object]:
    record = methods[method]
    if not isinstance(record, dict):
        raise AssertionError(f"malformed result for {method}")
    horizons = record.get("horizons")
    if not isinstance(horizons, dict) or "20" not in horizons:
        raise AssertionError(f"{method} is missing H=20 results")
    result = horizons["20"]
    if not isinstance(result, dict):
        raise AssertionError(f"malformed H=20 result for {method}")
    return result


def load_curve_data() -> tuple[list[dict[str, object]], float, float]:
    summary = _load_summary()
    methods = summary["methods"]
    if not isinstance(methods, dict):
        raise AssertionError("summary.json has no method results")

    base_h20 = _h20(methods, BASE_METHOD)
    moe_h20 = [_h20(methods, method) for method in MOE_METHODS]
    base_groups = base_h20.get("by_caserta_group")
    if not isinstance(base_groups, dict):
        raise AssertionError("base H=20 result has no group breakdown")

    ordered_groups = _group_order()
    rows: list[dict[str, object]] = []
    for metadata in ordered_groups:
        group = str(metadata["group"])
        base_group = base_groups.get(group)
        if not isinstance(base_group, dict):
            raise AssertionError(f"base result is missing group {group}")
        if int(base_group["episodes"]) != 40:
            raise AssertionError(f"base group {group} has an invalid episode count")
        base_rpn = float(base_group["mean_relocations_per_container"])

        seed_rpn: list[float] = []
        for method, result in zip(MOE_METHODS, moe_h20, strict=True):
            groups = result.get("by_caserta_group")
            if not isinstance(groups, dict) or not isinstance(groups.get(group), dict):
                raise AssertionError(f"{method} is missing group {group}")
            group_record = groups[group]
            if int(group_record["episodes"]) != 40:
                raise AssertionError(f"{method} group {group} has an invalid episode count")
            seed_rpn.append(float(group_record["mean_relocations_per_container"]))

        moe_mean_rpn = float(np.mean(seed_rpn))
        moe_seed_std_rpn = float(np.std(seed_rpn, ddof=1))
        if abs(moe_seed_std_rpn) < 1e-12:
            moe_seed_std_rpn = 0.0
        delta_reward = base_rpn - moe_mean_rpn
        if abs(delta_reward) < 1e-12:
            delta_reward = 0.0
        rows.append(
            {
                **metadata,
                "base_mean_rpn": base_rpn,
                "base_mean_reward": -base_rpn,
                "seed20260913_mean_rpn": seed_rpn[0],
                "seed20260923_mean_rpn": seed_rpn[1],
                "seed20260933_mean_rpn": seed_rpn[2],
                "moe_mean_rpn": moe_mean_rpn,
                "moe_seed_std_rpn": moe_seed_std_rpn,
                "moe_mean_reward": -moe_mean_rpn,
                "moe_minus_base_reward": delta_reward,
            }
        )

    expected_groups = set(base_groups)
    observed_groups = {str(row["group"]) for row in rows}
    if expected_groups != observed_groups:
        raise AssertionError("group labels differ between summary.json and group_comparison.csv")

    base_overall_rpn = float(base_h20["mean_relocations_per_container"])
    moe_overall_rpn = float(
        np.mean([float(result["mean_relocations_per_container"]) for result in moe_h20])
    )
    return rows, base_overall_rpn, moe_overall_rpn


def write_data(rows: list[dict[str, object]]) -> None:
    with DATA_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figure(
    rows: list[dict[str, object]], base_overall_rpn: float, moe_overall_rpn: float
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "legend.frameon": False,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.17,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 4.2,
        }
    )

    x = np.arange(len(rows), dtype=float)
    labels = [str(row["group"]) for row in rows]
    base_reward = np.asarray([row["base_mean_reward"] for row in rows], dtype=float)
    moe_reward = np.asarray([row["moe_mean_reward"] for row in rows], dtype=float)
    moe_std = np.asarray([row["moe_seed_std_rpn"] for row in rows], dtype=float)
    delta_reward = np.asarray(
        [row["moe_minus_base_reward"] for row in rows], dtype=float
    )

    base_color = "#0072B2"
    moe_color = "#D55E00"
    positive_color = "#009E73"
    negative_color = "#CC79A7"

    figure, (curve_axis, delta_axis) = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.9),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": (2.55, 1.0)},
    )

    curve_axis.plot(
        x,
        base_reward,
        color=base_color,
        marker="o",
        label="Original H20 base",
        zorder=4,
    )
    curve_axis.plot(
        x,
        moe_reward,
        color=moe_color,
        marker="s",
        label="MoE-LoRA (3-seed mean)",
        zorder=5,
    )
    curve_axis.fill_between(
        x,
        moe_reward - moe_std,
        moe_reward + moe_std,
        color=moe_color,
        alpha=0.14,
        linewidth=0,
        label=r"MoE-LoRA $\pm$ 1 seed SD",
        zorder=2,
    )
    curve_axis.set_ylabel(r"Mean reward $-R/N$")
    curve_axis.set_title("H=20 reward comparison on the Caserta benchmark", pad=9)
    curve_axis.legend(loc="lower left", ncol=3, columnspacing=1.25, handlelength=2.0)
    curve_axis.text(
        0.99,
        0.97,
        (
            f"Overall mean R/N: base {base_overall_rpn:.6f}, "
            f"MoE-LoRA {moe_overall_rpn:.6f}\n"
            f"Overall reward difference: {base_overall_rpn - moe_overall_rpn:+.6f}"
        ),
        transform=curve_axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
    )
    curve_axis.text(
        0.01,
        0.96,
        "Higher is better",
        transform=curve_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#4B5563",
        fontstyle="italic",
    )

    bar_colors = [positive_color if value >= 0 else negative_color for value in delta_reward]
    delta_axis.bar(x, delta_reward, width=0.72, color=bar_colors, edgecolor="white", linewidth=0.35)
    delta_axis.axhline(0.0, color="#374151", linewidth=0.85, zorder=3)
    delta_axis.set_ylabel("MoE - base\nreward")
    delta_axis.set_xlabel("Caserta group: initial height - number of stacks")
    delta_axis.text(
        0.01,
        0.93,
        "Positive values favor MoE-LoRA",
        transform=delta_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        color="#4B5563",
        fontstyle="italic",
    )

    # Delimit blocks sharing the same initial height.
    heights = [int(row["initial_height"]) for row in rows]
    for index in range(1, len(heights)):
        if heights[index] != heights[index - 1]:
            for axis in (curve_axis, delta_axis):
                axis.axvline(index - 0.5, color="#9CA3AF", linewidth=0.65, alpha=0.45)

    delta_axis.set_xticks(x)
    delta_axis.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    delta_axis.set_xlim(-0.6, len(rows) - 0.4)
    curve_axis.grid(axis="x", alpha=0.08)
    delta_axis.grid(axis="x", alpha=0.08)
    figure.savefig(FIGURE_PDF, facecolor="white")
    figure.savefig(FIGURE_PNG, facecolor="white", dpi=300)
    plt.close(figure)


def main() -> int:
    rows, base_overall_rpn, moe_overall_rpn = load_curve_data()
    write_data(rows)
    write_figure(rows, base_overall_rpn, moe_overall_rpn)
    print(f"data={DATA_PATH}")
    print(f"pdf={FIGURE_PDF}")
    print(f"png={FIGURE_PNG}")
    print(f"base_overall_mean_rpn={base_overall_rpn:.12f}")
    print(f"moe_overall_mean_rpn={moe_overall_rpn:.12f}")
    print(f"moe_minus_base_reward={base_overall_rpn - moe_overall_rpn:+.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
