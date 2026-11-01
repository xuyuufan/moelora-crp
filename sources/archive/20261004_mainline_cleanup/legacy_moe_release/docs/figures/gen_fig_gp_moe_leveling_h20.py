"""Generate the paper-ready H=20 GP vs. MoE vs. Leveling comparison.

The script reads only released result artifacts.  It deliberately keeps the
four GP terminal sets separate and reports all formal seeds; no method or seed
is selected using Caserta performance.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
GP_RUNS = ROOT / "GP/results/h20_four_terminal_sets/run_metrics.csv"
GP_SUMMARY = ROOT / "GP/results/h20_four_terminal_sets/summary.json"
MOE_SUMMARY = ROOT / "Mijn_RL/output/moe_lora_caserta_20260918/summary.json"
LEVELING_SUMMARY = ROOT / "Leveling/results/caserta_table2_840/summary.json"
FIGURE_DIR = Path(__file__).resolve().parent
TABLE_DIR = FIGURE_DIR.parent / "tables"

FIGURE_PDF = FIGURE_DIR / "fig_gp_moe_leveling_h20.pdf"
FIGURE_PNG = FIGURE_DIR / "fig_gp_moe_leveling_h20.png"
TABLE_CSV = TABLE_DIR / "tab_gp_moe_leveling_h20.csv"
TABLE_MD = TABLE_DIR / "tab_gp_moe_leveling_h20.md"
TABLE_TEX = TABLE_DIR / "tab_gp_moe_leveling_h20.tex"

CASERTA_SHA256 = "c28c975c8026a59a1b7995f8afe8a41e9e48b469835b9d2227a9bbe7a5371a12"


@dataclass(frozen=True)
class MethodResults:
    method: str
    variant: str
    plot_label: str
    mean_r_per_n: tuple[float, ...]
    total_relocations: tuple[int, ...]
    color: str
    marker: str = "o"

    @property
    def runs(self) -> int:
        return len(self.mean_r_per_n)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_results() -> list[MethodResults]:
    gp_summary = _load_json(GP_SUMMARY)
    moe_summary = _load_json(MOE_SUMMARY)
    leveling_summary = _load_json(LEVELING_SUMMARY)
    hashes = {
        str(gp_summary["test_manifest_sha256"]),
        str(moe_summary["manifest_sha256"]),
        str(leveling_summary["benchmark_sha256"]),
    }
    if hashes != {CASERTA_SHA256}:
        raise AssertionError(f"comparison uses mismatched test manifests: {hashes}")

    gp_rows: dict[str, list[tuple[float, int]]] = {
        name: [] for name in ("C", "C+E", "C+O", "C+E+O")
    }
    with GP_RUNS.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["horizon"]) != 20:
                raise AssertionError("formal GP comparison contains a non-H=20 row")
            terminal_set = row["terminal_set"]
            if terminal_set not in gp_rows:
                raise AssertionError(f"unexpected GP terminal set: {terminal_set}")
            gp_rows[terminal_set].append(
                (
                    float(row["mean_relocations_per_container"]),
                    int(row["total_relocations"]),
                )
            )
    if any(len(values) != 30 for values in gp_rows.values()):
        raise AssertionError("each GP terminal set must contain 30 formal runs")

    leveling = leveling_summary["baseline"]
    if not isinstance(leveling, dict):
        raise AssertionError("Leveling summary is malformed")
    leveling_r_per_n = float(gp_summary["leveling"]["mean_relocations_per_container"])
    leveling_total = int(leveling["total_relocations"])
    if leveling_total != 36_024:
        raise AssertionError("Leveling regression is not 36,024")

    methods = moe_summary["methods"]
    if not isinstance(methods, dict):
        raise AssertionError("MoE summary is malformed")
    moe_r_per_n: list[float] = []
    moe_totals: list[int] = []
    for name in ("MoE_seed20260913", "MoE_seed20260923", "MoE_seed20260933"):
        horizon = methods[name]["horizons"]["20"]
        moe_r_per_n.append(float(horizon["mean_relocations_per_container"]))
        moe_totals.append(int(horizon["total_relocations"]))

    # Okabe-Ito colorblind-safe palette.  MoE-LoRA is highlighted in vermillion.
    result = [
        MethodResults(
            "Leveling",
            "Deterministic",
            "Leveling",
            (leveling_r_per_n,),
            (leveling_total,),
            "#7A7A7A",
            "D",
        )
    ]
    gp_colors = {
        "C": "#0072B2",
        "C+E": "#56B4E9",
        "C+O": "#009E73",
        "C+E+O": "#E69F00",
    }
    for terminal_set in ("C", "C+E", "C+O", "C+E+O"):
        values = gp_rows[terminal_set]
        result.append(
            MethodResults(
                "GP",
                terminal_set,
                f"GP\n{terminal_set}",
                tuple(value[0] for value in values),
                tuple(value[1] for value in values),
                gp_colors[terminal_set],
            )
        )
    result.append(
        MethodResults(
            "MoE-LoRA",
            "Dynamic routing",
            "MoE-LoRA",
            tuple(moe_r_per_n),
            tuple(moe_totals),
            "#D55E00",
            "s",
        )
    )
    return result


def _distribution(values: Sequence[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def write_figure(results: Sequence[MethodResults]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
        }
    )
    figure, axis = plt.subplots(figsize=(6.75, 3.15))
    rng = np.random.default_rng(20260921)
    for index, result in enumerate(results):
        values = np.asarray(result.mean_r_per_n, dtype=np.float64)
        stats = _distribution(values)
        if result.runs > 1:
            jitter = rng.uniform(-0.115, 0.115, size=result.runs)
            axis.scatter(
                index + jitter,
                values,
                s=18 if result.runs > 3 else 34,
                marker=result.marker,
                color=result.color,
                alpha=0.58 if result.runs > 3 else 0.9,
                linewidths=0,
                zorder=3,
            )
            axis.vlines(
                index,
                stats["min"],
                stats["max"],
                color=result.color,
                linewidth=1.0,
                alpha=0.8,
                zorder=2,
            )
            axis.vlines(
                index,
                stats["q1"],
                stats["q3"],
                color=result.color,
                linewidth=5.5,
                alpha=0.95,
                zorder=4,
            )
            axis.scatter(
                index,
                stats["median"],
                s=42,
                marker="o",
                facecolor="white",
                edgecolor=result.color,
                linewidth=1.6,
                zorder=5,
            )
        else:
            axis.scatter(
                index,
                values[0],
                s=50,
                marker=result.marker,
                color=result.color,
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )
        axis.text(
            index,
            1.067,
            f"n={result.runs}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="#4B5563",
        )

    axis.axvspan(0.55, 4.45, color="#0072B2", alpha=0.035, zorder=0)
    axis.axvspan(4.55, 5.45, color="#D55E00", alpha=0.045, zorder=0)
    axis.set_xlim(-0.55, len(results) - 0.45)
    axis.set_ylim(0.715, 1.078)
    axis.set_xticks(range(len(results)))
    axis.set_xticklabels([result.plot_label for result in results])
    axis.set_ylabel(r"Mean relocations per container, $\mathrm{mean}(R/N)$")
    axis.set_title("Caserta benchmark (840 instances, H=20)", pad=10)
    axis.text(
        0.012,
        0.04,
        "↓ Lower is better",
        transform=axis.transAxes,
        color="#374151",
        fontsize=8.2,
        fontstyle="italic",
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="#4B5563",
                markerfacecolor="white",
                markeredgewidth=1.4,
                linewidth=4.0,
                markersize=5.5,
                label="Median and IQR",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#6B7280",
                alpha=0.65,
                markersize=5,
                label="Independent run",
            ),
        ],
        loc="center right",
        bbox_to_anchor=(0.992, 0.52),
        ncol=1,
        handlelength=1.7,
        columnspacing=1.0,
    )
    figure.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PDF, facecolor="white")
    figure.savefig(FIGURE_PNG, facecolor="white", dpi=300)
    plt.close(figure)


def _summary_rows(results: Sequence[MethodResults]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        ratio = _distribution(result.mean_r_per_n)
        total = _distribution(result.total_relocations)
        rows.append(
            {
                "method": result.method,
                "variant": result.variant,
                "horizon": 20 if result.method != "Leveling" else "independent",
                "runs": result.runs,
                **{f"mean_r_per_n_{key}": value for key, value in ratio.items()},
                **{f"total_relocations_{key}": value for key, value in total.items()},
            }
        )
    return rows


def _ratio_cell(row: dict[str, object]) -> str:
    median = float(row["mean_r_per_n_median"])
    if int(row["runs"]) == 1:
        return f"{median:.4f}"
    return (
        f"{median:.4f} "
        f"[{float(row['mean_r_per_n_q1']):.4f}, "
        f"{float(row['mean_r_per_n_q3']):.4f}]"
    )


def _total_cell(row: dict[str, object]) -> str:
    median = float(row["total_relocations_median"])
    if int(row["runs"]) == 1:
        return f"{median:,.0f}"
    return (
        f"{median:,.1f} "
        f"[{float(row['total_relocations_q1']):,.1f}, "
        f"{float(row['total_relocations_q3']):,.1f}]"
    )


def write_tables(results: Sequence[MethodResults]) -> None:
    rows = _summary_rows(results)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with TABLE_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| Method | Variant | Runs | mean(R/N), median [IQR] | Total R, median [IQR] |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['method']} | {row['variant']} | {row['runs']} | "
            f"{_ratio_cell(row)} | {_total_cell(row)} |"
        )
    markdown.extend(
        [
            "",
            "Lower is better. Results use the same 840-instance Caserta manifest at H=20; "
            "Leveling is horizon-independent. GP uses 30 independent runs per terminal set, "
            "MoE-LoRA uses three formal seeds, and Leveling is deterministic. No seed is "
            "selected using Caserta performance.",
        ]
    )
    TABLE_MD.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    latex_rows: list[str] = []
    for row in rows:
        method = str(row["method"])
        variant = str(row["variant"])
        if method == "MoE-LoRA":
            method = r"\textbf{MoE-LoRA}"
            ratio = rf"\textbf{{{_ratio_cell(row)}}}"
            total = rf"\textbf{{{_total_cell(row)}}}"
        else:
            ratio = _ratio_cell(row)
            total = _total_cell(row)
        variant = {
            "Deterministic": "--",
            "C": r"$C$",
            "C+E": r"$C{+}E$",
            "C+O": r"$C{+}O$",
            "C+E+O": r"$C{+}E{+}O$",
            "Dynamic routing": "Dynamic routing",
        }[variant]
        latex_rows.append(
            f"    {method} & {variant} & {row['runs']} & {ratio} & {total} \\\\"
        )
    latex = rf"""% Requires \usepackage{{booktabs}}.
\begin{{table}}[t]
  \centering
  \caption{{H=20 comparison on the 840-instance Caserta benchmark. Values are
  medians [IQR] across independent runs; lower is better.}}
  \label{{tab:gp_moe_leveling_h20}}
  \setlength{{\tabcolsep}}{{4.5pt}}
  \renewcommand{{\arraystretch}}{{1.08}}
  \small
  \begin{{tabular}}{{llrcc}}
    \toprule
    Method & Variant & Runs & $\mathrm{{mean}}(R/N)$ & Total relocations \\
    \midrule
{chr(10).join(latex_rows)}
    \bottomrule
  \end{{tabular}}
  \vspace{{2pt}}
  \parbox{{0.98\linewidth}}{{\footnotesize \textit{{Notes:}} All methods use the
  same Caserta manifest. GP reports 30 independently evolved rules per terminal
  set; MoE-LoRA reports three formal seeds; Leveling is deterministic and
  horizon-independent. No seed was selected using Caserta performance.}}
\end{{table}}
"""
    TABLE_TEX.write_text(latex, encoding="utf-8")


def main() -> int:
    results = load_results()
    write_figure(results)
    write_tables(results)
    print(f"figure_pdf={FIGURE_PDF}")
    print(f"figure_png={FIGURE_PNG}")
    print(f"table_tex={TABLE_TEX}")
    print(f"table_csv={TABLE_CSV}")
    print(f"table_md={TABLE_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
