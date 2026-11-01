"""Generate the paper-ready Caserta group comparison table.

The LaTeX fragment is suitable for direct inclusion in a manuscript.  A
standalone wrapper is also generated so the table can be compiled and visually
checked without the full paper source.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DEFAULT_REPORT = Path("output/moe_lora_caserta_20260918/group_comparison.json")
DEFAULT_TABLE = Path("docs/tables/tab_moe_caserta_groups.tex")
DEFAULT_STANDALONE = Path("docs/tables/tab_moe_caserta_groups_standalone.tex")
HORIZONS = (0, 5, 10, 20)
SEED_KEYS = (
    "seed20260913_mean_rpn",
    "seed20260923_mean_rpn",
    "seed20260933_mean_rpn",
)


def _cell(value: float, *, bold: bool, std: float | None = None) -> str:
    if std is None:
        rendered = f"{value:.4f}"
    else:
        rendered = rf"{value:.4f}$\pm${std:.4f}"
    return rf"\textbf{{{rendered}}}" if bold else rendered


def _table_source(report: dict[str, object]) -> str:
    groups = report["groups"]
    detail = report["seed_detail"]
    overall = report["overall"]
    assert isinstance(groups, list)
    assert isinstance(detail, list)
    assert isinstance(overall, dict)

    detail_index = {
        (str(row["group"]), int(row["horizon"])): row for row in detail
    }
    lines = [
        r"% Requires \usepackage{booktabs} and \usepackage{graphicx}.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Group-wise comparison of Leveling and MoE on Caserta Table 2.}",
        r"  \label{tab:moe_caserta_groups}",
        r"  \setlength{\tabcolsep}{3.8pt}",
        r"  \renewcommand{\arraystretch}{1.08}",
        r"  \small",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lrrrrrrrrr}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{$H=0$} & \multicolumn{2}{c}{$H=5$} & \multicolumn{2}{c}{$H=10$} & \multicolumn{2}{c}{$H=20$} \\",
        r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"    Group & $N$ & Leveling & MoE & Leveling & MoE & Leveling & MoE & Leveling & MoE \\",
        r"    \midrule",
    ]

    for row in groups:
        group = str(row["group"])
        leveling = float(row["leveling_mean_rpn"])
        cells = [group, str(int(row["num_containers"]))]
        for horizon in HORIZONS:
            entry = detail_index[(group, horizon)]
            moe = float(entry["moe_mean_rpn"])
            std = float(entry["seed_std_rpn"])
            cells.extend(
                [
                    _cell(leveling, bold=leveling < moe),
                    _cell(moe, std=std, bold=moe < leveling),
                ]
            )
        lines.append("    " + " & ".join(cells) + r" \\")

    overall_leveling = float(overall["leveling_mean_rpn"])
    # N varies across groups, so the aggregate row must not place the number
    # of benchmark instances (840) in the per-instance container-count column.
    overall_cells = ["Overall", "--"]
    for horizon in HORIZONS:
        horizon_rows = [
            row for row in detail if int(row["horizon"]) == horizon
        ]
        seed_overall = [
            statistics.fmean(float(row[key]) for row in horizon_rows)
            for key in SEED_KEYS
        ]
        moe = statistics.fmean(seed_overall)
        std = statistics.stdev(seed_overall)
        overall_cells.extend(
            [
                _cell(overall_leveling, bold=overall_leveling < moe),
                _cell(moe, std=std, bold=moe < overall_leveling),
            ]
        )
    lines.extend(
        [
            r"    \midrule",
            "    " + " & ".join(overall_cells) + r" \\",
            r"    \bottomrule",
            r"  \end{tabular}%",
            r"  }",
            r"  \vspace{2pt}",
            r"  \parbox{\linewidth}{\footnotesize \textit{Notes:} Entries are mean relocations per container ($R/N$); lower is better. Each group contains 40 instances. MoE reports the mean $\pm$ sample standard deviation across three independently trained seeds. Bold indicates the lower value within each horizon. Group $S$-$W$ denotes initial height $S$ and number of stacks $W$. Leveling uses only public stack heights, so its complete action sequence is identical for $H\in\{0,5,10,20\}$; repeated values are shown for direct comparison.}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _standalone_source() -> str:
    return "\n".join(
        [
            r"% !TeX program = lualatex",
            r"\documentclass[10pt]{article}",
            r"\usepackage[a4paper,margin=22mm]{geometry}",
            r"\usepackage{graphicx}",
            r"% Preview-only fallbacks. The manuscript should load booktabs.",
            r"\newcommand{\toprule}{\hline}",
            r"\newcommand{\midrule}{\hline}",
            r"\newcommand{\bottomrule}{\hline}",
            r"\def\cmidrule(#1)#2{\cline{#2}}",
            r"\makeatletter",
            r"\renewenvironment{table}[1][]{\par\noindent\begin{minipage}{\textwidth}\def\@captype{table}}{\end{minipage}\par}",
            r"\makeatother",
            r"\pagestyle{empty}",
            r"\begin{document}",
            r"\input{tab_moe_caserta_groups.tex}",
            r"\end{document}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--standalone", type=Path, default=DEFAULT_STANDALONE)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.standalone.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table_source(report), encoding="utf-8")
    args.standalone.write_text(_standalone_source(), encoding="utf-8")
    print(args.table)
    print(args.standalone)


if __name__ == "__main__":
    main()
