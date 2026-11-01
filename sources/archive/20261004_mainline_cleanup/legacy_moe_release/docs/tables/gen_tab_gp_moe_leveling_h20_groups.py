"""Generate the group-wise H=20 GP, MoE-LoRA, and Leveling table.

Each displayed method/group entry is one arithmetic mean of per-instance R/N.
GP averages 30 independently evolved rules and MoE-LoRA averages three formal
training seeds.  Every rule/checkpoint is evaluated on the same 40 fixed
Caserta instances in each size group; no seed is selected.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[3]
GP_ROOT = ROOT / "GP/results/h20_four_terminal_sets"
MOE_SUMMARY = ROOT / "Mijn_RL/output/moe_lora_caserta_20260918/summary.json"
LEVELING_SUMMARY = ROOT / "Leveling/results/caserta_table2_840/summary.json"
OUTPUT_DIR = Path(__file__).resolve().parent

OUTPUT_CSV = OUTPUT_DIR / "tab_gp_moe_leveling_h20_groups.csv"
OUTPUT_MD = OUTPUT_DIR / "tab_gp_moe_leveling_h20_groups.md"
OUTPUT_TEX = OUTPUT_DIR / "tab_gp_moe_leveling_h20_groups.tex"
OUTPUT_STANDALONE_TEX = OUTPUT_DIR / "tab_gp_moe_leveling_h20_groups_standalone.tex"
OUTPUT_JSON = OUTPUT_DIR / "tab_gp_moe_leveling_h20_groups_data.json"

CASERTA_SHA256 = "c28c975c8026a59a1b7995f8afe8a41e9e48b469835b9d2227a9bbe7a5371a12"
GROUPS = (
    "3-3",
    "3-4",
    "3-5",
    "3-6",
    "3-7",
    "3-8",
    "4-4",
    "4-5",
    "4-6",
    "4-7",
    "5-4",
    "5-5",
    "5-6",
    "5-7",
    "5-8",
    "5-9",
    "5-10",
    "6-6",
    "6-10",
    "10-6",
    "10-10",
)
GP_VARIANTS = {
    "C": "C",
    "C_E": "C+E",
    "C_O": "C+O",
    "C_E_O": "C+E+O",
}
MOE_RUNS = ("MoE_seed20260913", "MoE_seed20260923", "MoE_seed20260933")
METHOD_COLUMNS = ("Leveling", "GP-C", "GP-C+E", "GP-C+O", "GP-C+E+O", "MoE-LoRA")


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def group_n(group: str) -> int:
    height, stacks = (int(value) for value in group.split("-"))
    return height * stacks


def load_rows() -> list[dict[str, object]]:
    gp_summary = load_json(GP_ROOT / "summary.json")
    moe_summary = load_json(MOE_SUMMARY)
    leveling_summary = load_json(LEVELING_SUMMARY)
    hashes = {
        str(gp_summary["test_manifest_sha256"]),
        str(moe_summary["manifest_sha256"]),
        str(leveling_summary["benchmark_sha256"]),
    }
    if hashes != {CASERTA_SHA256}:
        raise AssertionError(f"mismatched Caserta manifests: {hashes}")

    leveling_by_group: dict[str, float] = {}
    for entry in leveling_summary["baseline"]["by_scale"].values():
        group = str(entry["paper_group_W-S"])
        if int(entry["episode_count"]) != 40:
            raise AssertionError(f"Leveling group {group} does not contain 40 instances")
        leveling_by_group[group] = float(entry["mean_relocations"]) / group_n(group)

    gp_by_variant: dict[str, dict[str, list[float]]] = {}
    gp_overall: dict[str, list[float]] = {}
    for directory_name, label in GP_VARIANTS.items():
        paths = sorted((GP_ROOT / "runs" / "H20" / directory_name).glob("run_*/run_complete.json"))
        if len(paths) != 30:
            raise AssertionError(f"GP {label} has {len(paths)} completed runs, expected 30")
        gp_by_variant[label] = {group: [] for group in GROUPS}
        gp_overall[label] = []
        for path in paths:
            run = load_json(path)
            if str(run["terminal_set"]) != label:
                raise AssertionError(f"terminal-set mismatch in {path}")
            summary = run["summary"]
            gp_overall[label].append(float(summary["mean_relocations_per_container"]))
            groups = summary["by_caserta_group"]
            if set(groups) != set(GROUPS):
                raise AssertionError(f"unexpected GP group set in {path}")
            for group in GROUPS:
                entry = groups[group]
                if int(entry["episodes"]) != 40:
                    raise AssertionError(f"GP {label}/{group} does not contain 40 instances")
                gp_by_variant[label][group].append(float(entry["mean_relocations_per_container"]))

    methods = moe_summary["methods"]
    moe_by_group = {group: [] for group in GROUPS}
    moe_overall: list[float] = []
    for name in MOE_RUNS:
        horizon = methods[name]["horizons"]["20"]
        moe_overall.append(float(horizon["mean_relocations_per_container"]))
        groups = horizon["by_caserta_group"]
        if set(groups) != set(GROUPS):
            raise AssertionError(f"unexpected MoE group set for {name}")
        for group in GROUPS:
            entry = groups[group]
            if int(entry["episodes"]) != 40:
                raise AssertionError(f"MoE {name}/{group} does not contain 40 instances")
            moe_by_group[group].append(float(entry["mean_relocations_per_container"]))

    rows: list[dict[str, object]] = []
    for group in GROUPS:
        rows.append(
            {
                "Group": group,
                "N": group_n(group),
                "Instances": 40,
                "Leveling": leveling_by_group[group],
                **{
                    f"GP-{label}": fmean(gp_by_variant[label][group])
                    for label in GP_VARIANTS.values()
                },
                "MoE-LoRA": fmean(moe_by_group[group]),
            }
        )

    # Since every group contains 40 instances, the overall per-instance mean is
    # the unweighted mean of the 21 group means.  Use released overall values for
    # GP/MoE as an independent consistency check.
    overall = {
        "Group": "Overall",
        "N": None,
        "Instances": 840,
        "Leveling": fmean(leveling_by_group[group] for group in GROUPS),
        **{f"GP-{label}": fmean(gp_overall[label]) for label in GP_VARIANTS.values()},
        "MoE-LoRA": fmean(moe_overall),
    }
    for column in METHOD_COLUMNS:
        group_mean = fmean(float(row[column]) for row in rows)
        if abs(group_mean - float(overall[column])) > 1e-12:
            raise AssertionError(f"overall mismatch for {column}: {group_mean} vs {overall[column]}")
    rows.append(overall)
    return rows


def display_value(value: float, best: float, *, latex: bool = False) -> str:
    text = f"{value:.4f}"
    if abs(value - best) <= 5e-13:
        return rf"\textbf{{{text}}}" if latex else f"**{text}**"
    return text


def write_outputs(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ("Group", "N", "Instances", *METHOD_COLUMNS)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (f"{float(row[key]):.8f}" if key in METHOD_COLUMNS else row[key])
                    for key in fields
                }
            )

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "horizon": 20,
                "metric": "mean_relocations_per_container",
                "test_manifest_sha256": CASERTA_SHA256,
                "aggregation": {
                    "instances_per_group": 40,
                    "gp_runs_averaged": 30,
                    "moe_runs_averaged": 3,
                    "seed_selection": False,
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown = [
        "| Group | N | Leveling | GP-C | GP-C+E | GP-C+O | GP-C+E+O | MoE-LoRA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best = min(float(row[column]) for column in METHOD_COLUMNS)
        n_text = "--" if row["N"] is None else str(row["N"])
        values = [display_value(float(row[column]), best) for column in METHOD_COLUMNS]
        markdown.append(f"| {row['Group']} | {n_text} | " + " | ".join(values) + " |")
    markdown.extend(
        [
            "",
            "Entries are arithmetic means of per-instance R/N; lower is better. Each size group contains 40 fixed Caserta instances. GP averages 30 independent runs per terminal set; MoE-LoRA averages three formal seeds; Leveling is deterministic. No seed is selected using Caserta performance.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    latex_rows: list[str] = []
    for index, row in enumerate(rows):
        best = min(float(row[column]) for column in METHOD_COLUMNS)
        values = [display_value(float(row[column]), best, latex=True) for column in METHOD_COLUMNS]
        n_text = "--" if row["N"] is None else str(row["N"])
        if index == len(rows) - 1:
            latex_rows.append(r"    \midrule")
        latex_rows.append(
            f"    {row['Group']} & {n_text} & " + " & ".join(values) + r" \\"
        )

    latex = rf"""% Requires \usepackage{{booktabs}} and \usepackage{{graphicx}}.
\begin{{table*}}[t]
  \centering
  \caption{{Group-wise H=20 comparison on the 840-instance Caserta benchmark.
  Entries are arithmetic means of per-instance $R/N$; lower is better.}}
  \label{{tab:gp_moe_leveling_h20_groups}}
  \setlength{{\tabcolsep}}{{4.0pt}}
  \renewcommand{{\arraystretch}}{{1.06}}
  \small
  \resizebox{{\textwidth}}{{!}}{{%
  \begin{{tabular}}{{lrrrrrrr}}
    \toprule
    Group & $N$ & Leveling & GP-$C$ & GP-$C{{+}}E$ & GP-$C{{+}}O$ & GP-$C{{+}}E{{+}}O$ & MoE-LoRA \\
    \midrule
{chr(10).join(latex_rows)}
    \bottomrule
  \end{{tabular}}%
  }}
  \vspace{{2pt}}
  \parbox{{0.98\textwidth}}{{\footnotesize \textit{{Notes:}} Each size group
  contains 40 fixed Caserta instances. GP entries average 30 independently
  evolved rules per terminal set; MoE-LoRA entries average three formal training
  seeds; Leveling is deterministic. No seed was selected using Caserta
  performance. Group $S$-$W$ denotes initial height $S$ and number of stacks
  $W$, so $N=S\times W$.}}
\end{{table*}}
"""
    OUTPUT_TEX.write_text(latex, encoding="utf-8")
    standalone = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=14mm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\pagestyle{empty}
\begin{document}
\input{tab_gp_moe_leveling_h20_groups.tex}
\end{document}
"""
    OUTPUT_STANDALONE_TEX.write_text(standalone, encoding="utf-8")


def main() -> int:
    rows = load_rows()
    write_outputs(rows)
    print(f"rows={len(rows)}")
    print(f"csv={OUTPUT_CSV}")
    print(f"markdown={OUTPUT_MD}")
    print(f"latex={OUTPUT_TEX}")
    print(f"standalone_latex={OUTPUT_STANDALONE_TEX}")
    print(f"data_json={OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
