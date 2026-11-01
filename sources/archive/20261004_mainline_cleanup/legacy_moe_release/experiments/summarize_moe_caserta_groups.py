"""Build the 21-group Caserta comparison for Leveling and final MoE seeds.

The Caserta Table 2 manifest contains 21 parameter groups with 40 instances
per group.  This script recomputes Leveling from the public environment,
checks that its decisions are invariant to H, and aggregates the immutable
per-instance MoE evaluation records by group and evaluation horizon.

Negative deltas mean that MoE uses fewer relocations per container.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from scrp.online_crp import (
    LevelingBaseline,
    OnlineCRPEpisode,
    OnlineCRPEnvironment,
    load_episode_manifest,
)


DEFAULT_MANIFEST = Path("data/online_crp/caserta_table2_840.json")
DEFAULT_RESULTS = Path("output/moe_lora_caserta_20260918/episode_results.jsonl")
DEFAULT_OUTPUT_DIR = Path("output/moe_lora_caserta_20260918")
HORIZONS = (0, 5, 10, 20)
MOE_METHODS = (
    "MoE_seed20260913",
    "MoE_seed20260923",
    "MoE_seed20260933",
)


def _run_leveling(
    episode: OnlineCRPEpisode, horizon: int
) -> tuple[int, tuple[int, ...]]:
    """Run Leveling using public stack heights and return cost plus actions."""

    environment = OnlineCRPEnvironment(episode, lookahead_horizon=horizon)
    actions: list[int] = []
    while not environment.state.terminated:
        legal = environment.legal_destinations()
        destination = LevelingBaseline.select_destination(environment.state, legal)
        actions.append(destination)
        environment.step(destination)
    return environment.relocation_count, tuple(actions)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"no evaluation records found in {path}")
    return records


def _mean(values: Iterable[float | int]) -> float:
    return float(statistics.fmean(values))


def build_comparison(
    episodes: Sequence[OnlineCRPEpisode], records: Sequence[dict[str, object]]
) -> dict[str, object]:
    """Return validated group summaries and workbook-ready detail rows."""

    if len(episodes) != 840:
        raise ValueError(f"expected 840 Caserta instances, got {len(episodes)}")
    group_order = tuple(dict.fromkeys(episode.parameter_group for episode in episodes))
    if len(group_order) != 21:
        raise ValueError(f"expected 21 Caserta groups, got {len(group_order)}")

    episodes_by_id = {episode.episode_id: episode for episode in episodes}
    if len(episodes_by_id) != len(episodes):
        raise ValueError("duplicate episode_id in Caserta manifest")
    group_episodes: dict[str, list[OnlineCRPEpisode]] = defaultdict(list)
    for episode in episodes:
        group_episodes[episode.parameter_group].append(episode)
    if any(len(group_episodes[group]) != 40 for group in group_order):
        raise ValueError("every Caserta group must contain exactly 40 instances")

    # Leveling depends on stack heights only.  Re-running all four horizons
    # makes that protocol property executable rather than merely documented.
    leveling_by_id: dict[str, int] = {}
    for episode in episodes:
        reference_cost, reference_actions = _run_leveling(episode, HORIZONS[0])
        for horizon in HORIZONS[1:]:
            cost, actions = _run_leveling(episode, horizon)
            if (cost, actions) != (reference_cost, reference_actions):
                raise AssertionError(
                    f"Leveling changed with H for {episode.episode_id}: "
                    f"H=0 cost {reference_cost}, H={horizon} cost {cost}"
                )
        leveling_by_id[episode.episode_id] = reference_cost
    if sum(leveling_by_id.values()) != 36_024:
        raise AssertionError("Leveling total no longer matches the verified 36,024")

    moe_records = [record for record in records if record.get("algorithm") in MOE_METHODS]
    expected_records = len(episodes) * len(HORIZONS) * len(MOE_METHODS)
    if len(moe_records) != expected_records:
        raise ValueError(
            f"expected {expected_records} MoE records, got {len(moe_records)}"
        )

    indexed: dict[tuple[str, int, str], dict[str, object]] = {}
    for record in moe_records:
        episode_id = str(record["episode_id"])
        horizon = int(record["lookahead_horizon"])
        method = str(record["algorithm"])
        key = (method, horizon, episode_id)
        if key in indexed:
            raise ValueError(f"duplicate evaluation record: {key}")
        episode = episodes_by_id.get(episode_id)
        if episode is None:
            raise ValueError(f"unknown episode in evaluation results: {episode_id}")
        if str(record["parameter_group"]) != episode.parameter_group:
            raise ValueError(f"group mismatch for {episode_id}")
        indexed[key] = record

    group_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for group in group_order:
        members = group_episodes[group]
        height_text, stacks_text = group.split("-", maxsplit=1)
        initial_height = int(height_text)
        num_stacks = int(stacks_text)
        num_containers = members[0].num_containers
        if any(episode.num_containers != num_containers for episode in members):
            raise ValueError(f"group {group} contains mixed container counts")
        if num_containers != initial_height * num_stacks:
            raise ValueError(f"group {group} does not satisfy N=S*W")

        leveling_costs = [leveling_by_id[episode.episode_id] for episode in members]
        leveling_mean_relocations = _mean(leveling_costs)
        leveling_mean_rpn = _mean(cost / num_containers for cost in leveling_costs)
        wide: dict[str, object] = {
            "group": group,
            "initial_height": initial_height,
            "num_stacks": num_stacks,
            "num_containers": num_containers,
            "episodes": len(members),
            "leveling_mean_relocations": leveling_mean_relocations,
            "leveling_mean_rpn": leveling_mean_rpn,
        }

        for horizon in HORIZONS:
            seed_means_rpn: list[float] = []
            seed_means_relocations: list[float] = []
            wins = ties = losses = 0
            for method in MOE_METHODS:
                method_relocations: list[int] = []
                for episode in members:
                    record = indexed[(method, horizon, episode.episode_id)]
                    relocation = int(record["relocations"])
                    method_relocations.append(relocation)
                    delta = relocation - leveling_by_id[episode.episode_id]
                    wins += delta < 0
                    ties += delta == 0
                    losses += delta > 0
                seed_means_relocations.append(_mean(method_relocations))
                seed_means_rpn.append(
                    _mean(value / num_containers for value in method_relocations)
                )

            moe_mean_relocations = _mean(seed_means_relocations)
            moe_mean_rpn = _mean(seed_means_rpn)
            delta_rpn = moe_mean_rpn - leveling_mean_rpn
            relative_change = delta_rpn / leveling_mean_rpn
            prefix = f"h{horizon}"
            wide.update(
                {
                    f"{prefix}_moe_mean_rpn": moe_mean_rpn,
                    f"{prefix}_seed_std_rpn": float(statistics.stdev(seed_means_rpn)),
                    f"{prefix}_delta_rpn": delta_rpn,
                    f"{prefix}_relative_change": relative_change,
                    f"{prefix}_wins": wins,
                    f"{prefix}_ties": ties,
                    f"{prefix}_losses": losses,
                }
            )
            detail_rows.append(
                {
                    "group": group,
                    "initial_height": initial_height,
                    "num_stacks": num_stacks,
                    "num_containers": num_containers,
                    "episodes": len(members),
                    "horizon": horizon,
                    "leveling_mean_rpn": leveling_mean_rpn,
                    "seed20260913_mean_rpn": seed_means_rpn[0],
                    "seed20260923_mean_rpn": seed_means_rpn[1],
                    "seed20260933_mean_rpn": seed_means_rpn[2],
                    "moe_mean_rpn": moe_mean_rpn,
                    "seed_std_rpn": float(statistics.stdev(seed_means_rpn)),
                    "delta_rpn": delta_rpn,
                    "relative_change": relative_change,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            )
        group_rows.append(wide)

    overall_leveling_rpn = _mean(
        leveling_by_id[episode.episode_id] / episode.num_containers
        for episode in episodes
    )
    overall: dict[str, object] = {
        "episodes": len(episodes),
        "groups": len(group_order),
        "leveling_total_relocations": sum(leveling_by_id.values()),
        "leveling_mean_rpn": overall_leveling_rpn,
        "leveling_horizon_invariant": True,
    }
    for horizon in HORIZONS:
        seed_means = []
        for method in MOE_METHODS:
            method_rows = [
                indexed[(method, horizon, episode.episode_id)] for episode in episodes
            ]
            seed_means.append(
                _mean(float(record["relocations_per_container"]) for record in method_rows)
            )
        mean_rpn = _mean(seed_means)
        overall[f"h{horizon}_moe_mean_rpn"] = mean_rpn
        overall[f"h{horizon}_relative_change"] = (
            mean_rpn - overall_leveling_rpn
        ) / overall_leveling_rpn
        overall[f"h{horizon}_improved_groups"] = sum(
            float(row[f"h{horizon}_delta_rpn"]) < 0 for row in group_rows
        )
        overall[f"h{horizon}_tied_groups"] = sum(
            abs(float(row[f"h{horizon}_delta_rpn"])) < 1e-12
            for row in group_rows
        )
        overall[f"h{horizon}_worse_groups"] = sum(
            float(row[f"h{horizon}_delta_rpn"]) > 0 for row in group_rows
        )

    return {
        "definition": {
            "group_label": "S-W: initial physical height S and number of stacks W",
            "group_size": 40,
            "metric": "mean of per-instance R/N",
            "moe_aggregation": "mean of three seed-level group means",
            "relative_change": "(MoE - Leveling) / Leveling; negative is better",
            "paired_counts": "wins/ties/losses over 3 seeds x 40 paired outcomes",
        },
        "methods": list(MOE_METHODS),
        "horizons": list(HORIZONS),
        "overall": overall,
        "groups": group_rows,
        "seed_detail": detail_rows,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_result(mean_rpn: float, relative_change: float) -> str:
    return f"{mean_rpn:.6f} ({relative_change:+.2%})"


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    overall = report["overall"]
    groups = report["groups"]
    assert isinstance(overall, dict) and isinstance(groups, list)
    lines = [
        "# Caserta 21-group comparison: Leveling vs MoE",
        "",
        "每组 40 个实例。指标为逐实例 `R/N` 的平均值；MoE 为三个最终 seed 的组均值再取平均。括号内为相对 Leveling 的变化，负数表示 MoE 更好。",
        "",
        "| 组别 S-W | N | Leveling | H=0 MoE | H=5 MoE | H=10 MoE | H=20 MoE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        assert isinstance(row, dict)
        cells = [str(row["group"]), str(row["num_containers"]), f'{float(row["leveling_mean_rpn"]):.6f}']
        for horizon in HORIZONS:
            cells.append(
                _format_result(
                    float(row[f"h{horizon}_moe_mean_rpn"]),
                    float(row[f"h{horizon}_relative_change"]),
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Overall", ""])
    lines.append(
        f'- Leveling: mean R/N = {float(overall["leveling_mean_rpn"]):.6f}; '
        f'total relocations = {int(overall["leveling_total_relocations"]):,}.'
    )
    for horizon in HORIZONS:
        lines.append(
            f'- H={horizon}: MoE mean R/N = '
            f'{float(overall[f"h{horizon}_moe_mean_rpn"]):.6f} '
            f'({float(overall[f"h{horizon}_relative_change"]):+.2%}); '
            f'21 groups improved/tied/worse = '
            f'{int(overall[f"h{horizon}_improved_groups"])}/'
            f'{int(overall[f"h{horizon}_tied_groups"])}/'
            f'{int(overall[f"h{horizon}_worse_groups"])}.'
        )
    lines.extend(
        [
            "",
            "Leveling was recomputed at H=0,5,10,20. All 840 complete action sequences and relocation counts were identical across horizons.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    episodes = load_episode_manifest(args.manifest)
    report = build_comparison(episodes, _read_jsonl(args.results))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "group_comparison.csv", report["groups"])
    _write_csv(args.output_dir / "group_seed_detail.csv", report["seed_detail"])
    _write_markdown(args.output_dir / "group_comparison.md", report)
    (args.output_dir / "group_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
