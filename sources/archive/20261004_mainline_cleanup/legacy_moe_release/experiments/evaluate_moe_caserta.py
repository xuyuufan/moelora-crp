"""Evaluate the frozen base and three final MoE adapters on Caserta Table 2.

This is an external public-benchmark evaluation, not a sealed test and not a
model-selection stage.  The four evaluation horizons change only which future
targets are revealed by the environment; every model remains the H=20-trained
policy and no parameters are updated.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from scrp.adaptation import load_h20_base_policy
from scrp.adaptation_data import manifest_sha256
from scrp.moe_lora import (
    configure_moe_lora_training,
    load_moe_lora_adapter,
    parameter_counts,
)
from scrp.moe_training import append_jsonl, evaluate_fixed_validation
from scrp.online_crp import OnlineCRPEpisode, load_episode_manifest


BASE_CHECKPOINT = Path("checkpoints/h20.best.pt")
CASERTA_MANIFEST = Path("data/online_crp/caserta_table2_840.json")
HORIZONS = (0, 5, 10, 20)
FINAL_ADAPTERS = (
    (
        "MoE_seed20260913",
        Path("output/moe_lora_long_20260917_seed20260913/best.adapter.pt"),
    ),
    (
        "MoE_seed20260923",
        Path("output/moe_lora_long_20260917_seed20260923/best.adapter.pt"),
    ),
    (
        "MoE_seed20260933",
        Path("output/moe_lora_long_20260917_seed20260933/best.adapter.pt"),
    ),
)

# These totals were produced by the original H=20 base evaluation on the exact
# same 840-instance manifest.  Matching them guards against environment or
# observation drift before any new MoE result is interpreted.
HISTORICAL_BASE_TOTALS = {0: 41_277, 5: 30_907, 10: 26_639, 20: 23_993}


def _validate_caserta(episodes: Sequence[OnlineCRPEpisode]) -> None:
    """Check that the input is the exact 21x40 original benchmark structure."""

    if len(episodes) != 840:
        raise ValueError(f"Caserta Table 2 requires 840 episodes, got {len(episodes)}")
    groups: dict[str, int] = defaultdict(int)
    canonical_ids: set[str] = set()
    for episode in episodes:
        if episode.metadata.get("protocol") != "paper_table2_original_caserta_benchmark":
            raise ValueError("manifest is not the original Caserta Table 2 protocol")
        if episode.metadata.get("exact_original_benchmark") is not True:
            raise ValueError("manifest contains a non-original benchmark episode")
        groups[episode.parameter_group] += 1
        canonical_ids.add(episode.canonical_id)
    if len(groups) != 21 or set(groups.values()) != {40}:
        raise ValueError("Caserta Table 2 must contain 21 groups of 40 episodes")
    if len(canonical_ids) != len(episodes):
        raise ValueError("Caserta manifest contains duplicate canonical episodes")


def _load_policy(
    *,
    base_checkpoint: str | Path,
    adapter_checkpoint: str | Path | None,
    device: str,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Load one immutable inference policy and its compact provenance record."""

    policy, _, base_hash = load_h20_base_policy(
        base_checkpoint, device=device, expected_lookahead_horizon=20
    )
    record: dict[str, object] = {
        "base_checkpoint_sha256": base_hash,
        "training_lookahead_horizon": 20,
        "inference": "greedy_fp32",
    }
    if adapter_checkpoint is not None:
        payload = load_moe_lora_adapter(
            policy,
            adapter_checkpoint,
            expected_base_checkpoint_sha256=base_hash,
            map_location="cpu",
        )
        configure_moe_lora_training(policy)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("adapter metadata must be a mapping")
        config = metadata.get("adaptation_config", {})
        if not isinstance(config, Mapping):
            raise ValueError("adapter adaptation_config must be a mapping")
        record.update(
            {
                "adapter_checkpoint": str(adapter_checkpoint),
                "adapter_sha256": manifest_sha256(adapter_checkpoint),
                "seed": config.get("seed"),
                "selected_epoch": metadata.get("completed_epoch"),
                "moe_lora_config": payload.get("moe_lora_config"),
            }
        )
    else:
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
    policy.eval()
    record["parameter_counts"] = parameter_counts(policy)
    return policy, record


def _subset_summary(
    episodes: Sequence[OnlineCRPEpisode],
    relocations: Sequence[int],
    indices: Sequence[int],
) -> dict[str, float | int]:
    if not indices:
        return {"episodes": 0}
    values = [relocations[index] for index in indices]
    ratios = [
        relocations[index] / episodes[index].num_containers for index in indices
    ]
    return {
        "episodes": len(indices),
        "mean_relocations": float(statistics.fmean(values)),
        "mean_relocations_per_container": float(statistics.fmean(ratios)),
        "total_relocations": int(sum(values)),
    }


def _summarize(
    episodes: Sequence[OnlineCRPEpisode],
    relocations: Sequence[int],
    *,
    horizon: int,
    runtime_seconds: float,
) -> dict[str, object]:
    """Summarize overall, benchmark-group, and information-regime results."""

    all_indices = list(range(len(episodes)))
    small = [i for i, episode in enumerate(episodes) if episode.num_containers <= 21]
    large = [i for i, episode in enumerate(episodes) if episode.num_containers > 21]
    fully_visible = [
        i for i, episode in enumerate(episodes)
        if episode.num_containers <= horizon + 1
    ]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        grouped[episode.parameter_group].append(index)
    group_summaries = {
        group: _subset_summary(episodes, relocations, indices)
        for group, indices in sorted(grouped.items())
    }
    result: dict[str, object] = _subset_summary(
        episodes, relocations, all_indices
    )
    result.update(
        {
            "lookahead_horizon": horizon,
            "visible_target_count": horizon + 1,
            "runtime_seconds": runtime_seconds,
            "sample_std_relocations": float(statistics.stdev(relocations)),
            # Every Caserta group has 40 instances, but record the explicit
            # group-macro value so the aggregation rule remains auditable.
            "macro_mean_relocations": float(
                statistics.fmean(
                    entry["mean_relocations"] for entry in group_summaries.values()
                )
            ),
            "by_fixed_size_regime": {
                "N_le_21": _subset_summary(episodes, relocations, small),
                "N_gt_21": _subset_summary(episodes, relocations, large),
            },
            "fully_visible_from_initial_state": _subset_summary(
                episodes, relocations, fully_visible
            ),
            "by_caserta_group": group_summaries,
        }
    )
    return result


def _bootstrap_mean_ci(
    differences: np.ndarray, *, seed: int = 20260918, samples: int = 10_000
) -> dict[str, float | int]:
    """Return a deterministic paired 95% percentile interval."""

    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1_000):
        end = min(start + 1_000, samples)
        indices = rng.integers(0, len(differences), (end - start, len(differences)))
        means[start:end] = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "confidence": 0.95,
        "samples": samples,
        "seed": seed,
        "lower": float(lower),
        "upper": float(upper),
    }


def _compare_with_base(
    episodes: Sequence[OnlineCRPEpisode],
    candidate: Sequence[int],
    base: Sequence[int],
) -> dict[str, object]:
    deltas = np.asarray(candidate, dtype=np.int64) - np.asarray(base, dtype=np.int64)
    ratio_deltas = np.asarray(
        [delta / episode.num_containers for delta, episode in zip(deltas, episodes, strict=True)],
        dtype=np.float64,
    )
    return {
        "candidate_minus_base_mean_relocations": float(deltas.mean()),
        "candidate_minus_base_mean_relocations_per_container": float(
            ratio_deltas.mean()
        ),
        "paired_mean_relocation_difference_ci": _bootstrap_mean_ci(deltas),
        "win_tie_loss_vs_base": {
            "wins": int((deltas < 0).sum()),
            "ties": int((deltas == 0).sum()),
            "losses": int((deltas > 0).sum()),
        },
    }


def evaluate(
    *,
    manifest: str | Path,
    base_checkpoint: str | Path,
    output_dir: str | Path,
    device: str,
    quick_check: bool,
) -> Path:
    """Run all frozen models at H=0/5/10/20 and write auditable results."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Caserta run: {destination}")
    destination.mkdir(parents=True)
    progress_path = destination / "progress.jsonl"

    episodes = load_episode_manifest(manifest)
    _validate_caserta(episodes)
    if quick_check:
        episodes = episodes[:16]

    named_checkpoints: tuple[tuple[str, Path | None], ...] = (
        ("MijnRL_base", None),
        *FINAL_ADAPTERS,
    )
    costs: dict[tuple[str, int], list[int]] = {}
    methods: dict[str, dict[str, object]] = {}
    for method, checkpoint in named_checkpoints:
        policy, provenance = _load_policy(
            base_checkpoint=base_checkpoint,
            adapter_checkpoint=checkpoint,
            device=device,
        )
        methods[method] = {"provenance": provenance, "horizons": {}}
        for horizon in HORIZONS:
            started = time.perf_counter()
            evaluation = evaluate_fixed_validation(
                policy,
                episodes,
                lookahead_horizon=horizon,
                device=device,
            )
            by_id = evaluation["validation_episode_relocations"]
            if not isinstance(by_id, Mapping):
                raise AssertionError("evaluation did not return per-episode costs")
            relocation_values = [int(by_id[episode.canonical_id]) for episode in episodes]
            costs[(method, horizon)] = relocation_values
            horizon_summary = _summarize(
                episodes,
                relocation_values,
                horizon=horizon,
                runtime_seconds=time.perf_counter() - started,
            )
            methods[method]["horizons"][str(horizon)] = horizon_summary  # type: ignore[index]
            append_jsonl(
                progress_path,
                {
                    "method": method,
                    "lookahead_horizon": horizon,
                    "episodes": len(episodes),
                    "total_relocations": sum(relocation_values),
                    "mean_relocations_per_container": horizon_summary[
                        "mean_relocations_per_container"
                    ],
                },
            )
            print(
                json.dumps(
                    {
                        "method": method,
                        "H": horizon,
                        "mean_R_per_N": horizon_summary[
                            "mean_relocations_per_container"
                        ],
                        "total_R": sum(relocation_values),
                    }
                ),
                flush=True,
            )
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not quick_check:
        for horizon, expected_total in HISTORICAL_BASE_TOTALS.items():
            actual = sum(costs[("MijnRL_base", horizon)])
            if actual != expected_total:
                raise AssertionError(
                    f"base H={horizon} drift: expected {expected_total}, got {actual}"
                )

    comparisons = {
        method: {
            str(horizon): _compare_with_base(
                episodes,
                costs[(method, horizon)],
                costs[("MijnRL_base", horizon)],
            )
            for horizon in HORIZONS
        }
        for method, _ in FINAL_ADAPTERS
    }
    summary = {
        "protocol": "final_moe_on_exact_original_caserta_table2_840_v1",
        "benchmark_role": "external_public_benchmark_not_used_for_moe_selection",
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256(manifest),
        "episode_count": len(episodes),
        "evaluation_horizons": list(HORIZONS),
        "horizon_definition": "H future targets plus the current target",
        "models_are_frozen": True,
        "quick_check": quick_check,
        "methods": methods,
        "comparisons_vs_base": comparisons,
        "historical_base_totals_matched": not quick_check,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (destination / "episode_results.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for method, _ in named_checkpoints:
            for horizon in HORIZONS:
                for episode, relocations in zip(
                    episodes, costs[(method, horizon)], strict=True
                ):
                    handle.write(
                        json.dumps(
                            {
                                "algorithm": method,
                                "lookahead_horizon": horizon,
                                "episode_id": episode.episode_id,
                                "canonical_id": episode.canonical_id,
                                "parameter_group": episode.parameter_group,
                                "num_containers": episode.num_containers,
                                "relocations": relocations,
                                "relocations_per_container": (
                                    relocations / episode.num_containers
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(CASERTA_MANIFEST))
    parser.add_argument("--base-checkpoint", default=str(BASE_CHECKPOINT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="run the first 16 episodes; skips historical-total assertions",
    )
    args = parser.parse_args(argv)
    destination = evaluate(
        manifest=args.manifest,
        base_checkpoint=args.base_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        quick_check=args.quick_check,
    )
    print(f"Caserta evaluation complete: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
