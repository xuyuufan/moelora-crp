"""Paired RL-versus-leveling evaluation for Zehendner et al. OCRP_0."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from scrp.online_crp import (
    ONLINE_MMAX,
    PAPER_TABLE1_LAYOUTS,
    PAPER_TABLE2_GROUPS,
    LevelingBaseline,
    OnlineCRPEpisode,
    OnlineCRPEnvironment,
    OnlineOCRPObservationAdapter,
    generate_table1_sample,
    generate_table2_structural_sample,
    load_episode_manifest,
)


@dataclass(frozen=True)
class EpisodeEvaluation:
    episode_id: str
    canonical_id: str
    scale_id: str
    algorithm: str
    relocations: int
    runtime_seconds: float


DestinationSelector = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, int], int
]


def physical_scale_id(episode: OnlineCRPEpisode) -> str:
    initial_height = max(len(stack) for stack in episode.initial_stacks)
    return f"S{initial_height}-W{episode.num_stacks}"


def leveling_selector(
    observation: np.ndarray,
    forbidden: np.ndarray,
    node_padding_mask: np.ndarray,
    num_stacks: int,
    max_tiers: int,
) -> int:
    del node_padding_mask, max_tiers
    nodes = observation.reshape(-1, 12)
    legal = tuple(int(index) for index in np.flatnonzero(~forbidden))
    if not legal:
        raise ValueError("leveling baseline requires a legal destination")
    # Stack feature 2 is normalized height; stable stack index is the tie break.
    return min(legal, key=lambda stack_id: (float(nodes[stack_id, 2]), stack_id))


class TorchCheckpointSelector:
    """Greedy runner for a scale-diverse online 2D checkpoint."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        import torch

        from scrp.scale_diverse_policy import ScaleDiverseRelocationPolicy

        self._torch = torch
        self._device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "online_crp_training_config" not in checkpoint:
            raise ValueError("standalone online project accepts online-CRP checkpoints only")
        self._config = dict(checkpoint["online_crp_training_config"])
        self.policy_architecture = str(checkpoint.get("policy_architecture", ""))
        if self.policy_architecture != "scale_diverse_online_2d_v1":
            raise ValueError("only V1 relocation checkpoints are supported")
        self.lookahead_horizon = int(self._config.get("lookahead_horizon", 0))
        self.observation_kind = "priority_2d"
        self._policy = ScaleDiverseRelocationPolicy(
            lookahead_horizon=self.lookahead_horizon,
            embed_dim=int(self._value("embed_dim")),
            num_encoder_layers=int(self._value("num_encoder_layers")),
            num_heads=int(self._value("num_heads")),
            ffn_dim=int(self._value("ffn_dim")),
            clip_constant=float(self._value("clip_constant")),
        ).to(self._device)
        self._policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self._policy.eval()

    def _value(self, name: str):
        if isinstance(self._config, dict):
            return self._config[name]
        return getattr(self._config, name)

    def __call__(
        self,
        observation: np.ndarray,
        forbidden: np.ndarray,
        node_padding_mask: np.ndarray,
        num_stacks: int,
        max_tiers: int,
    ) -> int:
        torch = self._torch
        priorities = torch.as_tensor(
            observation.reshape(num_stacks, max_tiers),
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)
        forbidden_tensor = torch.as_tensor(
            forbidden, dtype=torch.bool, device=self._device
        ).unsqueeze(0)
        with torch.inference_mode():
            action, _ = self._policy(priorities, forbidden_tensor, greedy=True)
        selected = int(action.item())
        legal = tuple(int(index) for index in np.flatnonzero(~forbidden))
        if selected not in legal:
            raise AssertionError(f"RL selected illegal destination {selected}")
        return selected


def run_episode(
    episode: OnlineCRPEpisode,
    algorithm: str,
    selector: DestinationSelector,
) -> EpisodeEvaluation:
    lookahead_horizon = int(getattr(selector, "lookahead_horizon", 0))
    observation_kind = str(getattr(selector, "observation_kind", "legacy_o2"))
    env = OnlineCRPEnvironment(episode, lookahead_horizon=lookahead_horizon)
    if observation_kind == "priority_2d":
        from scrp.scale_diverse_policy import Online2DPriorityAdapter

        adapter = Online2DPriorityAdapter(episode.max_tiers)
    else:
        adapter = OnlineOCRPObservationAdapter(episode)
    started = time.perf_counter()
    while not env.state.terminated:
        legal = env.legal_destinations()
        if not legal:
            raise RuntimeError("non-terminal OCRP episode has no legal destination")
        observation = adapter.build(env.state)
        forbidden = np.ones(episode.num_stacks, dtype=np.bool_)
        forbidden[list(legal)] = False
        padding = (
            np.zeros(episode.num_stacks, dtype=np.bool_)
            if observation_kind == "priority_2d"
            else adapter.node_padding_mask(env.state)
        )
        env.step(selector(
            observation, forbidden, padding, episode.num_stacks, episode.max_tiers
        ))
    elapsed = time.perf_counter() - started
    return EpisodeEvaluation(
        episode_id=episode.episode_id,
        canonical_id=episode.canonical_id,
        scale_id=physical_scale_id(episode),
        algorithm=algorithm,
        relocations=env.relocation_count,
        runtime_seconds=elapsed,
    )


def evaluate_paired(
    episodes: Sequence[OnlineCRPEpisode], rl_selector: DestinationSelector
) -> tuple[EpisodeEvaluation, ...]:
    if not episodes:
        raise ValueError("paired evaluation requires at least one episode")
    results: list[EpisodeEvaluation] = []
    for episode in episodes:
        rl = run_episode(episode, "RL", rl_selector)
        baseline = run_episode(episode, LevelingBaseline.name, leveling_selector)
        if rl.canonical_id != baseline.canonical_id:
            raise AssertionError("RL and baseline did not use the same episode")
        results.extend((rl, baseline))
    return tuple(results)


def _sample_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def paired_bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260902,
) -> dict[str, float | int]:
    """Return a deterministic percentile CI for a paired mean difference."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired bootstrap requires at least one difference")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap samples or confidence")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1_000):
        end = min(start + 1_000, samples)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
        "lower": float(lower),
        "upper": float(upper),
    }


def _summary_for_pairs(
    pairs: Sequence[tuple[EpisodeEvaluation, EpisodeEvaluation]]
) -> dict[str, object]:
    rl_values = [pair[0].relocations for pair in pairs]
    baseline_values = [pair[1].relocations for pair in pairs]
    deltas = [rl - baseline for rl, baseline in zip(rl_values, baseline_values)]
    rl_mean = float(statistics.fmean(rl_values))
    baseline_mean = float(statistics.fmean(baseline_values))
    mean_difference = rl_mean - baseline_mean
    return {
        "episode_count": len(pairs),
        "RL": {
            "mean_relocations": rl_mean,
            "sample_std_relocations": _sample_std(rl_values),
            "total_relocations": sum(rl_values),
            "mean_runtime_seconds": float(statistics.fmean(pair[0].runtime_seconds for pair in pairs)),
        },
        LevelingBaseline.name: {
            "mean_relocations": baseline_mean,
            "sample_std_relocations": _sample_std(baseline_values),
            "total_relocations": sum(baseline_values),
            "mean_runtime_seconds": float(statistics.fmean(pair[1].runtime_seconds for pair in pairs)),
        },
        "RL_minus_baseline": {
            "mean_difference": mean_difference,
            "absolute_mean_difference": abs(mean_difference),
            "percentage_difference": (
                100.0 * mean_difference / baseline_mean if baseline_mean else None
            ),
            "sample_std_of_paired_differences": _sample_std(deltas),
            "paired_bootstrap_mean_difference_ci": paired_bootstrap_mean_ci(
                deltas
            ),
        },
        "win_tie_loss": {
            "RL_wins": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "RL_losses": sum(delta > 0 for delta in deltas),
        },
    }


def summarize(results: Sequence[EpisodeEvaluation]) -> dict[str, object]:
    grouped: dict[str, dict[str, EpisodeEvaluation]] = {}
    for result in results:
        grouped.setdefault(result.episode_id, {})[result.algorithm] = result
    pairs = []
    for episode_id, algorithms in grouped.items():
        if set(algorithms) != {"RL", LevelingBaseline.name}:
            raise AssertionError(f"incomplete algorithm pair for {episode_id}")
        pairs.append((algorithms["RL"], algorithms[LevelingBaseline.name]))
    pairs.sort(key=lambda pair: pair[0].episode_id)
    by_group: dict[str, list[tuple[EpisodeEvaluation, EpisodeEvaluation]]] = {}
    for pair in pairs:
        if pair[0].scale_id != pair[1].scale_id:
            raise AssertionError("paired physical scale mismatch")
        by_group.setdefault(pair[0].scale_id, []).append(pair)
    return {
        "protocol": "Zehendner2017_OCRP_0_restricted",
        "pairing_verified": True,
        "overall": _summary_for_pairs(pairs),
        "by_scale": {
            group: _summary_for_pairs(group_pairs)
            for group, group_pairs in sorted(by_group.items())
        },
    }


def _episodes_from_args(args: argparse.Namespace) -> tuple[OnlineCRPEpisode, ...]:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.episodes_json:
        episodes = load_episode_manifest(args.episodes_json)
        return episodes if args.limit is None else episodes[:args.limit]
    if args.protocol == "table1-sample":
        layouts = PAPER_TABLE1_LAYOUTS if args.group == "all" else (
            tuple(int(value) for value in args.group.split("-")),
        )
        return tuple(
            episode
            for layout in layouts
            for episode in generate_table1_sample(
                layout, episode_count=args.episodes_per_group, master_seed=args.seed
            )
        )
    groups = PAPER_TABLE2_GROUPS if args.group == "all" else (
        tuple(int(value) for value in args.group.split("-")),
    )
    return tuple(
        episode
        for group in groups
        for episode in generate_table2_structural_sample(
            group, episode_count=args.episodes_per_group, master_seed=args.seed
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes-json")
    parser.add_argument(
        "--protocol", choices=("table1-sample", "table2-structural"),
        default="table2-structural",
    )
    parser.add_argument("--group", default="3-3", help="W-S/layout tuple, or all")
    parser.add_argument("--episodes-per-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--limit", type=int, help="optional smoke-test prefix length")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--raw-output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes = _episodes_from_args(args)
    selector = TorchCheckpointSelector(args.checkpoint, device=args.device)
    results = evaluate_paired(episodes, selector)
    summary = summarize(results)
    if selector.lookahead_horizon > 0:
        summary["protocol"] = (
            f"online_restricted_relocation_H{selector.lookahead_horizon}"
        )
    summary["policy_architecture"] = selector.policy_architecture
    summary["lookahead_horizon"] = selector.lookahead_horizon
    summary["generation"] = {
        "master_seed": args.seed,
        "protocol": args.protocol if not args.episodes_json else "external_fixed_manifest",
        "exact_original_caserta_benchmark": all(
            episode.metadata.get("exact_original_benchmark") is True
            for episode in episodes
        ),
        "warning": (
            None if all(
                episode.metadata.get("exact_original_benchmark") is True
                for episode in episodes
            ) else
            "Engineering generator only; the paper's original Caserta files/seeds are unavailable."
        ),
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    if args.raw_output:
        destination = Path(args.raw_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in results),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
