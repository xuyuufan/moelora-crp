"""Evaluate final MoE-LoRA checkpoints on the sealed large-scale test set."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from scrp.adaptation import load_h20_base_policy
from scrp.adaptation_data import load_large_scale_split, manifest_sha256
from scrp.moe_lora import (
    configure_moe_lora_training,
    load_moe_lora_adapter,
    parameter_counts,
)
from scrp.online_crp import LevelingBaseline, OnlineCRPEpisode, OnlineCRPEnvironment
from scrp.scale_diverse_policy import Online2DPriorityAdapter


BASE_CHECKPOINT = Path("checkpoints/h20.best.pt")
SEALED_MANIFEST = Path(
    "data/online_crp/lora_domains/large_scale_n100_150/test_sealed.json"
)
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


@dataclass(frozen=True)
class EpisodeEvaluation:
    """One method's immutable result on one sealed episode."""

    episode_id: str
    canonical_id: str
    scale_id: str
    algorithm: str
    relocations: int
    runtime_seconds: float


DestinationSelector = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, int], int
]


class H20Selector:
    """Greedy base or MoE policy using only the public H=20 observation."""

    def __init__(
        self,
        base_checkpoint: str | Path,
        *,
        adapter_checkpoint: str | Path | None = None,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.policy, _, self.base_checkpoint_sha256 = load_h20_base_policy(
            base_checkpoint, device=self.device, expected_lookahead_horizon=20
        )
        self.lookahead_horizon = 20
        self.adapter_record: dict[str, object] | None = None
        if adapter_checkpoint is not None:
            payload = load_moe_lora_adapter(
                self.policy,
                adapter_checkpoint,
                expected_base_checkpoint_sha256=self.base_checkpoint_sha256,
                map_location="cpu",
            )
            configure_moe_lora_training(self.policy)
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError("adapter metadata must be a mapping")
            run_config = metadata.get("adaptation_config", {})
            if not isinstance(run_config, Mapping):
                raise ValueError("adapter adaptation_config must be a mapping")
            # The checkpoint retains full provenance.  The evaluation summary
            # records only fields needed to identify the selected run, instead
            # of duplicating hundreds of validation episode scores.
            self.adapter_record = {
                "seed": run_config.get("seed"),
                "completed_epoch": metadata.get("completed_epoch"),
                "selection_metric": metadata.get("selection_metric"),
                "base_checkpoint_sha256": payload.get("base_checkpoint_sha256"),
                "moe_lora_config": payload.get("moe_lora_config"),
                "inference_mode": "dynamic_moe_lora",
            }
        self.policy.eval()

    def __call__(
        self,
        observation: np.ndarray,
        forbidden: np.ndarray,
        node_padding_mask: np.ndarray,
        num_stacks: int,
        max_tiers: int,
    ) -> int:
        del node_padding_mask
        priorities = torch.as_tensor(
            observation.reshape(num_stacks, max_tiers),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        forbidden_tensor = torch.as_tensor(
            forbidden, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        # Final scoring is explicitly outside autocast even when training used
        # FP16; this is part of the published evaluation protocol.
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=False
        ):
            action, _ = self.policy(priorities, forbidden_tensor, greedy=True)
        selected = int(action.item())
        if bool(forbidden[selected]):
            raise AssertionError("policy selected a forbidden destination")
        return selected


def _leveling_selector(
    observation: np.ndarray,
    forbidden: np.ndarray,
    node_padding_mask: np.ndarray,
    num_stacks: int,
    max_tiers: int,
) -> int:
    del node_padding_mask
    priorities = observation.reshape(num_stacks, max_tiers)
    legal = tuple(int(index) for index in np.flatnonzero(~forbidden))
    if not legal:
        raise ValueError("leveling requires a legal destination")
    # Empty tiers are zero; both revealed priorities and -1 placeholders are
    # occupied containers.
    return min(
        legal,
        key=lambda stack_id: (
            int(np.count_nonzero(priorities[stack_id])),
            stack_id,
        ),
    )


def _scale_id(episode: OnlineCRPEpisode) -> str:
    return f"S{max(len(stack) for stack in episode.initial_stacks)}-W{episode.num_stacks}"


def _run_episode(
    episode: OnlineCRPEpisode,
    algorithm: str,
    selector: DestinationSelector,
) -> EpisodeEvaluation:
    horizon = int(getattr(selector, "lookahead_horizon", 0))
    environment = OnlineCRPEnvironment(episode, lookahead_horizon=horizon)
    adapter = Online2DPriorityAdapter(episode.max_tiers)
    started = time.perf_counter()
    while not environment.state.terminated:
        legal = environment.legal_destinations()
        if not legal:
            raise RuntimeError("non-terminal OCRP episode has no legal destination")
        observation = adapter.build(environment.state)
        forbidden = np.ones(episode.num_stacks, dtype=np.bool_)
        forbidden[list(legal)] = False
        padding = np.zeros(episode.num_stacks, dtype=np.bool_)
        destination = selector(
            observation,
            forbidden,
            padding,
            episode.num_stacks,
            episode.max_tiers,
        )
        environment.step(destination)
    return EpisodeEvaluation(
        episode_id=episode.episode_id,
        canonical_id=episode.canonical_id,
        scale_id=_scale_id(episode),
        algorithm=algorithm,
        relocations=environment.relocation_count,
        runtime_seconds=time.perf_counter() - started,
    )


def _bootstrap_ci(
    differences: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260902,
) -> dict[str, float | int]:
    """Deterministic paired percentile interval over fixed test instances."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired bootstrap requires differences")
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


def _method_summary(
    results: Sequence[EpisodeEvaluation],
    episodes: Sequence[OnlineCRPEpisode],
) -> dict[str, object]:
    by_id = {result.canonical_id: result for result in results}
    relocations = [by_id[episode.canonical_id].relocations for episode in episodes]
    r_per_n = [
        value / episode.num_containers
        for value, episode in zip(relocations, episodes, strict=True)
    ]
    runtimes = [by_id[episode.canonical_id].runtime_seconds for episode in episodes]
    return {
        "episodes": len(episodes),
        "mean_relocations": float(statistics.fmean(relocations)),
        "sample_std_relocations": float(statistics.stdev(relocations))
        if len(relocations) > 1
        else 0.0,
        "mean_relocations_per_container": float(statistics.fmean(r_per_n)),
        "mean_episode_runtime_seconds": float(statistics.fmean(runtimes)),
        "total_relocations": int(sum(relocations)),
    }


def _paired_comparison(
    candidate: Sequence[EpisodeEvaluation],
    baseline: Sequence[EpisodeEvaluation],
) -> dict[str, object]:
    candidate_by_id = {result.canonical_id: result for result in candidate}
    baseline_by_id = {result.canonical_id: result for result in baseline}
    if candidate_by_id.keys() != baseline_by_id.keys():
        raise AssertionError("paired methods used different test instances")
    deltas = [
        candidate_by_id[key].relocations - baseline_by_id[key].relocations
        for key in sorted(candidate_by_id)
    ]
    return {
        "candidate_minus_baseline_mean_relocations": float(
            statistics.fmean(deltas)
        ),
        "paired_bootstrap_mean_difference_ci": _bootstrap_ci(deltas),
        "win_tie_loss_vs_baseline": {
            "wins": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta > 0 for delta in deltas),
        },
    }


def evaluate(
    episodes: Sequence[OnlineCRPEpisode],
    *,
    base_checkpoint: str | Path,
    named_adapters: Sequence[tuple[str, str | Path]],
    device: str,
) -> tuple[dict[str, object], tuple[EpisodeEvaluation, ...]]:
    """Evaluate base, all adapters, and public-information Leveling in pairs."""

    selectors: dict[str, DestinationSelector] = {
        "MijnRL_base": H20Selector(base_checkpoint, device=device)
    }
    adapter_selectors: dict[str, H20Selector] = {}
    for name, checkpoint in named_adapters:
        if name in selectors or name == LevelingBaseline.name:
            raise ValueError(f"duplicate or reserved method name {name!r}")
        selector = H20Selector(
            base_checkpoint,
            adapter_checkpoint=checkpoint,
            device=device,
        )
        selectors[name] = selector
        adapter_selectors[name] = selector
    per_method: dict[str, list[EpisodeEvaluation]] = {
        name: [] for name in (*selectors, LevelingBaseline.name)
    }
    for episode in episodes:
        for name, selector in selectors.items():
            per_method[name].append(_run_episode(episode, name, selector))
        per_method[LevelingBaseline.name].append(
            _run_episode(episode, LevelingBaseline.name, _leveling_selector)
        )
    base_results = per_method["MijnRL_base"]
    summary: dict[str, object] = {
        "lookahead_horizon": 20,
        "online_information": "current target plus 20 revealed future targets only",
        "leveling_information": "public state and legal actions only; no future target order",
        "methods": {
            name: _method_summary(results, episodes)
            for name, results in per_method.items()
        },
        "comparisons_vs_base": {
            name: _paired_comparison(results, base_results)
            for name, results in per_method.items()
            if name != "MijnRL_base"
        },
        "adaptation_metadata": {
            name: {
                "format": "mijnrl_moe_lora_adapter_v1",
                "metadata": selector.adapter_record,
                "inference_parameter_counts": parameter_counts(selector.policy),
            }
            for name, selector in adapter_selectors.items()
        },
    }
    raw = tuple(result for results in per_method.values() for result in results)
    return summary, raw


def _parse_adapter(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("adapter must be NAME=PATH")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-checkpoint", default=str(BASE_CHECKPOINT))
    parser.add_argument("--test-manifest", default=str(SEALED_MANIFEST))
    parser.add_argument(
        "--adapter",
        type=_parse_adapter,
        action="append",
        help="NAME=PATH; omit to evaluate the three released adapters",
    )
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="evaluate four sealed instances to verify the pipeline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    raw_path = output_dir / "episode_results.jsonl"
    if summary_path.exists() or raw_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_large_scale_split(args.test_manifest, "test_sealed")
    if args.quick_check:
        episodes = episodes[:4]
    named_adapters = args.adapter or list(FINAL_ADAPTERS)
    summary, raw = evaluate(
        episodes,
        base_checkpoint=args.base_checkpoint,
        named_adapters=named_adapters,
        device=args.device,
    )
    summary.update(
        {
            "manifest": str(args.test_manifest),
            "manifest_sha256": manifest_sha256(args.test_manifest),
            "domain": "large_scale_n100_150",
            "split": "test_sealed",
            "test_manifest_used_for_selection": False,
            "quick_check": bool(args.quick_check),
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raw_path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in raw),
        encoding="utf-8",
    )
    print(f"wrote {len(raw)} episode records and summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
