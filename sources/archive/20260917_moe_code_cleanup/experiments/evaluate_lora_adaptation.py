"""Paired H=20 evaluation of the frozen base and adapted MijnRL policies."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from experiments.online_crp_fair_evaluation import (
    EpisodeEvaluation,
    paired_bootstrap_mean_ci,
    physical_scale_id,
    run_episode,
    leveling_selector,
)
from scrp.adaptation import load_adapter, load_h20_base_policy
from scrp.adaptation_data import TARGET_DOMAINS, load_and_validate_target_domain, manifest_sha256
from scrp.lora import LORA_ADAPTER_FORMAT, configure_trainable_parameters, merge_lora_, parameter_counts
from scrp.moe_lora import MOE_LORA_ADAPTER_FORMAT, configure_moe_lora_training
from scrp.online_crp import LevelingBaseline, OnlineCRPEpisode
from scrp.scale_diverse_policy import Online2DPriorityAdapter


class H20AdaptationSelector:
    """Greedy policy selector that accepts only the public H=20 observation."""

    observation_kind = "priority_2d"

    def __init__(
        self,
        base_checkpoint: str | Path,
        *,
        adaptation_checkpoint: str | Path | None = None,
        device: str = "cpu",
        merge_lora_for_inference: bool = False,
    ) -> None:
        self._device = torch.device(device)
        self._policy, _, self.base_checkpoint_sha256 = load_h20_base_policy(
            base_checkpoint, device=self._device, expected_lookahead_horizon=20
        )
        self.lookahead_horizon = 20
        self.adaptation_metadata: dict[str, object] | None = None
        self.adaptation_format: str | None = None
        if adaptation_checkpoint is not None:
            payload = load_adapter(
                self._policy,
                adaptation_checkpoint,
                expected_base_checkpoint_sha256=self.base_checkpoint_sha256,
                map_location="cpu",
            )
            self.adaptation_metadata = dict(payload.get("metadata", {}))
            self.adaptation_format = str(payload.get("format"))
            if self.adaptation_format == LORA_ADAPTER_FORMAT:
                configure_trainable_parameters(self._policy)
            elif self.adaptation_format == MOE_LORA_ADAPTER_FORMAT:
                configure_moe_lora_training(self._policy)
                self.adaptation_metadata["inference_mode"] = "dynamic_moe_lora"
                if merge_lora_for_inference:
                    raise ValueError(
                        "MoE-LoRA cannot be merged: routing depends on each stack input"
                    )
            else:  # load_adapter already rejects this; retain a defensive guard.
                raise ValueError(f"unsupported adapter format {self.adaptation_format!r}")
            if merge_lora_for_inference and self.adaptation_format == LORA_ADAPTER_FORMAT:
                merge_lora_(self._policy)
        self._policy.eval()

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
            device=self._device,
        ).unsqueeze(0)
        forbidden_tensor = torch.as_tensor(
            forbidden, dtype=torch.bool, device=self._device
        ).unsqueeze(0)
        with torch.inference_mode():
            selected, _ = self._policy(priorities, forbidden_tensor, greedy=True)
        action = int(selected.item())
        if bool(forbidden[action]):
            raise AssertionError("policy selected a forbidden destination")
        return action


def _parse_named_checkpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adaptation must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("adaptation must be NAME=PATH")
    return name, path


def _method_summary(
    results: Sequence[EpisodeEvaluation], episodes: Sequence[OnlineCRPEpisode]
) -> dict[str, object]:
    by_id = {result.canonical_id: result for result in results}
    if len(by_id) != len(episodes):
        raise AssertionError("method did not produce exactly one result per episode")
    relocations = [by_id[episode.canonical_id].relocations for episode in episodes]
    r_per_n = [value / episode.num_containers for value, episode in zip(relocations, episodes, strict=True)]
    runtimes = [by_id[episode.canonical_id].runtime_seconds for episode in episodes]
    return {
        "mean_relocations": float(statistics.fmean(relocations)),
        "sample_std_relocations": float(statistics.stdev(relocations)) if len(relocations) > 1 else 0.0,
        "mean_relocations_per_container": float(statistics.fmean(r_per_n)),
        "mean_episode_runtime_seconds": float(statistics.fmean(runtimes)),
        "total_relocations": int(sum(relocations)),
    }


def _paired_comparison(
    candidate: Sequence[EpisodeEvaluation], baseline: Sequence[EpisodeEvaluation]
) -> dict[str, object]:
    candidate_by_id = {result.canonical_id: result for result in candidate}
    baseline_by_id = {result.canonical_id: result for result in baseline}
    if candidate_by_id.keys() != baseline_by_id.keys():
        raise AssertionError("paired methods were evaluated on different instances")
    deltas = [
        candidate_by_id[key].relocations - baseline_by_id[key].relocations
        for key in sorted(candidate_by_id)
    ]
    return {
        "candidate_minus_baseline_mean_relocations": float(statistics.fmean(deltas)),
        "paired_bootstrap_mean_difference_ci": paired_bootstrap_mean_ci(deltas),
        "win_tie_loss_vs_baseline": {
            "wins": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta > 0 for delta in deltas),
        },
    }


def evaluate_adaptations(
    episodes: Sequence[OnlineCRPEpisode],
    *,
    base_checkpoint: str | Path,
    named_adaptations: Sequence[tuple[str, str | Path]],
    device: str,
    merge_lora_for_inference: bool = False,
) -> tuple[dict[str, object], tuple[EpisodeEvaluation, ...]]:
    """Evaluate base, requested adaptations, and H-agnostic Leveling fairly."""

    if not episodes:
        raise ValueError("evaluation manifest must not be empty")
    selectors: dict[str, object] = {
        "MijnRL_base": H20AdaptationSelector(base_checkpoint, device=device)
    }
    for name, checkpoint in named_adaptations:
        if name in selectors or name == LevelingBaseline.name:
            raise ValueError(f"duplicate or reserved method name {name!r}")
        selectors[name] = H20AdaptationSelector(
            base_checkpoint,
            adaptation_checkpoint=checkpoint,
            device=device,
            merge_lora_for_inference=merge_lora_for_inference,
        )
    per_method: dict[str, list[EpisodeEvaluation]] = {name: [] for name in selectors}
    per_method[LevelingBaseline.name] = []
    for episode in episodes:
        for name, selector in selectors.items():
            per_method[name].append(run_episode(episode, name, selector))
        per_method[LevelingBaseline.name].append(
            run_episode(episode, LevelingBaseline.name, leveling_selector)
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
                "format": selector.adaptation_format,
                "metadata": selector.adaptation_metadata,
                "inference_parameter_counts": parameter_counts(selector._policy),
            }
            for name, selector in selectors.items()
            if isinstance(selector, H20AdaptationSelector) and name != "MijnRL_base"
        },
    }
    raw = tuple(result for results in per_method.values() for result in results)
    return summary, raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--episodes-json", required=True)
    parser.add_argument("--domain", choices=tuple(TARGET_DOMAINS), required=True)
    parser.add_argument(
        "--split", choices=("validation", "test_sealed"), default="test_sealed"
    )
    parser.add_argument("--adaptation", type=_parse_named_checkpoint, action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge-lora-for-inference", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes = load_and_validate_target_domain(
        args.episodes_json, domain=args.domain, allowed_splits=(args.split,)
    )
    summary, raw = evaluate_adaptations(
        episodes,
        base_checkpoint=args.base_checkpoint,
        named_adaptations=args.adaptation,
        device=args.device,
        merge_lora_for_inference=args.merge_lora_for_inference,
    )
    summary.update(
        {
            "manifest": str(args.episodes_json),
            "manifest_sha256": manifest_sha256(args.episodes_json),
            "domain": args.domain,
            "split": args.split,
            "test_manifest_used_for_selection": False,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.raw_output:
        raw_output = Path(args.raw_output)
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text(
            "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in raw),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
