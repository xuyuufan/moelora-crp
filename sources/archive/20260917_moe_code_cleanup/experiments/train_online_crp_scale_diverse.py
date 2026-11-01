"""Train and evaluate the V1 relocation-only OCRP policy.

The sampling, replay, precision and validation helpers are shared with generic
LoRA and MoE-LoRA adaptation. Training uses public observations and fresh structural instances.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import warnings
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from scrp.online_crp import (
    PAPER_TABLE2_GROUPS,
    OnlineCRPEpisode,
    OnlineCRPEnvironment,
    generate_table2_structural_sample,
    load_episode_manifest,
)
from scrp.scale_diverse_policy import (
    SCALE_DIVERSE_POLICY_V1,
    Online2DPriorityAdapter,
    ScaleDiverseRelocationPolicy,
)
from scrp.vectorized_online_crp import VectorizedOnlineCRPEnvironment


LOSS_WEIGHTING_LEGACY = "per_decision_sum_log_probability"
LOSS_WEIGHTING_EQUAL_TRAJECTORY = "equal_trajectory_mean_log_probability"


@dataclass(frozen=True)
class ScaleDiverseOnlineTrainingConfig:
    policy_architecture: str = SCALE_DIVERSE_POLICY_V1
    observation_version: str = "online_2d_priority_v1"
    loss_weighting: str = LOSS_WEIGHTING_EQUAL_TRAJECTORY
    lookahead_horizon: int = 0
    cover_first_actions: bool = False
    embed_dim: int = 128
    num_encoder_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 512
    clip_constant: float = 5.0
    learning_rate: float = 3e-4
    learning_rate_schedule: str = "cosine"
    minimum_learning_rate_ratio: float = 0.1
    learning_rate_constant_epochs: int = 2
    epochs: int = 100
    steps_per_epoch: int = 105
    scales_per_step: int = 4
    instances_per_scale: int = 32
    rollouts_per_instance: int = 16
    replay_decision_batch_size: int = 2048
    entropy_coefficient: float = 0.002
    entropy_final_coefficient: float = 0.0002
    entropy_decay_start_epoch: int = 1
    entropy_decay_end_epoch: int = 30
    gradient_clip_norm: float = 1.0
    runtime_validation: bool = False
    precision_mode: str = "fp32"
    environment_engine: str = "vectorized"
    seed: int = 20260901
    validation_every: int = 1
    early_stopping_enabled: bool = False
    early_stop_min_epochs: int = 30
    early_stop_patience: int = 25
    early_stop_min_delta: float = 0.05

    def validate(self) -> None:
        positive = {
            "embed_dim": self.embed_dim,
            "num_encoder_layers": self.num_encoder_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "scales_per_step": self.scales_per_step,
            "instances_per_scale": self.instances_per_scale,
            "rollouts_per_instance": self.rollouts_per_instance,
            "replay_decision_batch_size": self.replay_decision_batch_size,
            "validation_every": self.validation_every,
            "early_stop_min_epochs": self.early_stop_min_epochs,
            "early_stop_patience": self.early_stop_patience,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"training values must be positive: {positive}")
        if self.lookahead_horizon < 0:
            raise ValueError("lookahead_horizon must be non-negative")
        if self.policy_architecture != SCALE_DIVERSE_POLICY_V1:
            raise ValueError("only the V1 relocation policy is supported")
        if self.scales_per_step > len(PAPER_TABLE2_GROUPS):
            raise ValueError("scales_per_step exceeds the available structural groups")
        if self.rollouts_per_instance < 2:
            raise ValueError("normalized-return training requires at least two rollouts")
        if self.loss_weighting not in {
            LOSS_WEIGHTING_LEGACY,
            LOSS_WEIGHTING_EQUAL_TRAJECTORY,
        }:
            raise ValueError("unsupported loss_weighting")
        if self.precision_mode not in {"fp32", "tf32", "bf16", "fp16"}:
            raise ValueError(
                "precision_mode must be one of fp32, tf32, bf16, or fp16"
            )
        if self.environment_engine not in {"scalar", "vectorized"}:
            raise ValueError("environment_engine must be scalar or vectorized")
        if self.learning_rate_schedule not in {"constant", "cosine"}:
            raise ValueError("learning_rate_schedule must be constant or cosine")
        bounded = {
            "minimum_learning_rate_ratio": self.minimum_learning_rate_ratio,
            "entropy_coefficient": self.entropy_coefficient,
            "entropy_final_coefficient": self.entropy_final_coefficient,
            "gradient_clip_norm": self.gradient_clip_norm,
            "early_stop_min_delta": self.early_stop_min_delta,
        }
        if any(value < 0 for value in bounded.values()):
            raise ValueError(f"training values must be non-negative: {bounded}")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("learning_rate and gradient_clip_norm must be positive")
        if not 0 < self.minimum_learning_rate_ratio <= 1:
            raise ValueError("minimum_learning_rate_ratio must be in (0, 1]")
        if self.learning_rate_constant_epochs < 0:
            raise ValueError("learning_rate_constant_epochs must be non-negative")
        if not 0 <= self.entropy_decay_start_epoch <= self.entropy_decay_end_epoch:
            raise ValueError("invalid entropy decay epoch range")


def physical_scale_id(episode: OnlineCRPEpisode) -> str:
    """Return an unambiguous physical ``S<height>-W<stacks>`` scale key."""

    initial_height = max(len(stack) for stack in episode.initial_stacks)
    return f"S{initial_height}-W{episode.num_stacks}"


def build_balanced_scale_schedule(
    config: ScaleDiverseOnlineTrainingConfig, epoch_index: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Build a deterministic, quota-balanced scale schedule for one epoch.

    Counts differ by at most one for arbitrary smoke-test configurations.  With
    the production 105 x 4 setup every one of the 21 scales occurs exactly 20
    times and no optimizer step contains a duplicate scale.
    """

    groups = tuple(PAPER_TABLE2_GROUPS)
    slot_count = config.steps_per_epoch * config.scales_per_step
    base, remainder = divmod(slot_count, len(groups))
    rng = random.Random(config.seed + epoch_index * 1_000_003 + 97_531)
    remainder_groups = set(rng.sample(groups, remainder)) if remainder else set()
    remaining = {
        group: base + int(group in remainder_groups)
        for group in groups
    }
    schedule: list[tuple[tuple[int, int], ...]] = []
    for _ in range(config.steps_per_epoch):
        candidates = [group for group, count in remaining.items() if count > 0]
        rng.shuffle(candidates)
        candidates.sort(key=lambda group: remaining[group], reverse=True)
        selected = tuple(candidates[: config.scales_per_step])
        if len(selected) != config.scales_per_step:
            raise RuntimeError("could not construct a distinct balanced scale batch")
        for group in selected:
            remaining[group] -= 1
        schedule.append(selected)
    if any(remaining.values()):
        raise RuntimeError("balanced scale scheduler left unassigned quota")
    return tuple(schedule)


def entropy_coefficient_for_epoch(
    config: ScaleDiverseOnlineTrainingConfig, epoch_number: int
) -> float:
    """Return the configured constant/cosine entropy coefficient."""

    if epoch_number <= config.entropy_decay_start_epoch:
        return config.entropy_coefficient
    if epoch_number >= config.entropy_decay_end_epoch:
        return config.entropy_final_coefficient
    span = config.entropy_decay_end_epoch - config.entropy_decay_start_epoch
    if span <= 0:
        return config.entropy_final_coefficient
    progress = (epoch_number - config.entropy_decay_start_epoch) / span
    blend = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.entropy_final_coefficient + (
        config.entropy_coefficient - config.entropy_final_coefficient
    ) * blend


def _learning_rate_multiplier(
    config: ScaleDiverseOnlineTrainingConfig, applied_update: int
) -> float:
    if config.learning_rate_schedule == "constant":
        return 1.0
    constant_updates = config.learning_rate_constant_epochs * config.steps_per_epoch
    total_updates = config.epochs * config.steps_per_epoch
    if applied_update <= constant_updates or total_updates <= constant_updates:
        return 1.0
    progress = min(
        1.0,
        (applied_update - constant_updates) / (total_updates - constant_updates),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    ratio = config.minimum_learning_rate_ratio
    return ratio + (1.0 - ratio) * cosine


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _strict_json(record: Mapping[str, object]) -> str:
    return json.dumps(record, sort_keys=True, allow_nan=False)


def _console_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def update_early_stopping_state(
    *,
    epoch_number: int,
    validation_reward: float,
    reference_reward: float,
    epochs_without_improvement: int,
    config: ScaleDiverseOnlineTrainingConfig,
) -> tuple[float, int, bool]:
    """Advance validation-based early stopping after one evaluated epoch."""

    if epoch_number <= config.early_stop_min_epochs:
        return max(reference_reward, validation_reward), 0, False
    if not config.early_stopping_enabled:
        return reference_reward, epochs_without_improvement, False
    if validation_reward >= reference_reward + config.early_stop_min_delta:
        return validation_reward, 0, False
    epochs_without_improvement += 1
    return (
        reference_reward,
        epochs_without_improvement,
        epochs_without_improvement >= config.early_stop_patience,
    )


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_strict_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_validation_episodes(episodes: Sequence[OnlineCRPEpisode]) -> None:
    if not episodes:
        raise ValueError("validation manifest must contain at least one episode")
    canonical_ids = [episode.canonical_id for episode in episodes]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("validation manifest contains duplicate episodes")
    if any(
        episode.metadata.get("exact_original_benchmark") is not False
        or "caserta" in str(episode.metadata.get("protocol", "")).lower()
        for episode in episodes
    ):
        raise ValueError(
            "training-time validation must use generated non-Caserta episodes"
        )


def evaluate_fixed_validation(
    policy,
    episodes: Sequence[OnlineCRPEpisode],
    *,
    lookahead_horizon: int,
    device: str,
) -> dict[str, object]:
    """Evaluate relocation count with FP32 greedy decisions on fixed instances."""
    import torch

    if not episodes:
        raise ValueError("validation episodes must not be empty")
    grouped = defaultdict(list)
    for index, episode in enumerate(episodes):
        grouped[(episode.num_stacks, episode.max_tiers)].append((index, episode))
    costs: list[int | None] = [None] * len(episodes)
    entropy_sum = 0.0
    maximum_probability_sum = 0.0
    decision_count = 0
    was_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode():
            for indexed_episodes in grouped.values():
                indices, batch = zip(*indexed_episodes)
                environment = VectorizedOnlineCRPEnvironment(
                    batch, rollouts_per_episode=1, lookahead_horizon=lookahead_horizon,
                    device=device, validate_actions=True,
                )
                while environment.active_indices.numel():
                    active = environment.active_indices
                    forbidden = environment.forbidden_mask[active]
                    log_probabilities = policy.action_log_probabilities(
                        environment.priorities[active], forbidden
                    ).float()
                    probabilities = log_probabilities.exp()
                    entropy = -(
                        probabilities * log_probabilities.masked_fill(forbidden, 0.0)
                    ).sum(dim=1)
                    entropy_sum += float(entropy.sum().cpu())
                    maximum_probability_sum += float(probabilities.max(dim=1).values.sum().cpu())
                    decision_count += int(active.numel())
                    environment.step_active_(log_probabilities.argmax(dim=1))
                for index, count in zip(indices, environment.relocation_count.cpu().tolist(), strict=True):
                    costs[index] = int(count)
    finally:
        policy.train(was_training)
    if any(cost is None for cost in costs):
        raise AssertionError("fixed validation produced incomplete results")
    complete_costs = [int(cost) for cost in costs]
    per_scale_costs = defaultdict(list)
    for episode, cost in zip(episodes, complete_costs, strict=True):
        per_scale_costs[physical_scale_id(episode)].append(cost)
    per_scale = {
        scale: {
            "episodes": len(values),
            "mean_relocations": float(np.mean(values)),
            "mean_reward": -float(np.mean(values)),
        }
        for scale, values in sorted(per_scale_costs.items())
    }
    macro = float(np.mean([entry["mean_relocations"] for entry in per_scale.values()]))
    micro = float(np.mean(complete_costs))
    return {
        "validation_macro_mean_relocations": macro,
        "validation_macro_reward": -macro,
        "validation_micro_mean_relocations": micro,
        "validation_micro_reward": -micro,
        "validation_policy_entropy": _finite_or_none(entropy_sum / max(decision_count, 1)),
        "validation_max_action_probability": _finite_or_none(maximum_probability_sum / max(decision_count, 1)),
        "validation_decisions": decision_count,
        "validation_per_scale": per_scale,
        "validation_episode_relocations": {
            episode.canonical_id: cost
            for episode, cost in zip(episodes, complete_costs, strict=True)
        },
    }


def _configure_precision(precision_mode: str, device: str) -> None:
    """Configure global FP32 matmul behavior for one training process."""

    import torch

    device_type = torch.device(device).type
    if precision_mode in {"tf32", "fp16"} and device_type != "cuda":
        raise ValueError(f"{precision_mode} training requires a CUDA device")
    if precision_mode == "bf16" and device_type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("this CUDA device does not support BF16 training")
    use_tf32 = precision_mode == "tf32"
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")


def _autocast_context(precision_mode: str, device: str):
    import torch

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(precision_mode)
    if dtype is None:
        return nullcontext()
    return torch.autocast(
        device_type=torch.device(device).type,
        dtype=dtype,
        enabled=True,
    )


def _make_policy(config: ScaleDiverseOnlineTrainingConfig, device: str):
    return ScaleDiverseRelocationPolicy(
        lookahead_horizon=config.lookahead_horizon,
        embed_dim=config.embed_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        clip_constant=config.clip_constant,
        validate_inputs=config.runtime_validation,
    ).to(device)


def _generate_training_episodes(
    config: ScaleDiverseOnlineTrainingConfig,
    group: tuple[int, int],
    *,
    episode_seed: int,
) -> tuple[OnlineCRPEpisode, ...]:
    return generate_table2_structural_sample(
        group, episode_count=config.instances_per_scale, master_seed=episode_seed
    )


def _collect_scale_rollouts_scalar(
    policy,
    episodes: Sequence[OnlineCRPEpisode],
    *,
    rollouts_per_instance: int,
    lookahead_horizon: int,
    device: str,
    precision_mode: str = "fp32",
    cover_first_actions: bool = False,
):
    import torch

    if not episodes:
        raise ValueError("scale rollout collection requires episodes")
    shape = (episodes[0].num_stacks, episodes[0].max_tiers)
    if any((episode.num_stacks, episode.max_tiers) != shape for episode in episodes):
        raise ValueError("all episodes in a scale batch must share a layout shape")
    environments = [
        OnlineCRPEnvironment(
            episode,
            lookahead_horizon=lookahead_horizon,
        )
        for episode in episodes
        for _ in range(rollouts_per_instance)
    ]
    states = [environment.state for environment in environments]
    adapter = Online2DPriorityAdapter(episodes[0].max_tiers)
    saved_priorities: list[np.ndarray] = []
    saved_forbidden: list[np.ndarray] = []
    saved_actions: list[np.ndarray] = []
    saved_trajectory_indices: list[np.ndarray] = []
    policy_entropy_sum = 0.0
    maximum_probability_sum = 0.0
    decision_count = 0
    was_training = policy.training
    policy.eval()
    first_decision = True
    while True:
        active = [index for index, state in enumerate(states) if not state.terminated]
        if not active:
            break
        priorities_np = np.stack([adapter.build(states[index]) for index in active])
        forbidden_np = np.ones(
            (len(active), episodes[0].num_stacks), dtype=np.bool_
        )
        legal_by_rollout = []
        for row, index in enumerate(active):
            legal = environments[index].legal_destinations()
            if not legal:
                raise RuntimeError("active OCRP rollout has no legal destination")
            legal_by_rollout.append(legal)
            forbidden_np[row, list(legal)] = False
        priorities = torch.as_tensor(
            priorities_np,
            dtype=torch.float32,
            device=device,
        )
        forbidden = torch.as_tensor(
            forbidden_np, dtype=torch.bool, device=device
        )
        # Sampling does not need a retained computation graph.  States and
        # actions are replayed below in large decision batches for gradients.
        with torch.inference_mode():
            with _autocast_context(precision_mode, device):
                all_log_probabilities = policy.action_log_probabilities(priorities, forbidden)
            probabilities = all_log_probabilities.exp()
            finite_log_probabilities = all_log_probabilities.masked_fill(
                forbidden, 0.0
            )
            entropy = -(probabilities * finite_log_probabilities).sum(dim=1)
            policy_entropy_sum += float(entropy.sum().cpu())
            maximum_probability_sum += float(
                probabilities.max(dim=1).values.sum().cpu()
            )
            decision_count += len(active)
            if cover_first_actions and first_decision:
                actions = torch.as_tensor(
                    [
                        legal_by_rollout[row][
                            (index % rollouts_per_instance)
                            % len(legal_by_rollout[row])
                        ]
                        for row, index in enumerate(active)
                    ],
                    dtype=torch.int64,
                    device=device,
                )
            else:
                actions = torch.distributions.Categorical(
                    logits=all_log_probabilities
                ).sample()
        actions_np = actions.cpu().numpy()
        saved_priorities.append(priorities_np)
        saved_forbidden.append(forbidden_np)
        saved_actions.append(actions_np)
        saved_trajectory_indices.append(np.asarray(active, dtype=np.int64))
        for row, index in enumerate(active):
            selected = int(actions_np[row])
            if selected not in legal_by_rollout[row]:
                raise AssertionError("masked policy selected an illegal destination")
            next_state, _, _ = environments[index].step(selected)
            states[index] = next_state
        first_decision = False
    if was_training:
        policy.train()

    costs = np.asarray(
        [environment.relocation_count for environment in environments],
        dtype=np.float32,
    ).reshape(len(episodes), rollouts_per_instance)
    centered = costs - costs.mean(axis=1, keepdims=True)
    rollout_standard_deviation = costs.std(axis=1, keepdims=True)
    normalized_costs = centered / (rollout_standard_deviation + 1e-8)
    r_per_n_costs = costs / np.asarray(
        [episode.num_containers for episode in episodes], dtype=np.float32
    )[:, None]
    trajectory_indices = np.concatenate(saved_trajectory_indices)
    return {
        "priorities": np.concatenate(saved_priorities),
        "forbidden": np.concatenate(saved_forbidden),
        "actions": np.concatenate(saved_actions),
        "trajectory_indices": trajectory_indices,
        "normalized_costs": normalized_costs.reshape(-1),
        "trajectory_lengths": np.bincount(
            trajectory_indices, minlength=len(environments)
        ).astype(np.float32),
        "costs": costs,
        "policy_entropy_sum": policy_entropy_sum,
        "maximum_probability_sum": maximum_probability_sum,
        "decision_count": decision_count,
        "zero_advantage_instances": int(
            np.count_nonzero(rollout_standard_deviation.reshape(-1) == 0)
        ),
        "instance_count": len(episodes),
        "mean_rollout_cost_std": float(rollout_standard_deviation.mean()),
        "mean_rollout_r_per_n_std": float(r_per_n_costs.std(axis=1).mean()),
    }


def _collect_scale_rollouts_vectorized(
    policy,
    episodes: Sequence[OnlineCRPEpisode],
    *,
    rollouts_per_instance: int,
    lookahead_horizon: int,
    device: str,
    precision_mode: str = "fp32",
    validate_actions: bool = False,
    cover_first_actions: bool = False,
):
    """Sample same-scale rollouts without leaving Torch/CUDA tensors."""

    import torch

    if not episodes:
        raise ValueError("scale rollout collection requires episodes")
    environment = VectorizedOnlineCRPEnvironment(
        episodes,
        rollouts_per_episode=rollouts_per_instance,
        lookahead_horizon=lookahead_horizon,
        device=device,
        validate_actions=validate_actions,
    )
    saved_priorities: list[torch.Tensor] = []
    saved_forbidden: list[torch.Tensor] = []
    saved_actions: list[torch.Tensor] = []
    saved_trajectory_indices: list[torch.Tensor] = []
    policy_entropy_sum = 0.0
    maximum_probability_sum = 0.0
    decision_count = 0
    was_training = policy.training
    policy.eval()
    first_decision = True
    while True:
        active_indices = environment.active_indices
        if active_indices.numel() == 0:
            break
        priorities = environment.priorities[active_indices]
        forbidden = environment.forbidden_mask[active_indices]
        with torch.inference_mode():
            with _autocast_context(precision_mode, device):
                log_probabilities = policy.action_log_probabilities(priorities, forbidden)
            probabilities = log_probabilities.exp()
            finite_log_probabilities = log_probabilities.masked_fill(forbidden, 0.0)
            entropy = -(probabilities * finite_log_probabilities).sum(dim=1)
            policy_entropy_sum += float(entropy.sum().cpu())
            maximum_probability_sum += float(
                probabilities.max(dim=1).values.sum().cpu()
            )
            decision_count += int(active_indices.numel())
            if cover_first_actions and first_decision:
                actions = torch.empty_like(active_indices)
                legal = ~forbidden
                for row in range(active_indices.numel()):
                    destinations = torch.nonzero(legal[row], as_tuple=False).squeeze(1)
                    rollout = int(environment.rollout_indices[active_indices[row]])
                    actions[row] = destinations[rollout % destinations.numel()]
            else:
                actions = torch.distributions.Categorical(
                    logits=log_probabilities
                ).sample()
        saved_priorities.append(priorities)
        saved_forbidden.append(forbidden)
        saved_actions.append(actions)
        saved_trajectory_indices.append(active_indices)
        environment.step_active_(actions)
        first_decision = False
    if was_training:
        policy.train()

    costs = environment.relocation_count.to(torch.float32).reshape(
        len(episodes), rollouts_per_instance
    )
    centered = costs - costs.mean(dim=1, keepdim=True)
    rollout_standard_deviation = costs.std(
        dim=1, keepdim=True, correction=0
    )
    normalized_costs = centered / (rollout_standard_deviation + 1e-8)
    r_per_n_costs = costs / torch.as_tensor(
        [episode.num_containers for episode in episodes],
        dtype=costs.dtype,
        device=costs.device,
    )[:, None]
    trajectory_indices = torch.cat(saved_trajectory_indices)
    return {
        "priorities": torch.cat(saved_priorities),
        "forbidden": torch.cat(saved_forbidden),
        "actions": torch.cat(saved_actions),
        "trajectory_indices": trajectory_indices,
        "normalized_costs": normalized_costs.reshape(-1),
        "trajectory_lengths": torch.bincount(
            trajectory_indices, minlength=environment.batch_size
        ).to(torch.float32),
        "costs": costs,
        "policy_entropy_sum": policy_entropy_sum,
        "maximum_probability_sum": maximum_probability_sum,
        "decision_count": decision_count,
        "zero_advantage_instances": int(
            (rollout_standard_deviation == 0).sum().item()
        ),
        "instance_count": len(episodes),
        "mean_rollout_cost_std": float(rollout_standard_deviation.mean().cpu()),
        "mean_rollout_r_per_n_std": float(
            r_per_n_costs.std(dim=1, correction=0).mean().cpu()
        ),
    }


def _collect_scale_rollouts(
    policy,
    episodes: Sequence[OnlineCRPEpisode],
    *,
    rollouts_per_instance: int,
    lookahead_horizon: int,
    device: str,
    precision_mode: str = "fp32",
    environment_engine: str = "scalar",
    runtime_validation: bool = False,
    cover_first_actions: bool = False,
):
    if environment_engine == "vectorized":
        return _collect_scale_rollouts_vectorized(
            policy,
            episodes,
            rollouts_per_instance=rollouts_per_instance,
            lookahead_horizon=lookahead_horizon,
            device=device,
            precision_mode=precision_mode,
            validate_actions=runtime_validation,
            cover_first_actions=cover_first_actions,
        )
    if environment_engine != "scalar":
        raise ValueError("environment_engine must be scalar or vectorized")
    return _collect_scale_rollouts_scalar(
        policy,
        episodes,
        rollouts_per_instance=rollouts_per_instance,
        lookahead_horizon=lookahead_horizon,
        device=device,
        precision_mode=precision_mode,
        cover_first_actions=cover_first_actions,
    )


def _backward_replay(
    policy,
    collected,
    *,
    decision_batch_size: int,
    loss_denominator: int,
    entropy_coefficient: float,
    device: str,
    loss_weighting: str = LOSS_WEIGHTING_EQUAL_TRAJECTORY,
    precision_mode: str = "fp32",
    grad_scaler=None,
    router_balance_coefficient: float = 0.0,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict[str, object]]:
    """Replay sampled decisions and backpropagate the normalized relocation loss."""
    import torch
    from scrp.moe_lora import capture_moe_lora_routing, iter_moe_lora_layers

    if loss_weighting not in {LOSS_WEIGHTING_EQUAL_TRAJECTORY, LOSS_WEIGHTING_LEGACY}:
        raise ValueError("unsupported loss_weighting")
    if (
        not math.isfinite(router_balance_coefficient)
        or router_balance_coefficient < 0
    ):
        raise ValueError("router_balance_coefficient must be finite and non-negative")
    if decision_batch_size <= 0:
        raise ValueError("decision_batch_size must be positive")
    moe_layers = list(iter_moe_lora_layers(policy))
    if router_balance_coefficient and not moe_layers:
        raise ValueError("router balancing requires an installed MoE-LoRA layer")
    if len(moe_layers) > 1:
        raise ValueError("replay supports exactly one MoE-LoRA layer")
    capture_routing = bool(moe_layers) and (
        router_balance_coefficient > 0 or return_diagnostics
    )
    num_experts = moe_layers[0][1].num_experts if moe_layers else 0
    diagnostics: dict[str, object] = {
        "decision_count": 0,
        "token_count": 0,
        "router_weight_sums": [0.0] * num_experts,
        "router_entropy_sum": 0.0,
        "balance_loss_weighted_sum": 0.0,
        "expert_delta_norm_sums": [0.0] * num_experts,
    }
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    decision_count = len(collected["actions"])
    for start in range(0, decision_count, decision_batch_size):
        indices = slice(start, min(start + decision_batch_size, decision_count))
        priorities = torch.as_tensor(collected["priorities"][indices], dtype=torch.float32, device=device)
        forbidden = torch.as_tensor(collected["forbidden"][indices], dtype=torch.bool, device=device)
        actions = torch.as_tensor(collected["actions"][indices], dtype=torch.int64, device=device)
        trajectory_indices = collected["trajectory_indices"][indices]
        trajectory_lengths = torch.as_tensor(collected["trajectory_lengths"], dtype=torch.float32, device=device)
        decision_weights = torch.as_tensor(
            collected["normalized_costs"][trajectory_indices],
            dtype=torch.float32, device=device,
        )
        if loss_weighting == LOSS_WEIGHTING_EQUAL_TRAJECTORY:
            decision_weights = decision_weights / trajectory_lengths[trajectory_indices].clamp_min(1.0)
        capture_context = (
            capture_moe_lora_routing(policy) if capture_routing else nullcontext(None)
        )
        with capture_context as routing_capture:
            with _autocast_context(precision_mode, device):
                log_probabilities = policy.action_log_probabilities(priorities, forbidden)
            log_probabilities = log_probabilities.float()
            selected = log_probabilities.gather(1, actions[:, None]).squeeze(1)
            loss = (decision_weights * selected).sum() / loss_denominator
            if entropy_coefficient:
                entropy = -(log_probabilities.exp() * log_probabilities.masked_fill(forbidden, 0.0)).sum(dim=1)
                if loss_weighting == LOSS_WEIGHTING_EQUAL_TRAJECTORY:
                    entropy = entropy / trajectory_lengths[trajectory_indices].clamp_min(1.0)
                loss = loss - entropy_coefficient * entropy.sum() / loss_denominator

            if routing_capture is not None:
                if not routing_capture.probabilities:
                    raise RuntimeError("MoE-LoRA forward produced no routing probabilities")
                routing = torch.cat(
                    [probability.reshape(-1, num_experts) for probability in routing_capture.probabilities],
                    dim=0,
                )
                chunk_decisions = int(actions.numel())
                balance_loss = -torch.log(
                    routing.mean(dim=0).clamp_min(1e-8)
                ).mean() - math.log(num_experts)
                chunk_weight = chunk_decisions / decision_count if decision_count else 0.0
                loss = loss + (
                    router_balance_coefficient * chunk_weight * balance_loss
                )
                routing_detached = routing.detach().float()
                diagnostics["decision_count"] = int(diagnostics["decision_count"]) + chunk_decisions
                diagnostics["token_count"] = int(diagnostics["token_count"]) + routing.shape[0]
                diagnostics["router_weight_sums"] = [
                    left + float(right)
                    for left, right in zip(
                        diagnostics["router_weight_sums"],
                        routing_detached.sum(dim=0).cpu().tolist(),
                        strict=True,
                    )
                ]
                diagnostics["router_entropy_sum"] = float(
                    diagnostics["router_entropy_sum"]
                ) + float(
                    (-(routing_detached * routing_detached.clamp_min(1e-8).log()).sum(dim=-1))
                    .sum()
                    .cpu()
                )
                diagnostics["balance_loss_weighted_sum"] = float(
                    diagnostics["balance_loss_weighted_sum"]
                ) + chunk_weight * float(balance_loss.detach().float().cpu())
                if routing_capture.expert_delta_norm_sums:
                    norm_sums = torch.stack(
                        routing_capture.expert_delta_norm_sums
                    ).sum(dim=0).detach().float().cpu().tolist()
                    diagnostics["expert_delta_norm_sums"] = [
                        left + float(right)
                        for left, right in zip(
                            diagnostics["expert_delta_norm_sums"],
                            norm_sums,
                            strict=True,
                        )
                    ]
            if grad_scaler is None:
                loss.backward()
            else:
                grad_scaler.scale(loss).backward()
        total_loss = total_loss + loss.detach().float()
    result = float(total_loss.cpu())
    return (result, diagnostics) if return_diagnostics else result


_FULL_RESUME_MUTABLE_CONFIG_FIELDS = {
    "epochs",
    "validation_every",
    "early_stopping_enabled",
    "early_stop_min_epochs",
    "early_stop_patience",
    "early_stop_min_delta",
}


def _validate_full_resume_config(
    config: ScaleDiverseOnlineTrainingConfig,
    checkpoint_config: Mapping[str, object],
) -> None:
    current = asdict(config)
    # Original H20 checkpoints predate explicit loss-weighting metadata.
    legacy_defaults = {
        "loss_weighting": LOSS_WEIGHTING_LEGACY,
    }
    mismatches = {
        name: (
            checkpoint_config.get(name, legacy_defaults.get(name, "<missing>")),
            value,
        )
        for name, value in current.items()
        if name not in _FULL_RESUME_MUTABLE_CONFIG_FIELDS
        and checkpoint_config.get(name, legacy_defaults.get(name, "<missing>"))
        != value
    }
    if mismatches:
        raise ValueError(
            "full resume requires an identical training configuration; "
            f"use resume_mode='weights-only' to change it: {mismatches}"
        )


def train(
    config: ScaleDiverseOnlineTrainingConfig,
    output: str | Path,
    *,
    device: str = "cpu",
    resume: str | Path | None = None,
    resume_mode: str = "full",
    epoch_checkpoint_dir: str | Path | None = None,
    validation_manifest: str | Path | None = None,
    metrics_output: str | Path | None = None,
    live_plot_output: str | Path | None = None,
    best_output: str | Path | None = None,
    json_console: bool = False,
) -> Path:
    import torch

    config.validate()
    if resume_mode not in {"full", "weights-only"}:
        raise ValueError("resume_mode must be full or weights-only")
    if resume is None and resume_mode != "full":
        raise ValueError("resume_mode is only meaningful when resume is provided")
    _configure_precision(config.precision_mode, device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    policy = _make_policy(config, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update: _learning_rate_multiplier(config, int(update)),
    )
    grad_scaler = None
    if config.precision_mode == "fp16":
        grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=True, init_scale=1024.0, growth_interval=2_000
        )

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    epoch_destination = (
        Path(epoch_checkpoint_dir)
        if epoch_checkpoint_dir is not None
        else destination.parent / f"{destination.stem}_epochs"
    )
    epoch_destination.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(metrics_output)
        if metrics_output is not None
        else destination.with_name(f"{destination.stem}_epoch_metrics.jsonl")
    )
    live_plot_path = (
        Path(live_plot_output)
        if live_plot_output is not None
        else destination.with_name(f"{destination.stem}_live_reward_curve.png")
    )
    best_destination = (
        Path(best_output)
        if best_output is not None
        else destination.with_name(f"{destination.stem}.best{destination.suffix}")
    )
    if best_destination == destination:
        raise ValueError("best_output must differ from the latest checkpoint output")
    best_destination.parent.mkdir(parents=True, exist_ok=True)

    validation_episodes: tuple[OnlineCRPEpisode, ...] = ()
    if validation_manifest is not None:
        validation_episodes = load_episode_manifest(validation_manifest)
        _validate_validation_episodes(validation_episodes)
        expected_protocol = "engineering_table2_structural_sample"
        if any(
            episode.metadata.get("protocol") != expected_protocol
            for episode in validation_episodes
        ):
            raise ValueError(
                "validation manifest priority distribution does not match training"
            )
    if config.early_stopping_enabled and not validation_episodes:
        raise ValueError("automatic early stopping requires a validation manifest")

    history: list[dict[str, object]] = []
    start_epoch = 0
    attempted_updates = 0
    applied_updates = 0
    skipped_updates = 0
    best_validation_reward = -math.inf
    best_validation_epoch: int | None = None
    early_stop_reference_reward = -math.inf
    epochs_without_improvement = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint.get("policy_architecture") != config.policy_architecture:
            raise ValueError("resume checkpoint architecture does not match")
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if resume_mode == "full":
            checkpoint_config = checkpoint.get("online_crp_training_config")
            if not isinstance(checkpoint_config, Mapping):
                raise ValueError("full resume checkpoint has no training configuration")
            _validate_full_resume_config(config, checkpoint_config)
            checkpoint_validation_manifest = checkpoint.get("validation_manifest")
            current_validation_manifest = (
                str(validation_manifest) if validation_manifest is not None else None
            )
            if checkpoint_validation_manifest != current_validation_manifest:
                raise ValueError(
                    "full resume requires the same validation manifest; use "
                    "resume_mode='weights-only' to change it"
                )
            if "scheduler_state_dict" not in checkpoint:
                raise ValueError(
                    "legacy checkpoint has no scheduler state; use weights-only resume"
                )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            history = list(checkpoint.get("training_history", []))
            start_epoch = int(checkpoint.get("completed_epoch", len(history)))
            attempted_updates = int(checkpoint.get("attempted_updates", 0))
            applied_updates = int(checkpoint.get("applied_updates", attempted_updates))
            skipped_updates = int(
                checkpoint.get("skipped_updates", attempted_updates - applied_updates)
            )
            best_validation_reward = float(
                checkpoint.get("best_validation_reward", -math.inf)
            )
            best_validation_epoch = checkpoint.get("best_validation_epoch")
            early_stop_reference_reward = float(
                checkpoint.get("early_stop_reference_reward", best_validation_reward)
            )
            epochs_without_improvement = int(
                checkpoint.get("epochs_without_improvement", 0)
            )
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if (
                torch.cuda.is_available()
                and checkpoint.get("cuda_rng_state") is not None
            ):
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
            if checkpoint.get("python_rng_state") is not None:
                random.setstate(checkpoint["python_rng_state"])
            if checkpoint.get("numpy_rng_state") is not None:
                np.random.set_state(checkpoint["numpy_rng_state"])
            if grad_scaler is not None and checkpoint.get("grad_scaler_state_dict"):
                grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    if start_epoch > config.epochs:
        raise ValueError("resume checkpoint is beyond the configured epoch limit")

    if not (resume is not None and resume_mode == "full"):
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")
    elif not metrics_path.exists() and history:
        for previous_record in history:
            _append_jsonl(metrics_path, previous_record)

    def save_checkpoint(*, is_best: bool) -> Path:
        completed_epoch = len(history)
        payload = {
            "checkpoint_schema_version": 1,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "online_crp_training_config": asdict(config),
            "policy_architecture": config.policy_architecture,
            "training_history": history,
            "training_source": "fresh_table2_structural_scale_diverse",
            "training_engine": "batched_sampling_replay_live_v2",
            "precision_mode": config.precision_mode,
            "environment_engine": config.environment_engine,
            "grad_scaler_state_dict": (
                grad_scaler.state_dict() if grad_scaler is not None else None
            ),
            "original_caserta_benchmark_used": False,
            "validation_manifest": (
                str(validation_manifest) if validation_manifest is not None else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "completed_epoch": completed_epoch,
            "attempted_updates": attempted_updates,
            "applied_updates": applied_updates,
            "skipped_updates": skipped_updates,
            "best_validation_reward": best_validation_reward,
            "best_validation_epoch": best_validation_epoch,
            "early_stop_reference_reward": early_stop_reference_reward,
            "epochs_without_improvement": epochs_without_improvement,
        }
        epoch_snapshot = epoch_destination / (
            f"{destination.stem}_epoch_{completed_epoch:03d}{destination.suffix}"
        )
        snapshot_temporary = epoch_snapshot.with_name(f"{epoch_snapshot.name}.tmp")
        torch.save(payload, snapshot_temporary)
        snapshot_temporary.replace(epoch_snapshot)

        latest_temporary = destination.with_name(f"{destination.name}.tmp")
        shutil.copyfile(epoch_snapshot, latest_temporary)
        latest_temporary.replace(destination)
        if is_best:
            best_temporary = best_destination.with_name(f"{best_destination.name}.tmp")
            shutil.copyfile(epoch_snapshot, best_temporary)
            best_temporary.replace(best_destination)
        return epoch_snapshot

    try:
        for epoch in range(start_epoch, config.epochs):
            epoch_number = epoch + 1
            epoch_attempted_updates_start = attempted_updates
            epoch_applied_updates_start = applied_updates
            epoch_skipped_updates_start = skipped_updates
            epoch_entropy_coefficient = entropy_coefficient_for_epoch(
                config, epoch_number
            )
            scale_schedule = build_balanced_scale_schedule(config, epoch)
            epoch_costs: list[float] = []
            epoch_losses: list[float] = []
            epoch_scale_costs: dict[str, list[float]] = defaultdict(list)
            epoch_entropy_sum = 0.0
            epoch_maximum_probability_sum = 0.0
            epoch_decision_count = 0
            epoch_zero_advantage_instances = 0
            epoch_instance_count = 0
            epoch_rollout_std_sum = 0.0
            clipped_update_count = 0
            nonfinite_gradient_count = 0
            epoch_gradient_norms: list[float] = []
            last_loss_scale: float | None = None
            for step, sampled_scales in enumerate(scale_schedule):
                optimizer.zero_grad()
                step_losses: list[float] = []
                step_costs: list[float] = []
                step_entropy_sum = 0.0
                step_maximum_probability_sum = 0.0
                step_decision_count = 0
                step_zero_advantage_instances = 0
                step_instance_count = 0
                step_rollout_std_sum = 0.0
                for scale_index, group in enumerate(sampled_scales):
                    episode_seed = (
                        config.seed
                        + epoch * 1_000_003
                        + step * 10_007
                        + scale_index * 101
                    )
                    episodes = _generate_training_episodes(
                        config,
                        group,
                        episode_seed=episode_seed,
                    )
                    collected = _collect_scale_rollouts(
                        policy,
                        episodes,
                        rollouts_per_instance=config.rollouts_per_instance,
                        lookahead_horizon=config.lookahead_horizon,
                        device=device,
                        precision_mode=config.precision_mode,
                        environment_engine=config.environment_engine,
                        runtime_validation=config.runtime_validation,
                        cover_first_actions=config.cover_first_actions,
                    )
                    scale_loss = _backward_replay(
                        policy,
                        collected,
                        decision_batch_size=config.replay_decision_batch_size,
                        loss_denominator=(
                            config.scales_per_step
                            * config.instances_per_scale
                            * config.rollouts_per_instance
                        ),
                        entropy_coefficient=epoch_entropy_coefficient,
                        device=device,
                        loss_weighting=config.loss_weighting,
                        precision_mode=config.precision_mode,
                        grad_scaler=grad_scaler,
                    )
                    step_losses.append(scale_loss)
                    raw_costs = collected["costs"].reshape(-1).tolist()
                    rollout_costs = [float(value) for value in raw_costs]
                    scale_key = f"S{group[0]}-W{group[1]}"
                    epoch_scale_costs[scale_key].extend(rollout_costs)
                    step_costs.extend(rollout_costs)
                    epoch_costs.extend(rollout_costs)
                    step_entropy_sum += float(collected["policy_entropy_sum"])
                    step_maximum_probability_sum += float(
                        collected["maximum_probability_sum"]
                    )
                    step_decision_count += int(collected["decision_count"])
                    step_zero_advantage_instances += int(
                        collected["zero_advantage_instances"]
                    )
                    instances = int(collected["instance_count"])
                    step_instance_count += instances
                    step_rollout_std_sum += (
                        float(collected["mean_rollout_cost_std"]) * instances
                    )
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), config.gradient_clip_norm
                )
                gradient_norm_value = float(gradient_norm.detach().cpu())
                gradient_is_finite = math.isfinite(gradient_norm_value)
                if gradient_is_finite:
                    epoch_gradient_norms.append(gradient_norm_value)
                if gradient_is_finite and gradient_norm_value > config.gradient_clip_norm:
                    clipped_update_count += 1
                if not gradient_is_finite:
                    nonfinite_gradient_count += 1
                attempted_updates += 1
                if grad_scaler is None:
                    if gradient_is_finite:
                        optimizer.step()
                        optimizer_step_applied = True
                    else:
                        optimizer_step_applied = False
                    current_loss_scale = None
                else:
                    previous_loss_scale = float(grad_scaler.get_scale())
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    current_loss_scale = float(grad_scaler.get_scale())
                    optimizer_step_applied = (
                        gradient_is_finite
                        and current_loss_scale >= previous_loss_scale
                    )
                last_loss_scale = current_loss_scale
                if optimizer_step_applied:
                    applied_updates += 1
                    scheduler.step()
                else:
                    skipped_updates += 1
                epoch_loss = float(np.mean(step_losses))
                epoch_losses.append(epoch_loss)
                epoch_entropy_sum += step_entropy_sum
                epoch_maximum_probability_sum += step_maximum_probability_sum
                epoch_decision_count += step_decision_count
                epoch_zero_advantage_instances += step_zero_advantage_instances
                epoch_instance_count += step_instance_count
                epoch_rollout_std_sum += step_rollout_std_sum

            train_per_scale = {
                scale: {
                    "sampled_trajectories": len(values),
                    "mean_relocations": float(np.mean(values)),
                    "mean_reward": -float(np.mean(values)),
                }
                for scale, values in sorted(epoch_scale_costs.items())
            }
            train_scale_means = [
                value["mean_relocations"] for value in train_per_scale.values()
            ]
            train_macro_relocations = float(np.mean(train_scale_means))
            train_micro_relocations = float(np.mean(epoch_costs))
            epoch_record: dict[str, object] = {
                "record_type": "epoch",
                "epoch": epoch_number,
                "attempted_updates": attempted_updates,
                "applied_updates": applied_updates,
                "skipped_updates": skipped_updates,
                "optimizer_updates": applied_updates,
                "epoch_attempted_updates": (
                    attempted_updates - epoch_attempted_updates_start
                ),
                "epoch_applied_updates": applied_updates - epoch_applied_updates_start,
                "epoch_skipped_updates": skipped_updates - epoch_skipped_updates_start,
                "optimizer_skip_fraction": (
                    (skipped_updates - epoch_skipped_updates_start)
                    / max(attempted_updates - epoch_attempted_updates_start, 1)
                ),
                "mean_normalized_loss": _finite_or_none(float(np.mean(epoch_losses))),
                "mean_relocations": train_micro_relocations,
                "train_macro_mean_relocations": train_macro_relocations,
                "train_macro_reward": -train_macro_relocations,
                "train_micro_mean_relocations": train_micro_relocations,
                "train_micro_reward": -train_micro_relocations,
                "train_per_scale": train_per_scale,
                "scale_coverage": len(train_per_scale),
                "sampled_trajectories": len(epoch_costs),
                "policy_entropy": _finite_or_none(
                    epoch_entropy_sum / max(epoch_decision_count, 1)
                ),
                "max_action_probability": _finite_or_none(
                    epoch_maximum_probability_sum / max(epoch_decision_count, 1)
                ),
                "zero_advantage_fraction": (
                    epoch_zero_advantage_instances / max(epoch_instance_count, 1)
                ),
                "mean_rollout_cost_std": _finite_or_none(
                    epoch_rollout_std_sum / max(epoch_instance_count, 1)
                ),
                "gradient_clip_fraction": clipped_update_count
                / config.steps_per_epoch,
                "nonfinite_gradient_count": nonfinite_gradient_count,
                "mean_gradient_norm": (
                    float(np.mean(epoch_gradient_norms)) if epoch_gradient_norms else None
                ),
                "zero_gradient_fraction": (
                    float(np.mean(np.asarray(epoch_gradient_norms) == 0))
                    if epoch_gradient_norms else None
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "loss_scale": _finite_or_none(last_loss_scale),
                "entropy_coefficient": epoch_entropy_coefficient,
                "validation_macro_reward": None,
                "validation_macro_mean_relocations": None,
                "early_stop_triggered": False,
                "early_stop_reason": None,
            }
            validation_ran = bool(
                validation_episodes and epoch_number % config.validation_every == 0
            )
            is_best = False
            if validation_ran:
                validation_record = evaluate_fixed_validation(
                    policy, validation_episodes,
                    lookahead_horizon=config.lookahead_horizon, device=device,
                )
                epoch_record.update(validation_record)
                validation_reward = float(
                    validation_record["validation_macro_reward"]
                )
                if validation_reward > best_validation_reward:
                    best_validation_reward = validation_reward
                    best_validation_epoch = epoch_number
                    is_best = True
                (
                    early_stop_reference_reward,
                    epochs_without_improvement,
                    early_stop_triggered,
                ) = update_early_stopping_state(
                    epoch_number=epoch_number,
                    validation_reward=validation_reward,
                    reference_reward=early_stop_reference_reward,
                    epochs_without_improvement=epochs_without_improvement,
                    config=config,
                )
                if early_stop_triggered:
                    epoch_record["early_stop_triggered"] = True
                    epoch_record["early_stop_reason"] = (
                        "validation_macro_reward did not improve by at least "
                        f"{config.early_stop_min_delta} for "
                        f"{config.early_stop_patience} validation epochs after "
                        f"epoch {config.early_stop_min_epochs}"
                    )
            epoch_record["best_validation_reward"] = _finite_or_none(
                best_validation_reward
            )
            epoch_record["best_validation_epoch"] = best_validation_epoch
            epoch_record["epochs_without_improvement"] = epochs_without_improvement
            history.append(epoch_record)
            _append_jsonl(metrics_path, epoch_record)
            epoch_snapshot = save_checkpoint(is_best=is_best)
            try:
                from experiments.plot_h20_reward_curve import (
                    update_live_reward_curve,
                )

                update_live_reward_curve(history, live_plot_path)
            except Exception as error:  # plotting must never kill training
                warnings.warn(
                    f"could not update live reward curve {live_plot_path}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            epoch_console_record = {
                "record_type": "epoch_complete",
                "epoch": epoch_number,
                "checkpoint": str(epoch_snapshot),
                "metrics_output": str(metrics_path),
                "live_plot_output": str(live_plot_path),
                "train_macro_reward": epoch_record["train_macro_reward"],
                "validation_macro_reward": epoch_record[
                    "validation_macro_reward"
                ],
                "best_validation_epoch": best_validation_epoch,
                "early_stop_triggered": epoch_record["early_stop_triggered"],
            }
            if json_console:
                print(_strict_json(epoch_console_record), flush=True)
            else:
                print(
                    f"[epoch] {epoch_number:03d}/{config.epochs:03d} "
                    f"train={float(epoch_record['train_macro_reward']):.3f} "
                    f"validation={_console_number(epoch_record['validation_macro_reward'])} "
                    f"best_epoch={best_validation_epoch or 'n/a'} "
                    f"entropy={_console_number(epoch_record['policy_entropy'])} "
                    f"zero_adv={float(epoch_record['zero_advantage_fraction']):.1%} "
                    f"skipped={epoch_record['epoch_skipped_updates']} "
                    f"checkpoint={epoch_snapshot.name}",
                    flush=True,
                )
            if epoch_record["early_stop_triggered"]:
                _append_jsonl(
                    metrics_path,
                    {
                        "record_type": "training_stop",
                        "epoch": epoch_number,
                        "reason": epoch_record["early_stop_reason"],
                        "best_validation_epoch": best_validation_epoch,
                        "best_validation_reward": _finite_or_none(
                            best_validation_reward
                        ),
                    },
                )
                break
    except KeyboardInterrupt:
        interruption_record = {
            "record_type": "training_interrupted",
            "last_completed_epoch": len(history),
            "latest_checkpoint": str(destination) if destination.exists() else None,
        }
        _append_jsonl(metrics_path, interruption_record)
        if json_console:
            print(_strict_json(interruption_record), flush=True)
        else:
            print(
                "[stopped] interrupted by user; "
                f"last_complete_epoch={len(history)} "
                f"latest={interruption_record['latest_checkpoint'] or 'none'}",
                flush=True,
            )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=105)
    parser.add_argument("--scales-per-step", type=int, default=4)
    parser.add_argument("--instances-per-scale", type=int, default=32)
    parser.add_argument("--rollouts-per-instance", type=int, default=16)
    parser.add_argument("--replay-decision-batch-size", type=int, default=2048)
    parser.add_argument("--lookahead-horizon", type=int, default=20)
    parser.add_argument(
        "--cover-first-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--loss-weighting",
        choices=(LOSS_WEIGHTING_LEGACY, LOSS_WEIGHTING_EQUAL_TRAJECTORY),
        default=LOSS_WEIGHTING_LEGACY,
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--clip-constant", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "cosine"),
        default="cosine",
    )
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--learning-rate-constant-epochs", type=int, default=2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--entropy-final-coefficient", type=float, default=0.001)
    parser.add_argument("--entropy-decay-start-epoch", type=int, default=10)
    parser.add_argument("--entropy-decay-end-epoch", type=int, default=80)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--precision-mode",
        choices=("fp32", "tf32", "bf16", "fp16"),
        default="fp32",
    )
    parser.add_argument(
        "--environment-engine",
        choices=("scalar", "vectorized"),
        default="vectorized",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--resume")
    parser.add_argument(
        "--resume-mode", choices=("full", "weights-only"), default="full"
    )
    parser.add_argument(
        "--validation-manifest",
        default="data/online_crp/h20_validation_seed_20260902_420.json",
    )
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument(
        "--metrics-output"
    )
    parser.add_argument(
        "--json-console",
        action="store_true",
        help="print epoch summaries as JSON instead of concise text",
    )
    parser.add_argument(
        "--live-plot-output"
    )
    parser.add_argument("--best-output")
    parser.add_argument("--early-stop-min-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=25)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.05)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--runtime-validation", action="store_true")
    parser.add_argument(
        "--epoch-checkpoint-dir",
        help=(
            "directory for immutable per-epoch checkpoints; defaults to "
            "<output-stem>_epochs next to --output"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ScaleDiverseOnlineTrainingConfig(
        cover_first_actions=args.cover_first_actions,
        loss_weighting=args.loss_weighting,
        lookahead_horizon=args.lookahead_horizon,
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        clip_constant=args.clip_constant,
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        learning_rate_constant_epochs=args.learning_rate_constant_epochs,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        scales_per_step=args.scales_per_step,
        instances_per_scale=args.instances_per_scale,
        rollouts_per_instance=args.rollouts_per_instance,
        replay_decision_batch_size=args.replay_decision_batch_size,
        entropy_coefficient=args.entropy_coefficient,
        entropy_final_coefficient=args.entropy_final_coefficient,
        entropy_decay_start_epoch=args.entropy_decay_start_epoch,
        entropy_decay_end_epoch=args.entropy_decay_end_epoch,
        gradient_clip_norm=args.gradient_clip_norm,
        runtime_validation=args.runtime_validation,
        precision_mode=args.precision_mode,
        environment_engine=args.environment_engine,
        seed=args.seed,
        validation_every=args.validation_every,
        early_stopping_enabled=not args.disable_early_stopping,
        early_stop_min_epochs=args.early_stop_min_epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    print(
        train(
            config,
            args.output,
            device=args.device,
            resume=args.resume,
            resume_mode=args.resume_mode,
            epoch_checkpoint_dir=args.epoch_checkpoint_dir,
            validation_manifest=args.validation_manifest,
            metrics_output=args.metrics_output,
            live_plot_output=args.live_plot_output,
            best_output=args.best_output,
            json_console=args.json_console,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
