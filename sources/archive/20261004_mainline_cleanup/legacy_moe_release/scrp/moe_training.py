"""Training primitives used by the final H=20 MoE-LoRA experiment.

This module intentionally contains no generic LoRA or base-policy trainer.
The public experiment fixes vectorized OCRP rollouts, equal-trajectory loss
weighting, FP32 validation, and one dynamically routed MoE-LoRA layer.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .moe_lora import capture_moe_lora_routing, iter_moe_lora_layers
from .online_crp import OnlineCRPEpisode
from .vectorized_online_crp import VectorizedOnlineCRPEnvironment


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    """Append one durable, standards-compliant JSON record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(record, allow_nan=False, sort_keys=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def configure_precision(precision_mode: str, device: str) -> None:
    """Configure process-wide matmul behavior before creating the model."""

    device_type = torch.device(device).type
    if precision_mode in {"tf32", "fp16"} and device_type != "cuda":
        raise ValueError(f"{precision_mode} training requires a CUDA device")
    if precision_mode == "bf16" and device_type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("this CUDA device does not support BF16 training")
    if precision_mode not in {"fp32", "tf32", "bf16", "fp16"}:
        raise ValueError("precision_mode must be fp32, tf32, bf16, or fp16")
    use_tf32 = precision_mode == "tf32"
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")


def autocast_context(precision_mode: str, device: str):
    """Create the run's mixed-precision context or a no-op FP32 context."""

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision_mode)
    if dtype is None:
        return nullcontext()
    return torch.autocast(
        device_type=torch.device(device).type,
        dtype=dtype,
        enabled=True,
    )


def _physical_scale_id(episode: OnlineCRPEpisode) -> str:
    initial_height = max(len(stack) for stack in episode.initial_stacks)
    return f"S{initial_height}-W{episode.num_stacks}"


def evaluate_fixed_validation(
    policy: torch.nn.Module,
    episodes: Sequence[OnlineCRPEpisode],
    *,
    lookahead_horizon: int,
    device: str,
) -> dict[str, object]:
    """Evaluate fixed validation instances with greedy FP32 decisions."""

    if not episodes:
        raise ValueError("validation episodes must not be empty")
    grouped: dict[tuple[int, int], list[tuple[int, OnlineCRPEpisode]]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        grouped[(episode.num_stacks, episode.max_tiers)].append((index, episode))

    costs: list[int | None] = [None] * len(episodes)
    entropy_sum = 0.0
    maximum_probability_sum = 0.0
    decision_count = 0
    was_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(device).type, enabled=False
        ):
            for indexed_episodes in grouped.values():
                indices, batch = zip(*indexed_episodes)
                environment = VectorizedOnlineCRPEnvironment(
                    batch,
                    rollouts_per_episode=1,
                    lookahead_horizon=lookahead_horizon,
                    device=device,
                    validate_actions=True,
                )
                while environment.active_indices.numel():
                    active = environment.active_indices
                    forbidden = environment.forbidden_mask[active]
                    log_probabilities = policy.action_log_probabilities(
                        environment.priorities[active], forbidden
                    ).float()
                    probabilities = log_probabilities.exp()
                    entropy = -(
                        probabilities
                        * log_probabilities.masked_fill(forbidden, 0.0)
                    ).sum(dim=1)
                    entropy_sum += float(entropy.sum().cpu())
                    maximum_probability_sum += float(
                        probabilities.max(dim=1).values.sum().cpu()
                    )
                    decision_count += int(active.numel())
                    environment.step_active_(log_probabilities.argmax(dim=1))
                for index, count in zip(
                    indices,
                    environment.relocation_count.cpu().tolist(),
                    strict=True,
                ):
                    costs[index] = int(count)
    finally:
        policy.train(was_training)

    if any(cost is None for cost in costs):
        raise AssertionError("fixed validation produced incomplete results")
    complete_costs = [int(cost) for cost in costs]
    per_scale_costs: dict[str, list[int]] = defaultdict(list)
    for episode, cost in zip(episodes, complete_costs, strict=True):
        per_scale_costs[_physical_scale_id(episode)].append(cost)
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
        "validation_policy_entropy": entropy_sum / max(decision_count, 1),
        "validation_max_action_probability": (
            maximum_probability_sum / max(decision_count, 1)
        ),
        "validation_decisions": decision_count,
        "validation_per_scale": per_scale,
        "validation_episode_relocations": {
            episode.canonical_id: cost
            for episode, cost in zip(episodes, complete_costs, strict=True)
        },
    }


def collect_rollouts(
    policy: torch.nn.Module,
    episodes: Sequence[OnlineCRPEpisode],
    *,
    rollouts_per_instance: int,
    lookahead_horizon: int,
    device: str,
    precision_mode: str,
) -> dict[str, object]:
    """Sample same-shape trajectories without leaving Torch/CUDA tensors."""

    if not episodes:
        raise ValueError("rollout collection requires at least one episode")
    shape = (episodes[0].num_stacks, episodes[0].max_tiers)
    if any((episode.num_stacks, episode.max_tiers) != shape for episode in episodes):
        raise ValueError("a rollout batch must contain one layout shape")
    environment = VectorizedOnlineCRPEnvironment(
        episodes,
        rollouts_per_episode=rollouts_per_instance,
        lookahead_horizon=lookahead_horizon,
        device=device,
        validate_actions=False,
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
    while environment.active_indices.numel():
        active = environment.active_indices
        priorities = environment.priorities[active]
        forbidden = environment.forbidden_mask[active]
        # Sampling is graph-free; the selected decisions are replayed below
        # in bounded chunks to compute gradients.
        with torch.inference_mode(), autocast_context(precision_mode, device):
            log_probabilities = policy.action_log_probabilities(priorities, forbidden)
        probabilities = log_probabilities.exp()
        entropy = -(
            probabilities * log_probabilities.masked_fill(forbidden, 0.0)
        ).sum(dim=1)
        policy_entropy_sum += float(entropy.sum().cpu())
        maximum_probability_sum += float(
            probabilities.max(dim=1).values.sum().cpu()
        )
        decision_count += int(active.numel())
        actions = torch.distributions.Categorical(logits=log_probabilities).sample()
        saved_priorities.append(priorities)
        saved_forbidden.append(forbidden)
        saved_actions.append(actions)
        saved_trajectory_indices.append(active)
        environment.step_active_(actions)
    policy.train(was_training)

    costs = environment.relocation_count.to(torch.float32).reshape(
        len(episodes), rollouts_per_instance
    )
    centered = costs - costs.mean(dim=1, keepdim=True)
    rollout_std = costs.std(dim=1, keepdim=True, correction=0)
    normalized_costs = centered / (rollout_std + 1e-8)
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
        "zero_advantage_instances": int((rollout_std == 0).sum().item()),
        "instance_count": len(episodes),
        "mean_rollout_cost_std": float(rollout_std.mean().cpu()),
        "mean_rollout_r_per_n_std": float(
            r_per_n_costs.std(dim=1, correction=0).mean().cpu()
        ),
    }


def backward_replay(
    policy: torch.nn.Module,
    collected: Mapping[str, object],
    *,
    decision_batch_size: int,
    loss_denominator: int,
    entropy_coefficient: float,
    router_balance_coefficient: float,
    device: str,
    precision_mode: str,
    grad_scaler: torch.amp.GradScaler | None,
) -> tuple[float, dict[str, object]]:
    """Replay sampled decisions using the paper's equal-trajectory loss."""

    if decision_batch_size <= 0 or loss_denominator <= 0:
        raise ValueError("replay batch size and loss denominator must be positive")
    if not math.isfinite(router_balance_coefficient) or router_balance_coefficient < 0:
        raise ValueError("router balance coefficient must be finite and non-negative")
    moe_layers = list(iter_moe_lora_layers(policy))
    if len(moe_layers) != 1:
        raise ValueError("replay requires exactly one MoE-LoRA layer")
    num_experts = moe_layers[0][1].num_experts
    diagnostics: dict[str, object] = {
        "decision_count": 0,
        "token_count": 0,
        "router_weight_sums": [0.0] * num_experts,
        "router_entropy_sum": 0.0,
        "balance_loss_weighted_sum": 0.0,
        "expert_delta_norm_sums": [0.0] * num_experts,
    }
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    actions_all = collected["actions"]
    if not isinstance(actions_all, torch.Tensor):
        raise TypeError("collected actions must be a tensor")
    decision_count = len(actions_all)
    for start in range(0, decision_count, decision_batch_size):
        indices = slice(start, min(start + decision_batch_size, decision_count))
        priorities = torch.as_tensor(
            collected["priorities"][indices], dtype=torch.float32, device=device
        )
        forbidden = torch.as_tensor(
            collected["forbidden"][indices], dtype=torch.bool, device=device
        )
        actions = torch.as_tensor(
            collected["actions"][indices], dtype=torch.int64, device=device
        )
        trajectory_indices = torch.as_tensor(
            collected["trajectory_indices"][indices], dtype=torch.int64, device=device
        )
        trajectory_lengths = torch.as_tensor(
            collected["trajectory_lengths"], dtype=torch.float32, device=device
        )
        decision_weights = torch.as_tensor(
            collected["normalized_costs"], dtype=torch.float32, device=device
        )[trajectory_indices]
        decision_weights = decision_weights / trajectory_lengths[
            trajectory_indices
        ].clamp_min(1.0)

        with capture_moe_lora_routing(policy) as routing_capture:
            with autocast_context(precision_mode, device):
                log_probabilities = policy.action_log_probabilities(
                    priorities, forbidden
                )
            log_probabilities = log_probabilities.float()
            selected = log_probabilities.gather(1, actions[:, None]).squeeze(1)
            loss = (decision_weights * selected).sum() / loss_denominator
            if entropy_coefficient:
                entropy = -(
                    log_probabilities.exp()
                    * log_probabilities.masked_fill(forbidden, 0.0)
                ).sum(dim=1)
                entropy = entropy / trajectory_lengths[trajectory_indices].clamp_min(1.0)
                loss = loss - entropy_coefficient * entropy.sum() / loss_denominator

            routing = torch.cat(
                [probability.reshape(-1, num_experts) for probability in routing_capture.probabilities],
                dim=0,
            )
            chunk_decisions = int(actions.numel())
            balance_loss = -torch.log(
                routing.mean(dim=0).clamp_min(1e-8)
            ).mean() - math.log(num_experts)
            chunk_weight = chunk_decisions / decision_count
            loss = loss + router_balance_coefficient * chunk_weight * balance_loss

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
            ) + chunk_weight * float(balance_loss.detach().cpu())
            norm_sums = torch.stack(
                routing_capture.expert_delta_norm_sums
            ).sum(dim=0).detach().float().cpu().tolist()
            diagnostics["expert_delta_norm_sums"] = [
                left + float(right)
                for left, right in zip(
                    diagnostics["expert_delta_norm_sums"], norm_sums, strict=True
                )
            ]
            if grad_scaler is None:
                loss.backward()
            else:
                grad_scaler.scale(loss).backward()
        total_loss += loss.detach().float()
    return float(total_loss.cpu()), diagnostics


__all__ = [
    "append_jsonl",
    "backward_replay",
    "collect_rollouts",
    "configure_precision",
    "evaluate_fixed_validation",
]
