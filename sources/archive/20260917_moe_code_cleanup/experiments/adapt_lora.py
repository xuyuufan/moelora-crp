"""Offline H=20 generic LoRA or dynamic MoE-LoRA adaptation.

The base is frozen, only adapter parameters are trained, and sealed tests are
never used. Sampling and replay reuse the V1 relocation training helpers.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from experiments.train_online_crp_scale_diverse import (
    LOSS_WEIGHTING_EQUAL_TRAJECTORY,
    _append_jsonl,
    _backward_replay,
    _collect_scale_rollouts,
    _configure_precision,
    evaluate_fixed_validation,
)
from scrp.adaptation import load_h20_base_policy
from scrp.adaptation_data import (
    TARGET_DOMAINS,
    load_and_validate_target_domain,
    manifest_sha256,
)
from scrp.lora import (
    configure_trainable_parameters,
    install_lora,
    lora_config,
    parameter_counts,
    save_lora_adapter,
)
from scrp.moe_lora import (
    MOE_LORA_TARGET_MODULE,
    configure_moe_lora_training,
    install_moe_lora,
    moe_lora_config,
    save_moe_lora_adapter,
)
from scrp.online_crp import OnlineCRPEpisode


ADAPTATION_RUN_FORMAT = "mijnrl_h20_lora_run_v1"
MOE_LORA_RUN_FORMAT = "mijnrl_h20_moe_lora_run_v1"


@dataclass(frozen=True)
class AdaptationConfig:
    domain: str
    target_modules: tuple[str, ...]
    adapter_kind: str = "lora"
    lookahead_horizon: int = 20
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    num_experts: int = 2
    router_balance_coefficient: float = 0.0
    learning_rate: float = 3e-4
    epochs: int = 5
    batch_size: int = 16
    rollouts_per_instance: int = 8
    replay_decision_batch_size: int = 2_048
    entropy_coefficient: float = 0.002
    gradient_clip_norm: float = 1.0
    precision_mode: str = "fp16"
    environment_engine: str = "vectorized"
    runtime_validation: bool = False
    seed: int = 20260913
    initialization_seed: int | None = None
    batch_seed: int | None = None
    rollout_seed: int | None = None
    updates_per_epoch: int | None = None
    max_updates: int | None = None
    max_adaptation_minutes: float | None = None

    def validate(self) -> None:
        if self.domain not in TARGET_DOMAINS:
            raise ValueError("unsupported target domain")
        if self.lookahead_horizon != 20:
            raise ValueError("this study fixes lookahead_horizon=20")
        if self.adapter_kind not in {"lora", "moe_lora"}:
            raise ValueError("adapter_kind must be lora or moe_lora")
        if (
            self.rank <= 0
            or not np.isfinite(self.alpha)
            or self.alpha <= 0
            or not np.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("invalid LoRA configuration")
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must be non-empty and unique")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if (
            not np.isfinite(self.router_balance_coefficient)
            or self.router_balance_coefficient < 0
        ):
            raise ValueError("router_balance_coefficient must be finite and non-negative")
        if self.adapter_kind == "moe_lora":
            if self.target_modules != (MOE_LORA_TARGET_MODULE,):
                raise ValueError(
                    f"MoE-LoRA requires target_modules=({MOE_LORA_TARGET_MODULE!r},)"
                )
            if self.dropout != 0:
                raise ValueError("MoE-LoRA requires dropout=0")
        elif self.router_balance_coefficient != 0:
            raise ValueError("generic LoRA requires router_balance_coefficient=0")
        positive = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "rollouts_per_instance": self.rollouts_per_instance,
            "replay_decision_batch_size": self.replay_decision_batch_size,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"adaptation values must be positive: {positive}")
        if self.rollouts_per_instance < 2:
            raise ValueError("normalized-return adaptation requires at least two rollouts")
        if self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be non-negative")
        if self.precision_mode not in {"fp32", "tf32", "bf16", "fp16"}:
            raise ValueError("unsupported precision mode")
        if self.environment_engine not in {"scalar", "vectorized"}:
            raise ValueError("unsupported environment engine")
        if self.updates_per_epoch is not None and self.updates_per_epoch <= 0:
            raise ValueError("updates_per_epoch must be positive when supplied")
        if self.max_updates is not None and self.max_updates <= 0:
            raise ValueError("max_updates must be positive when supplied")
        if self.max_adaptation_minutes is not None and self.max_adaptation_minutes <= 0:
            raise ValueError("max_adaptation_minutes must be positive when supplied")

    def resolved_seeds(self) -> dict[str, int]:
        """Keep initialization, batch order, and rollout sampling independent."""

        return {
            "initialization": (
                self.initialization_seed
                if self.initialization_seed is not None
                else self.seed + 10_001
            ),
            "batch": self.batch_seed if self.batch_seed is not None else self.seed + 20_003,
            "rollout": (
                self.rollout_seed if self.rollout_seed is not None else self.seed + 30_007
            ),
        }


def _shape_batches(
    episodes: Sequence[OnlineCRPEpisode],
    *,
    batch_size: int,
    rng: random.Random,
    updates_per_epoch: int | None,
) -> list[tuple[OnlineCRPEpisode, ...]]:
    """Build same-shape batches while covering every fixed training instance."""

    grouped: dict[tuple[int, int], list[OnlineCRPEpisode]] = {}
    for episode in episodes:
        grouped.setdefault((episode.num_stacks, episode.max_tiers), []).append(episode)
    batches: list[tuple[OnlineCRPEpisode, ...]] = []
    for group in grouped.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            if len(batch) < batch_size:
                batch.extend(shuffled[index % len(shuffled)] for index in range(batch_size - len(batch)))
            batches.append(tuple(batch))
    rng.shuffle(batches)
    if not batches:
        raise AssertionError("non-empty manifest produced no batches")
    if updates_per_epoch is None:
        return batches
    return [batches[index % len(batches)] for index in range(updates_per_epoch)]


def _validation_r_per_n(
    validation_record: dict[str, object], episodes: Sequence[OnlineCRPEpisode]
) -> float:
    costs = validation_record.get("validation_episode_relocations")
    if not isinstance(costs, dict):
        raise AssertionError("validation produced no per-episode relocation counts")
    values = [float(costs[episode.canonical_id]) / episode.num_containers for episode in episodes]
    return float(np.mean(values))


def _frozen_parameter_snapshots(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def _seed_rollout_rng(seed: int, *, device: str) -> None:
    """Seed only sampling, after adapter initialization and batch shuffling."""

    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _apply_optimizer_update(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
    *,
    gradient_clip_norm: float,
    grad_scaler,
) -> tuple[bool, float]:
    """Apply one clipped update and always advance an active AMP scaler."""

    trainable_parameters = tuple(parameters)
    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters, gradient_clip_norm
    )
    finite_gradient = bool(torch.isfinite(gradient_norm).item())
    gradient_norm_value = float(gradient_norm.detach().cpu())
    if finite_gradient:
        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()
    else:
        optimizer.zero_grad(set_to_none=True)
        # unscale_ records the overflow and moves GradScaler into the state
        # that requires update() before the next unscale_ call.
        if grad_scaler is not None:
            grad_scaler.update()
    return finite_gradient, gradient_norm_value


def adapt(
    config: AdaptationConfig,
    *,
    base_checkpoint: str | Path,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    output: str | Path,
    best_output: str | Path | None = None,
    metrics_output: str | Path | None = None,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, object]:
    """Run offline adaptation and write a provenance-complete checkpoint."""

    config.validate()
    output_path = Path(output)
    best_path = Path(best_output) if best_output is not None else output_path.with_name(
        f"{output_path.stem}.best{output_path.suffix}"
    )
    metrics_path = Path(metrics_output) if metrics_output is not None else output_path.with_name(
        f"{output_path.stem}.metrics.jsonl"
    )
    if output_path == best_path:
        raise ValueError("output and best_output must differ")
    if not overwrite:
        existing = [path for path in (output_path, best_path, metrics_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite adaptation outputs: " + ", ".join(map(str, existing))
            )
    train_episodes = load_and_validate_target_domain(
        train_manifest, domain=config.domain, allowed_splits=("adapt_train",)
    )
    validation_episodes = load_and_validate_target_domain(
        validation_manifest, domain=config.domain, allowed_splits=("validation",)
    )
    train_ids = {episode.canonical_id for episode in train_episodes}
    validation_ids = {episode.canonical_id for episode in validation_episodes}
    if train_ids.intersection(validation_ids):
        raise ValueError("adaptation and validation manifests overlap")

    _configure_precision(config.precision_mode, device)
    seeds = config.resolved_seeds()
    torch.manual_seed(seeds["initialization"])
    np.random.seed(seeds["initialization"])
    random.seed(seeds["initialization"])
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    model, base_payload, base_hash = load_h20_base_policy(
        base_checkpoint, device=device, expected_lookahead_horizon=config.lookahead_horizon
    )
    if config.adapter_kind == "moe_lora":
        install_moe_lora(
            model,
            num_experts=config.num_experts,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        trainable_names = configure_moe_lora_training(model)
        adaptation_structure = moe_lora_config(model)
        run_format = MOE_LORA_RUN_FORMAT
        save_adapter = save_moe_lora_adapter
    else:
        install_lora(
            model, rank=config.rank, alpha=config.alpha, dropout=config.dropout,
            target_modules=config.target_modules,
        )
        trainable_names = configure_trainable_parameters(model)
        adaptation_structure = lora_config(model)
        run_format = ADAPTATION_RUN_FORMAT
        save_adapter = save_lora_adapter
    counts = parameter_counts(model)
    frozen_before = _frozen_parameter_snapshots(model)
    trainable_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    grad_scaler = (
        torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0, growth_interval=2_000)
        if config.precision_mode == "fp16"
        else None
    )
    run_metadata: dict[str, object] = {
        "run_format": run_format,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_hash,
        "base_policy_architecture": base_payload.get("policy_architecture"),
        "base_lookahead_horizon": config.lookahead_horizon,
        "adaptation_config": asdict(config),
        "resolved_seeds": seeds,
        "adapter_kind": config.adapter_kind,
        "adaptation_structure": adaptation_structure,
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
        "train_manifest_sha256": manifest_sha256(train_manifest),
        "validation_manifest_sha256": manifest_sha256(validation_manifest),
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "trainable_parameter_names": list(trainable_names),
        "parameter_counts": counts,
        "selection_metric": "validation_mean_relocations_per_container",
        "training_loss_definition": (
            "equal-trajectory normalized-cost policy loss plus action entropy "
            "and weighted router balance auxiliary loss"
            if config.adapter_kind == "moe_lora"
            else "equal-trajectory normalized-cost policy loss plus action entropy"
        ),
        "validation_precision": "fp32_greedy",
        "final_scoring_precision": "fp32",
        "test_manifest_used": False,
    }
    _append_jsonl(metrics_path, {"record_type": "run_start", **run_metadata})

    started = time.perf_counter()
    rng = random.Random(seeds["batch"])
    attempted_updates = 0
    applied_updates = 0
    skipped_updates = 0
    validation_started = time.perf_counter()
    base_validation = evaluate_fixed_validation(
        model,
        validation_episodes,
        lookahead_horizon=config.lookahead_horizon,
        device=device,
    )
    base_validation_seconds = time.perf_counter() - validation_started
    best_r_per_n = _validation_r_per_n(base_validation, validation_episodes)
    best_epoch: int | None = 0
    stop_reason: str | None = None
    history: list[dict[str, object]] = [
        {
            "record_type": "epoch",
            "epoch": 0,
            "attempted_updates": 0,
            "applied_updates": 0,
            "skipped_updates": 0,
            "epoch_mean_sampled_relocations": None,
            "epoch_mean_normalized_loss": None,
            "epoch_router_mean_weights": None,
            "epoch_router_mean_entropy": None,
            "epoch_router_balance_loss": None,
            "epoch_expert_mean_delta_norms": None,
            "epoch_decisions": 0,
            "training_seconds": 0.0,
            "validation_seconds": base_validation_seconds,
            "checkpoint_seconds": None,
            "epoch_seconds": base_validation_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "validation_mean_relocations_per_container": best_r_per_n,
            "validation_macro_mean_relocations": base_validation[
                "validation_macro_mean_relocations"
            ],
            "validation_micro_mean_relocations": base_validation[
                "validation_micro_mean_relocations"
            ],
            "validation_episode_relocations": base_validation[
                "validation_episode_relocations"
            ],
            "is_best": True,
            "stop_reason": None,
        }
    ]
    checkpoint_started = time.perf_counter()
    initial_metadata = {**run_metadata, "completed_epoch": 0, "history": history}
    save_adapter(
        output_path, model, base_checkpoint_sha256=base_hash, metadata=initial_metadata
    )
    save_adapter(
        best_path, model, base_checkpoint_sha256=base_hash, metadata=initial_metadata
    )
    history[0]["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
    history[0]["epoch_seconds"] = (
        base_validation_seconds + float(history[0]["checkpoint_seconds"])
    )
    history[0]["elapsed_seconds"] = time.perf_counter() - started
    _append_jsonl(metrics_path, history[0])

    for epoch in range(1, config.epochs + 1):
        epoch_costs: list[float] = []
        epoch_losses: list[float] = []
        epoch_decisions = 0
        epoch_gradient_norms: list[float] = []
        epoch_clipped_updates = 0
        epoch_zero_advantage_instances = 0
        epoch_instance_count = 0
        epoch_rpn_std_sum = 0.0
        epoch_router_decisions = 0
        epoch_router_tokens = 0
        epoch_router_weight_sums = [0.0] * config.num_experts
        epoch_router_entropy_sum = 0.0
        epoch_balance_decision_sum = 0.0
        epoch_expert_delta_norm_sums = [0.0] * config.num_experts
        epoch_start = time.perf_counter()
        for batch in _shape_batches(
            train_episodes,
            batch_size=config.batch_size,
            rng=rng,
            updates_per_epoch=config.updates_per_epoch,
        ):
            elapsed_minutes = (time.perf_counter() - started) / 60.0
            if config.max_updates is not None and attempted_updates >= config.max_updates:
                stop_reason = "max_updates"
                break
            if (
                config.max_adaptation_minutes is not None
                and elapsed_minutes >= config.max_adaptation_minutes
            ):
                stop_reason = "max_adaptation_minutes"
                break
            _seed_rollout_rng(seeds["rollout"] + attempted_updates, device=device)
            optimizer.zero_grad(set_to_none=True)
            collected = _collect_scale_rollouts(
                model,
                batch,
                rollouts_per_instance=config.rollouts_per_instance,
                lookahead_horizon=config.lookahead_horizon,
                device=device,
                precision_mode=config.precision_mode,
                environment_engine=config.environment_engine,
                runtime_validation=config.runtime_validation,
                cover_first_actions=False,
            )
            replay_result = _backward_replay(
                model,
                collected,
                decision_batch_size=config.replay_decision_batch_size,
                loss_denominator=len(batch),
                entropy_coefficient=config.entropy_coefficient,
                device=device,
                loss_weighting=LOSS_WEIGHTING_EQUAL_TRAJECTORY,
                precision_mode=config.precision_mode,
                grad_scaler=grad_scaler,
                router_balance_coefficient=config.router_balance_coefficient,
                return_diagnostics=config.adapter_kind == "moe_lora",
            )
            if config.adapter_kind == "moe_lora":
                loss, routing_diagnostics = replay_result
                routing_decisions = int(routing_diagnostics["decision_count"])
                epoch_router_decisions += routing_decisions
                epoch_router_tokens += int(routing_diagnostics["token_count"])
                epoch_router_weight_sums = [
                    left + float(right)
                    for left, right in zip(
                        epoch_router_weight_sums,
                        routing_diagnostics["router_weight_sums"],
                        strict=True,
                    )
                ]
                epoch_router_entropy_sum += float(
                    routing_diagnostics["router_entropy_sum"]
                )
                epoch_balance_decision_sum += routing_decisions * float(
                    routing_diagnostics["balance_loss_weighted_sum"]
                )
                epoch_expert_delta_norm_sums = [
                    left + float(right)
                    for left, right in zip(
                        epoch_expert_delta_norm_sums,
                        routing_diagnostics["expert_delta_norm_sums"],
                        strict=True,
                    )
                ]
            else:
                loss = replay_result
            attempted_updates += 1
            finite_gradient, gradient_norm_value = _apply_optimizer_update(
                optimizer,
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                gradient_clip_norm=config.gradient_clip_norm,
                grad_scaler=grad_scaler,
            )
            if finite_gradient:
                epoch_gradient_norms.append(gradient_norm_value)
                if gradient_norm_value > config.gradient_clip_norm:
                    epoch_clipped_updates += 1
            if finite_gradient:
                applied_updates += 1
            else:
                skipped_updates += 1
            epoch_losses.append(loss)
            epoch_costs.extend(torch.as_tensor(collected["costs"]).reshape(-1).cpu().tolist())
            epoch_decisions += int(collected["decision_count"])
            epoch_zero_advantage_instances += int(collected["zero_advantage_instances"])
            instances = int(collected["instance_count"])
            epoch_instance_count += instances
            epoch_rpn_std_sum += float(collected["mean_rollout_r_per_n_std"]) * instances
        training_seconds = time.perf_counter() - epoch_start
        validation_started = time.perf_counter()
        validation = evaluate_fixed_validation(
            model,
            validation_episodes,
            lookahead_horizon=config.lookahead_horizon,
            device=device,
        )
        validation_seconds = time.perf_counter() - validation_started
        validation_r_per_n = _validation_r_per_n(validation, validation_episodes)
        is_best = validation_r_per_n < best_r_per_n
        if is_best:
            best_r_per_n = validation_r_per_n
            best_epoch = epoch
        epoch_record: dict[str, object] = {
            "record_type": "epoch",
            "epoch": epoch,
            "attempted_updates": attempted_updates,
            "applied_updates": applied_updates,
            "skipped_updates": skipped_updates,
            "epoch_mean_sampled_relocations": float(np.mean(epoch_costs)) if epoch_costs else None,
            "epoch_mean_normalized_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "epoch_router_mean_weights": (
                [value / epoch_router_tokens for value in epoch_router_weight_sums]
                if epoch_router_tokens else None
            ),
            "epoch_router_mean_entropy": (
                epoch_router_entropy_sum / epoch_router_tokens
                if epoch_router_tokens else None
            ),
            "epoch_router_balance_loss": (
                epoch_balance_decision_sum / epoch_router_decisions
                if epoch_router_decisions else None
            ),
            "epoch_expert_mean_delta_norms": (
                [value / epoch_router_tokens for value in epoch_expert_delta_norm_sums]
                if epoch_router_tokens else None
            ),
            "epoch_decisions": epoch_decisions,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "checkpoint_seconds": None,
            "epoch_seconds": training_seconds + validation_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "zero_advantage_fraction": (
                epoch_zero_advantage_instances / max(epoch_instance_count, 1)
            ),
            "mean_rollout_r_per_n_std": (
                epoch_rpn_std_sum / max(epoch_instance_count, 1)
            ),
            "mean_gradient_norm": (
                float(np.mean(epoch_gradient_norms)) if epoch_gradient_norms else None
            ),
            "gradient_clip_fraction": epoch_clipped_updates
            / max(len(epoch_losses), 1),
            "validation_mean_relocations_per_container": validation_r_per_n,
            "validation_macro_mean_relocations": validation["validation_macro_mean_relocations"],
            "validation_micro_mean_relocations": validation["validation_micro_mean_relocations"],
            "validation_episode_relocations": validation["validation_episode_relocations"],
            "is_best": is_best,
            "stop_reason": stop_reason,
        }
        history.append(epoch_record)
        checkpoint_metadata = {**run_metadata, "completed_epoch": epoch, "history": history}
        checkpoint_started = time.perf_counter()
        save_adapter(
            output_path,
            model,
            base_checkpoint_sha256=base_hash,
            metadata=checkpoint_metadata,
        )
        if is_best:
            save_adapter(
                best_path,
                model,
                base_checkpoint_sha256=base_hash,
                metadata=checkpoint_metadata,
            )
        epoch_record["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
        epoch_record["epoch_seconds"] = (
            training_seconds
            + validation_seconds
            + float(epoch_record["checkpoint_seconds"])
        )
        epoch_record["elapsed_seconds"] = time.perf_counter() - started
        _append_jsonl(metrics_path, epoch_record)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "updates": applied_updates,
                    "validation_r_per_n": validation_r_per_n,
                    "best_epoch": best_epoch,
                    "stop_reason": stop_reason,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if stop_reason is not None:
            break

    parameters = dict(model.named_parameters())
    frozen_unchanged = all(
        name in parameters and not parameters[name].requires_grad
        and torch.equal(parameters[name].detach().cpu(), before)
        for name, before in frozen_before.items()
    )
    changed_trainable = [
        name
        for name, before in trainable_before.items()
        if not torch.equal(dict(model.named_parameters())[name].detach().cpu(), before)
    ]
    if not frozen_unchanged:
        raise AssertionError("frozen base parameters changed during adaptation")
    if applied_updates and not changed_trainable:
        raise AssertionError("no trainable adaptation parameter changed")
    elapsed_seconds = time.perf_counter() - started
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(torch.device(device)))
        if torch.cuda.is_available() and torch.device(device).type == "cuda"
        else 0
    )
    summary = {
        "record_type": "run_complete",
        **run_metadata,
        "output": str(output_path),
        "best_output": str(best_path),
        "metrics_output": str(metrics_path),
        "completed_epochs": len(history) - 1,
        "attempted_updates": attempted_updates,
        "applied_updates": applied_updates,
        "skipped_updates": skipped_updates,
        "best_epoch": best_epoch,
        "best_validation_mean_relocations_per_container": best_r_per_n,
        "elapsed_seconds": elapsed_seconds,
        "best_epoch_elapsed_seconds": next(
            record["elapsed_seconds"]
            for record in history
            if record["epoch"] == best_epoch
        ),
        "training_seconds_total": float(
            sum(float(record["training_seconds"]) for record in history)
        ),
        "validation_seconds_total": float(
            sum(float(record["validation_seconds"]) for record in history)
        ),
        "checkpoint_seconds_total": float(
            sum(float(record["checkpoint_seconds"] or 0.0) for record in history)
        ),
        "peak_memory_bytes": peak_memory_bytes,
        "changed_trainable_parameter_names": changed_trainable,
        "frozen_base_unchanged": frozen_unchanged,
        "stop_reason": stop_reason,
    }
    _append_jsonl(metrics_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--domain", choices=tuple(TARGET_DOMAINS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--best-output")
    parser.add_argument("--metrics-output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapter-kind", choices=("lora", "moe_lora"), default="lora")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--router-balance-coefficient", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rollouts-per-instance", type=int, default=8)
    parser.add_argument("--replay-decision-batch-size", type=int, default=2_048)
    parser.add_argument("--entropy-coefficient", type=float, default=0.002)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--precision-mode", choices=("fp32", "tf32", "bf16", "fp16"), default="fp16")
    parser.add_argument("--environment-engine", choices=("scalar", "vectorized"), default="vectorized")
    parser.add_argument("--runtime-validation", action="store_true")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--initialization-seed", type=int)
    parser.add_argument("--batch-seed", type=int)
    parser.add_argument("--rollout-seed", type=int)
    parser.add_argument("--updates-per-epoch", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--max-adaptation-minutes", type=float)
    parser.add_argument("--target-modules", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AdaptationConfig(
        domain=args.domain,
        adapter_kind=args.adapter_kind,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        num_experts=args.num_experts,
        router_balance_coefficient=args.router_balance_coefficient,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        rollouts_per_instance=args.rollouts_per_instance,
        replay_decision_batch_size=args.replay_decision_batch_size,
        entropy_coefficient=args.entropy_coefficient,
        gradient_clip_norm=args.gradient_clip_norm,
        precision_mode=args.precision_mode,
        environment_engine=args.environment_engine,
        runtime_validation=args.runtime_validation,
        seed=args.seed,
        initialization_seed=args.initialization_seed,
        batch_seed=args.batch_seed,
        rollout_seed=args.rollout_seed,
        updates_per_epoch=args.updates_per_epoch,
        max_updates=args.max_updates,
        max_adaptation_minutes=args.max_adaptation_minutes,
        target_modules=tuple(args.target_modules),
    )
    summary = adapt(
        config,
        base_checkpoint=args.base_checkpoint,
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        output=args.output,
        best_output=args.best_output,
        metrics_output=args.metrics_output,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
