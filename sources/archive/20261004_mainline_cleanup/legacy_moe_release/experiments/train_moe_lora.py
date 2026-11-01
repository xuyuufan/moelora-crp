"""Train the final H=20 dynamic MoE-LoRA adapter.

The paper configuration is fixed in :class:`MoETrainingConfig`. The CLI only
exposes the random seed, runtime device/precision, and output location. This
prevents an accidental hyperparameter change from being labeled a reproduction.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from scrp.adaptation import load_h20_base_policy
from scrp.adaptation_data import load_large_scale_split, manifest_sha256
from scrp.moe_lora import (
    configure_moe_lora_training,
    install_moe_lora,
    moe_lora_config,
    parameter_counts,
    save_moe_lora_adapter,
)
from scrp.moe_training import (
    append_jsonl,
    backward_replay,
    collect_rollouts,
    configure_precision,
    evaluate_fixed_validation,
)
from scrp.online_crp import OnlineCRPEpisode


BASE_CHECKPOINT = Path("checkpoints/h20.best.pt")
TRAIN_MANIFEST = Path(
    "data/online_crp/lora_domains/large_scale_n100_150/adapt_train.json"
)
VALIDATION_MANIFEST = Path(
    "data/online_crp/lora_domains/large_scale_n100_150/validation.json"
)
RUN_FORMAT = "mijnrl_h20_moe_lora_run_v1"


@dataclass(frozen=True)
class MoETrainingConfig:
    """Exact final experiment configuration plus an explicit CI-sized mode."""

    seed: int = 20260913
    precision_mode: str = "fp16"
    quick_check: bool = False
    domain: str = "large_scale_n100_150"
    lookahead_horizon: int = 20
    target_module: str = "stack_projection.2"
    num_experts: int = 2
    rank: int = 8
    alpha: float = 16.0
    router_balance_coefficient: float = 0.01
    learning_rate: float = 3e-4
    epochs: int = 5
    batch_size: int = 16
    rollouts_per_instance: int = 8
    replay_decision_batch_size: int = 2_048
    entropy_coefficient: float = 0.002
    gradient_clip_norm: float = 1.0
    train_limit: int | None = None
    validation_limit: int | None = None
    updates_per_epoch: int | None = None

    @classmethod
    def for_run(
        cls, *, seed: int, precision_mode: str, quick_check: bool
    ) -> "MoETrainingConfig":
        if quick_check:
            # This mode verifies the complete train/save/load path. Its outputs
            # are deliberately too small to be mistaken for paper results.
            return cls(
                seed=seed,
                precision_mode=precision_mode,
                quick_check=True,
                epochs=1,
                batch_size=2,
                rollouts_per_instance=2,
                replay_decision_batch_size=256,
                train_limit=8,
                validation_limit=4,
                updates_per_epoch=1,
            )
        return cls(seed=seed, precision_mode=precision_mode)

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.precision_mode not in {"fp32", "tf32", "bf16", "fp16"}:
            raise ValueError("unsupported precision mode")
        if self.rollouts_per_instance < 2:
            raise ValueError("normalized-return training needs at least two rollouts")
        if any(
            value <= 0
            for value in (
                self.epochs,
                self.batch_size,
                self.replay_decision_batch_size,
                self.learning_rate,
                self.gradient_clip_norm,
            )
        ):
            raise ValueError("training values must be positive")
        if not self.quick_check and any(
            value is not None
            for value in (
                self.train_limit,
                self.validation_limit,
                self.updates_per_epoch,
            )
        ):
            raise ValueError("dataset/update limits are reserved for --quick-check")

    def resolved_seeds(self) -> dict[str, int]:
        """Separate adapter initialization, batch order, and rollout sampling."""

        return {
            "initialization": self.seed + 10_001,
            "batch": self.seed + 20_003,
            "rollout": self.seed + 30_007,
        }


def _shape_batches(
    episodes: Sequence[OnlineCRPEpisode],
    *,
    batch_size: int,
    rng: random.Random,
    updates_per_epoch: int | None,
) -> list[tuple[OnlineCRPEpisode, ...]]:
    """Shuffle within shapes; pad only the last batch of each shape."""

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
                batch.extend(
                    shuffled[index % len(shuffled)]
                    for index in range(batch_size - len(batch))
                )
            batches.append(tuple(batch))
    rng.shuffle(batches)
    if not batches:
        raise AssertionError("non-empty manifest produced no batches")
    if updates_per_epoch is None:
        return batches
    return [batches[index % len(batches)] for index in range(updates_per_epoch)]


def _mean_r_per_n(
    validation_record: Mapping[str, object],
    episodes: Sequence[OnlineCRPEpisode],
) -> float:
    costs = validation_record.get("validation_episode_relocations")
    if not isinstance(costs, Mapping):
        raise AssertionError("validation produced no per-episode relocation counts")
    return float(
        np.mean(
            [
                float(costs[episode.canonical_id]) / episode.num_containers
                for episode in episodes
            ]
        )
    )


def _frozen_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def _seed_rollout(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
    *,
    gradient_clip_norm: float,
    grad_scaler: torch.amp.GradScaler | None,
) -> tuple[bool, float]:
    """Clip one update and recover the AMP scaler after an overflow."""

    trainable = tuple(parameters)
    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(trainable, gradient_clip_norm)
    finite = bool(torch.isfinite(norm).item())
    if finite:
        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()
    else:
        optimizer.zero_grad(set_to_none=True)
        # GradScaler requires update() after unscale_(), including skipped
        # overflows, before the next optimization attempt.
        if grad_scaler is not None:
            grad_scaler.update()
    return finite, float(norm.detach().cpu())


def train(
    config: MoETrainingConfig,
    output_dir: str | Path,
    *,
    device: str,
    base_checkpoint: str | Path = BASE_CHECKPOINT,
    train_manifest: str | Path = TRAIN_MANIFEST,
    validation_manifest: str | Path = VALIDATION_MANIFEST,
) -> dict[str, object]:
    """Train one seed and select ``best.adapter.pt`` on validation mean R/N."""

    config.validate()
    destination = Path(output_dir)
    latest_path = destination / "latest.adapter.pt"
    best_path = destination / "best.adapter.pt"
    metrics_path = destination / "metrics.jsonl"
    occupied = [path for path in (latest_path, best_path, metrics_path) if path.exists()]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite reproduction outputs: "
            + ", ".join(map(str, occupied))
        )
    destination.mkdir(parents=True, exist_ok=True)

    train_episodes = load_large_scale_split(train_manifest, "adapt_train")
    validation_episodes = load_large_scale_split(validation_manifest, "validation")
    if config.train_limit is not None:
        train_episodes = train_episodes[: config.train_limit]
    if config.validation_limit is not None:
        validation_episodes = validation_episodes[: config.validation_limit]
    if {episode.canonical_id for episode in train_episodes}.intersection(
        episode.canonical_id for episode in validation_episodes
    ):
        raise ValueError("adaptation and validation manifests overlap")

    configure_precision(config.precision_mode, device)
    seeds = config.resolved_seeds()
    torch.manual_seed(seeds["initialization"])
    np.random.seed(seeds["initialization"])
    random.seed(seeds["initialization"])
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    model, base_payload, base_hash = load_h20_base_policy(
        base_checkpoint, device=device, expected_lookahead_horizon=20
    )
    install_moe_lora(
        model,
        num_experts=config.num_experts,
        rank=config.rank,
        alpha=config.alpha,
    )
    trainable_names = configure_moe_lora_training(model)
    frozen_before = _frozen_snapshot(model)
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
        torch.amp.GradScaler(
            "cuda", enabled=True, init_scale=1024.0, growth_interval=2_000
        )
        if config.precision_mode == "fp16"
        else None
    )
    run_metadata: dict[str, object] = {
        "run_format": RUN_FORMAT,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_hash,
        "base_policy_architecture": base_payload.get("policy_architecture"),
        "base_lookahead_horizon": 20,
        "adaptation_config": asdict(config),
        "resolved_seeds": seeds,
        "adapter_kind": "moe_lora",
        "adaptation_structure": moe_lora_config(model),
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
        "train_manifest_sha256": manifest_sha256(train_manifest),
        "validation_manifest_sha256": manifest_sha256(validation_manifest),
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "trainable_parameter_names": list(trainable_names),
        "parameter_counts": parameter_counts(model),
        "selection_metric": "validation_mean_relocations_per_container",
        "training_loss_definition": (
            "equal-trajectory normalized-cost policy loss plus action entropy "
            "and weighted router balance auxiliary loss"
        ),
        "validation_precision": "fp32_greedy",
        "final_scoring_precision": "fp32",
        "test_manifest_used": False,
        "quick_check": config.quick_check,
    }
    append_jsonl(metrics_path, {"record_type": "run_start", **run_metadata})

    started = time.perf_counter()
    rng = random.Random(seeds["batch"])
    attempted_updates = applied_updates = skipped_updates = 0
    validation_started = time.perf_counter()
    validation = evaluate_fixed_validation(
        model, validation_episodes, lookahead_horizon=20, device=device
    )
    validation_seconds = time.perf_counter() - validation_started
    best_r_per_n = _mean_r_per_n(validation, validation_episodes)
    best_epoch = 0
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
            "validation_seconds": validation_seconds,
            "checkpoint_seconds": None,
            "epoch_seconds": validation_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "validation_mean_relocations_per_container": best_r_per_n,
            "validation_macro_mean_relocations": validation[
                "validation_macro_mean_relocations"
            ],
            "validation_micro_mean_relocations": validation[
                "validation_micro_mean_relocations"
            ],
            "validation_episode_relocations": validation[
                "validation_episode_relocations"
            ],
            "is_best": True,
            "stop_reason": None,
        }
    ]
    checkpoint_started = time.perf_counter()
    initial_metadata = {**run_metadata, "completed_epoch": 0, "history": history}
    save_moe_lora_adapter(
        latest_path, model, base_checkpoint_sha256=base_hash, metadata=initial_metadata
    )
    save_moe_lora_adapter(
        best_path, model, base_checkpoint_sha256=base_hash, metadata=initial_metadata
    )
    history[0]["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
    history[0]["epoch_seconds"] = validation_seconds + float(
        history[0]["checkpoint_seconds"]
    )
    history[0]["elapsed_seconds"] = time.perf_counter() - started
    append_jsonl(metrics_path, history[0])

    for epoch in range(1, config.epochs + 1):
        epoch_costs: list[float] = []
        epoch_losses: list[float] = []
        epoch_gradient_norms: list[float] = []
        epoch_decisions = epoch_clipped_updates = 0
        epoch_zero_advantage = epoch_instance_count = 0
        epoch_rpn_std_sum = 0.0
        router_decisions = router_tokens = 0
        router_weight_sums = [0.0] * config.num_experts
        router_entropy_sum = balance_decision_sum = 0.0
        expert_delta_norm_sums = [0.0] * config.num_experts
        epoch_started = time.perf_counter()
        for batch in _shape_batches(
            train_episodes,
            batch_size=config.batch_size,
            rng=rng,
            updates_per_epoch=config.updates_per_epoch,
        ):
            _seed_rollout(seeds["rollout"] + attempted_updates, device)
            optimizer.zero_grad(set_to_none=True)
            collected = collect_rollouts(
                model,
                batch,
                rollouts_per_instance=config.rollouts_per_instance,
                lookahead_horizon=20,
                device=device,
                precision_mode=config.precision_mode,
            )
            loss, routing = backward_replay(
                model,
                collected,
                decision_batch_size=config.replay_decision_batch_size,
                loss_denominator=len(batch),
                entropy_coefficient=config.entropy_coefficient,
                router_balance_coefficient=config.router_balance_coefficient,
                device=device,
                precision_mode=config.precision_mode,
                grad_scaler=grad_scaler,
            )
            attempted_updates += 1
            finite, gradient_norm = _optimizer_step(
                optimizer,
                (p for p in model.parameters() if p.requires_grad),
                gradient_clip_norm=config.gradient_clip_norm,
                grad_scaler=grad_scaler,
            )
            applied_updates += int(finite)
            skipped_updates += int(not finite)
            if finite:
                epoch_gradient_norms.append(gradient_norm)
                epoch_clipped_updates += int(gradient_norm > config.gradient_clip_norm)
            epoch_losses.append(loss)
            epoch_costs.extend(
                torch.as_tensor(collected["costs"]).reshape(-1).cpu().tolist()
            )
            epoch_decisions += int(collected["decision_count"])
            epoch_zero_advantage += int(collected["zero_advantage_instances"])
            instances = int(collected["instance_count"])
            epoch_instance_count += instances
            epoch_rpn_std_sum += float(collected["mean_rollout_r_per_n_std"]) * instances
            chunk_decisions = int(routing["decision_count"])
            router_decisions += chunk_decisions
            router_tokens += int(routing["token_count"])
            router_weight_sums = [
                left + float(right)
                for left, right in zip(
                    router_weight_sums, routing["router_weight_sums"], strict=True
                )
            ]
            router_entropy_sum += float(routing["router_entropy_sum"])
            balance_decision_sum += chunk_decisions * float(
                routing["balance_loss_weighted_sum"]
            )
            expert_delta_norm_sums = [
                left + float(right)
                for left, right in zip(
                    expert_delta_norm_sums,
                    routing["expert_delta_norm_sums"],
                    strict=True,
                )
            ]

        training_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        validation = evaluate_fixed_validation(
            model, validation_episodes, lookahead_horizon=20, device=device
        )
        validation_seconds = time.perf_counter() - validation_started
        validation_r_per_n = _mean_r_per_n(validation, validation_episodes)
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
            "epoch_mean_sampled_relocations": float(np.mean(epoch_costs)),
            "epoch_mean_normalized_loss": float(np.mean(epoch_losses)),
            "epoch_router_mean_weights": [
                value / router_tokens for value in router_weight_sums
            ],
            "epoch_router_mean_entropy": router_entropy_sum / router_tokens,
            "epoch_router_balance_loss": balance_decision_sum / router_decisions,
            "epoch_expert_mean_delta_norms": [
                value / router_tokens for value in expert_delta_norm_sums
            ],
            "epoch_decisions": epoch_decisions,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "checkpoint_seconds": None,
            "epoch_seconds": training_seconds + validation_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "zero_advantage_fraction": epoch_zero_advantage
            / max(epoch_instance_count, 1),
            "mean_rollout_r_per_n_std": epoch_rpn_std_sum
            / max(epoch_instance_count, 1),
            "mean_gradient_norm": float(np.mean(epoch_gradient_norms))
            if epoch_gradient_norms
            else None,
            "gradient_clip_fraction": epoch_clipped_updates
            / max(len(epoch_losses), 1),
            "validation_mean_relocations_per_container": validation_r_per_n,
            "validation_macro_mean_relocations": validation[
                "validation_macro_mean_relocations"
            ],
            "validation_micro_mean_relocations": validation[
                "validation_micro_mean_relocations"
            ],
            "validation_episode_relocations": validation[
                "validation_episode_relocations"
            ],
            "is_best": is_best,
            "stop_reason": None,
        }
        history.append(epoch_record)
        checkpoint_metadata = {
            **run_metadata,
            "completed_epoch": epoch,
            "history": history,
        }
        checkpoint_started = time.perf_counter()
        save_moe_lora_adapter(
            latest_path,
            model,
            base_checkpoint_sha256=base_hash,
            metadata=checkpoint_metadata,
        )
        if is_best:
            save_moe_lora_adapter(
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
        append_jsonl(metrics_path, epoch_record)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "updates": applied_updates,
                    "validation_r_per_n": validation_r_per_n,
                    "best_epoch": best_epoch,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    parameters = dict(model.named_parameters())
    frozen_unchanged = all(
        name in parameters
        and not parameters[name].requires_grad
        and torch.equal(parameters[name].detach().cpu(), before)
        for name, before in frozen_before.items()
    )
    changed_trainable = [
        name
        for name, before in trainable_before.items()
        if not torch.equal(parameters[name].detach().cpu(), before)
    ]
    if not frozen_unchanged:
        raise AssertionError("frozen base parameters changed during adaptation")
    if applied_updates and not changed_trainable:
        raise AssertionError("no trainable MoE-LoRA parameter changed")

    summary = {
        "record_type": "run_complete",
        **run_metadata,
        "output": str(latest_path),
        "best_output": str(best_path),
        "metrics_output": str(metrics_path),
        "completed_epochs": config.epochs,
        "attempted_updates": attempted_updates,
        "applied_updates": applied_updates,
        "skipped_updates": skipped_updates,
        "best_epoch": best_epoch,
        "best_validation_mean_relocations_per_container": best_r_per_n,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(device)))
            if torch.cuda.is_available() and torch.device(device).type == "cuda"
            else 0
        ),
        "changed_trainable_parameter_names": changed_trainable,
        "frozen_base_unchanged": frozen_unchanged,
        "stop_reason": None,
    }
    append_jsonl(metrics_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision-mode",
        choices=("fp32", "tf32", "bf16", "fp16"),
        default="fp16",
    )
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="one tiny update for pipeline verification; not a paper result",
    )
    parser.add_argument("--base-checkpoint", default=str(BASE_CHECKPOINT))
    parser.add_argument("--train-manifest", default=str(TRAIN_MANIFEST))
    parser.add_argument("--validation-manifest", default=str(VALIDATION_MANIFEST))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = MoETrainingConfig.for_run(
        seed=args.seed,
        precision_mode=args.precision_mode,
        quick_check=args.quick_check,
    )
    summary = train(
        config,
        args.output_dir,
        device=args.device,
        base_checkpoint=args.base_checkpoint,
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
