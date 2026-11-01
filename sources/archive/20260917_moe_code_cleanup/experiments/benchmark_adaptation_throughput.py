"""Measure H=20 generic LoRA or dynamic MoE-LoRA update throughput."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Sequence

import torch

from experiments.adapt_lora import _shape_batches
from experiments.train_online_crp_scale_diverse import (
    LOSS_WEIGHTING_EQUAL_TRAJECTORY,
    _backward_replay,
    _collect_scale_rollouts,
    _configure_precision,
)
from scrp.adaptation import load_h20_base_policy
from scrp.adaptation_data import TARGET_DOMAINS, load_and_validate_target_domain
from scrp.lora import configure_trainable_parameters, install_lora, parameter_counts
from scrp.moe_lora import (
    MOE_LORA_TARGET_MODULE,
    configure_moe_lora_training,
    install_moe_lora,
)


def _synchronize_if_cuda(device: str) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def benchmark_lora(
    *,
    adapter_kind: str,
    target_modules: Sequence[str],
    num_experts: int,
    router_balance_coefficient: float,
    rank: int,
    alpha: float,
    base_checkpoint: str,
    episodes,
    device: str,
    precision_mode: str,
    batch_size: int,
    rollouts_per_instance: int,
    replay_decision_batch_size: int,
    learning_rate: float,
    warmup_updates: int,
    measured_updates: int,
    seed: int,
) -> dict[str, object]:
    model, _, _ = load_h20_base_policy(base_checkpoint, device=device)
    torch.manual_seed(seed)
    if adapter_kind == "moe_lora":
        if tuple(target_modules) != (MOE_LORA_TARGET_MODULE,):
            raise ValueError("MoE-LoRA throughput requires its fixed target module")
        install_moe_lora(
            model, num_experts=num_experts, rank=rank, alpha=alpha, dropout=0
        )
        configure_moe_lora_training(model)
    elif adapter_kind == "lora":
        if router_balance_coefficient != 0:
            raise ValueError("generic LoRA does not support router balancing")
        install_lora(model, rank=rank, alpha=alpha, dropout=0, target_modules=target_modules)
        configure_trainable_parameters(model)
    else:
        raise ValueError("adapter_kind must be lora or moe_lora")
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True) if precision_mode == "fp16" else None
    batches = _shape_batches(
        episodes,
        batch_size=batch_size,
        rng=random.Random(seed),
        updates_per_epoch=warmup_updates + measured_updates,
    )
    elapsed: list[float] = []
    decisions = 0
    for update_index, batch in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        _synchronize_if_cuda(device)
        if update_index == warmup_updates and torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        started = time.perf_counter()
        collected = _collect_scale_rollouts(
            model,
            batch,
            rollouts_per_instance=rollouts_per_instance,
            lookahead_horizon=20,
            device=device,
            precision_mode=precision_mode,
            environment_engine="vectorized",
            cover_first_actions=False,
        )
        _backward_replay(
            model,
            collected,
            decision_batch_size=replay_decision_batch_size,
            loss_denominator=len(batch),
            entropy_coefficient=0.002,
            device=device,
            loss_weighting=LOSS_WEIGHTING_EQUAL_TRAJECTORY,
            precision_mode=precision_mode,
            grad_scaler=scaler,
            router_balance_coefficient=router_balance_coefficient,
        )
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
        )
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        _synchronize_if_cuda(device)
        if update_index >= warmup_updates:
            elapsed.append(time.perf_counter() - started)
            decisions += int(collected["decision_count"])
    peak = (
        int(torch.cuda.max_memory_allocated(torch.device(device)))
        if torch.cuda.is_available() and torch.device(device).type == "cuda"
        else 0
    )
    return {
        "adapter_kind": adapter_kind,
        "target_modules": list(target_modules),
        "num_experts": num_experts if adapter_kind == "moe_lora" else None,
        "router_balance_coefficient": router_balance_coefficient,
        "rank": rank,
        "alpha": alpha,
        "measured_updates": len(elapsed),
        "mean_update_seconds": sum(elapsed) / len(elapsed),
        "mean_decisions_per_second": decisions / sum(elapsed),
        "peak_memory_bytes": peak,
        "parameter_counts": parameter_counts(model),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--domain", choices=tuple(TARGET_DOMAINS), required=True)
    parser.add_argument("--adapter-kind", choices=("lora", "moe_lora"), default="lora")
    parser.add_argument("--target-modules", nargs="+", required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--router-balance-coefficient", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-mode", choices=("fp32", "tf32", "bf16", "fp16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rollouts-per-instance", type=int, default=8)
    parser.add_argument("--replay-decision-batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--measured-updates", type=int, default=2)
    parser.add_argument("--maximum-hours", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.batch_size <= 0
        or args.rollouts_per_instance < 2
        or args.warmup_updates < 0
        or args.measured_updates <= 0
    ):
        raise ValueError("invalid throughput batch, rollout, or update count")
    if args.maximum_hours <= 0:
        raise ValueError("maximum_hours must be positive")
    _configure_precision(args.precision_mode, args.device)
    episodes = load_and_validate_target_domain(
        args.train_manifest, domain=args.domain, allowed_splits=("adapt_train",)
    )
    record = benchmark_lora(
        adapter_kind=args.adapter_kind,
        target_modules=args.target_modules, rank=args.rank, alpha=args.alpha,
        num_experts=args.num_experts,
        router_balance_coefficient=args.router_balance_coefficient,
        base_checkpoint=args.base_checkpoint,
        episodes=episodes,
        device=args.device,
        precision_mode=args.precision_mode,
        batch_size=args.batch_size,
        rollouts_per_instance=args.rollouts_per_instance,
        replay_decision_batch_size=args.replay_decision_batch_size,
        learning_rate=args.learning_rate,
        warmup_updates=args.warmup_updates,
        measured_updates=args.measured_updates,
        seed=args.seed,
    )
    estimated_updates = int(args.maximum_hours * 3600 // float(record["mean_update_seconds"]))
    result = {
        "domain": args.domain,
        "batch_size": args.batch_size,
        "rollouts_per_instance": args.rollouts_per_instance,
        "warmup_updates": args.warmup_updates,
        "maximum_hours": args.maximum_hours,
        "estimated_updates_in_budget": estimated_updates,
        "measurement": record,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
