"""Verify zero-delta, adapter round-trip, and supported inference invariants."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch

from scrp.adaptation import load_adapter, load_h20_base_policy
from scrp.adaptation_data import TARGET_DOMAINS, load_and_validate_target_domain
from scrp.lora import (
    LORA_ADAPTER_FORMAT,
    configure_trainable_parameters,
    install_lora,
    lora_config,
    merge_lora_,
    save_lora_adapter,
)
from scrp.moe_lora import (
    MOE_LORA_ADAPTER_FORMAT,
    MOE_LORA_TARGET_MODULE,
    configure_moe_lora_training,
    install_moe_lora,
    moe_lora_config,
    save_moe_lora_adapter,
)
from scrp.vectorized_online_crp import VectorizedOnlineCRPEnvironment


def _initial_logits(model, episodes, *, device: str):
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[(episode.num_stacks, episode.max_tiers)].append(episode)
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in grouped.values():
            environment = VectorizedOnlineCRPEnvironment(
                batch, lookahead_horizon=20, device=device, validate_actions=True
            )
            active = environment.active_indices
            if active.numel() != len(batch):
                raise ValueError("invariant manifest includes an episode with no relocation decision")
            rows.append(
                model.action_log_probabilities(
                    environment.priorities[active], environment.forbidden_mask[active]
                ).cpu()
            )
    return rows


def _comparison(left, right) -> dict[str, object]:
    if len(left) != len(right):
        raise AssertionError("comparison produced a different number of shape batches")
    finite_max = 0.0
    exact = True
    actions_match = True
    masks_match = True
    legal_values_finite = True
    for first, second in zip(left, right, strict=True):
        if first.shape != second.shape:
            raise AssertionError("comparison changed policy output shape")
        exact = exact and bool(torch.equal(first, second))
        masks_match = masks_match and bool(torch.equal(torch.isneginf(first), torch.isneginf(second)))
        legal = ~torch.isneginf(first)
        legal_values_finite = legal_values_finite and bool(
            torch.isfinite(first[legal]).all() and torch.isfinite(second[legal]).all()
        )
        if legal.any() and legal_values_finite:
            finite_max = max(finite_max, float((first[legal] - second[legal]).abs().max()))
        actions_match = actions_match and bool(torch.equal(first.argmax(dim=1), second.argmax(dim=1)))
    return {
        "exact_tensor_equality": exact,
        "masks_identical": masks_match,
        "legal_values_finite": legal_values_finite,
        "max_finite_logit_absolute_difference": finite_max,
        "greedy_actions_identical": actions_match,
        "within_numerical_tolerance": masks_match and legal_values_finite and finite_max <= 2e-5 and actions_match,
    }


def verify(
    *,
    base_checkpoint: str | Path,
    validation_manifest: str | Path,
    domain: str,
    adapter: str | Path | None,
    device: str,
    limit: int | None,
    target_modules: Sequence[str] | None = None,
    rank: int | None = None,
    alpha: float | None = None,
    adapter_kind: str | None = None,
    num_experts: int | None = None,
) -> dict[str, object]:
    episodes = load_and_validate_target_domain(
        validation_manifest, domain=domain, allowed_splits=("validation",)
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        episodes = episodes[:limit]
    base, _, base_hash = load_h20_base_policy(base_checkpoint, device=device)
    loaded = None
    if adapter is not None:
        loaded, _, _ = load_h20_base_policy(base_checkpoint, device=device)
        payload = load_adapter(
            loaded, adapter,
            expected_base_checkpoint_sha256=base_hash, map_location="cpu",
        )
        adapter_format = str(payload.get("format"))
        if adapter_format == LORA_ADAPTER_FORMAT:
            inferred_kind = "lora"
        elif adapter_format == MOE_LORA_ADAPTER_FORMAT:
            inferred_kind = "moe_lora"
        else:  # load_adapter rejects this before reaching the guard.
            raise ValueError(f"unsupported adapter format {adapter_format!r}")
        structure = (
            lora_config(loaded) if inferred_kind == "lora" else moe_lora_config(loaded)
        )
        if adapter_kind is not None and adapter_kind != inferred_kind:
            raise ValueError("adapter_kind conflicts with adapter checkpoint format")
        expected_targets = (
            structure["target_modules"]
            if inferred_kind == "lora"
            else [structure["target_module"]]
        )
        if target_modules is not None and list(target_modules) != expected_targets:
            raise ValueError("target_modules conflict with adapter configuration")
        if rank is not None and rank != int(structure["rank"]):
            raise ValueError("rank conflicts with adapter configuration")
        if alpha is not None and alpha != float(structure["alpha"]):
            raise ValueError("alpha conflicts with adapter configuration")
        if (
            num_experts is not None
            and inferred_kind == "moe_lora"
            and num_experts != int(structure["num_experts"])
        ):
            raise ValueError("num_experts conflicts with adapter configuration")
        adapter_kind = inferred_kind
    else:
        if not target_modules:
            raise ValueError("provide an adapter or explicit target_modules")
        adapter_kind = adapter_kind or "lora"
        rank = 4 if rank is None else rank
        alpha = 8.0 if alpha is None else alpha
        if adapter_kind == "moe_lora":
            if tuple(target_modules) != (MOE_LORA_TARGET_MODULE,):
                raise ValueError("MoE-LoRA verifier requires its fixed target module")
            structure = {
                "target_module": MOE_LORA_TARGET_MODULE,
                "num_experts": 2 if num_experts is None else num_experts,
                "rank": rank,
                "alpha": alpha,
                "dropout": 0.0,
            }
        elif adapter_kind == "lora":
            structure = {
                "target_modules": list(target_modules),
                "rank": rank,
                "alpha": alpha,
                "dropout": 0.0,
            }
        else:
            raise ValueError("adapter_kind must be lora or moe_lora")
    zero_delta, _, zero_hash = load_h20_base_policy(base_checkpoint, device=device)
    if base_hash != zero_hash:
        raise AssertionError("base checkpoint changed during verification")
    if adapter_kind == "moe_lora":
        install_moe_lora(
            zero_delta,
            num_experts=int(structure["num_experts"]),
            rank=int(structure["rank"]),
            alpha=float(structure["alpha"]),
            dropout=float(structure["dropout"]),
        )
        configure_moe_lora_training(zero_delta)
    else:
        install_lora(zero_delta, **structure)
        configure_trainable_parameters(zero_delta)
    result = {
        "base_checkpoint_sha256": base_hash,
        "domain": domain,
        "adapter_kind": adapter_kind,
        "adapter_config": structure,
        "validation_episodes_checked": len(episodes),
        "zero_delta_vs_base": _comparison(
            _initial_logits(base, episodes, device=device),
            _initial_logits(zero_delta, episodes, device=device),
        ),
    }
    if loaded is not None:
        original = _initial_logits(loaded, episodes, device=device)
        with tempfile.TemporaryDirectory(prefix="mijnrl_lora_verify_") as temporary:
            save_function = (
                save_moe_lora_adapter
                if adapter_kind == "moe_lora"
                else save_lora_adapter
            )
            saved = save_function(
                Path(temporary) / "adapter.pt", loaded,
                base_checkpoint_sha256=base_hash, metadata=payload.get("metadata", {}),
            )
            restored, _, _ = load_h20_base_policy(base_checkpoint, device=device)
            load_adapter(restored, saved, expected_base_checkpoint_sha256=base_hash)
            result["adapter_round_trip"] = _comparison(
                original, _initial_logits(restored, episodes, device=device)
            )
        if adapter_kind == "lora":
            merge_lora_(loaded)
            result["merge_supported"] = True
            result["adapter_before_vs_after_merge"] = _comparison(
                original, _initial_logits(loaded, episodes, device=device)
            )
        else:
            result["merge_supported"] = False
            result["merge_unsupported_reason"] = (
                "dynamic per-stack input routing cannot be represented by one static weight"
            )
    checks = [
        value for value in result.values()
        if isinstance(value, dict) and "within_numerical_tolerance" in value
    ]
    result["passed"] = all(check["within_numerical_tolerance"] for check in checks)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--domain", choices=tuple(TARGET_DOMAINS), required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-kind", choices=("lora", "moe_lora"))
    parser.add_argument("--target-modules", nargs="+")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify(
        base_checkpoint=args.base_checkpoint,
        validation_manifest=args.validation_manifest,
        domain=args.domain,
        adapter=args.adapter,
        device=args.device,
        limit=args.limit,
        target_modules=args.target_modules, rank=args.rank, alpha=args.alpha,
        adapter_kind=args.adapter_kind, num_experts=args.num_experts,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
