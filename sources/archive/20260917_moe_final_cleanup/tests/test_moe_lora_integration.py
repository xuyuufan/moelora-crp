from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from experiments.adapt_lora import AdaptationConfig, _apply_optimizer_update, adapt
from experiments.evaluate_lora_adaptation import H20AdaptationSelector
from experiments.plot_adaptation_live import (
    load_adaptation_epochs,
    update_live_adaptation_curve,
    watch,
)
from experiments.smoke_moe_lora import smoke
from experiments.train_online_crp_scale_diverse import (
    ScaleDiverseOnlineTrainingConfig,
    _backward_replay,
)
from experiments.verify_lora_invariants import verify
from scrp.adaptation_data import LARGE_SCALE_DOMAIN, generate_target_domain
from scrp.moe_lora import (
    MOE_LORA_ADAPTER_FORMAT,
    configure_moe_lora_training,
    install_moe_lora,
)
from scrp.online_crp import save_episode_manifest
from scrp.scale_diverse_policy import (
    SCALE_DIVERSE_POLICY_V1,
    ScaleDiverseRelocationPolicy,
)


def _policy():
    torch.manual_seed(19)
    return ScaleDiverseRelocationPolicy(
        lookahead_horizon=20,
        embed_dim=16,
        num_encoder_layers=1,
        num_heads=4,
        ffn_dim=32,
        clip_constant=5.0,
    )


def _collected(decisions: int = 3):
    priorities = np.asarray(
        [
            [[1, -1, 0], [2, -1, 0], [-1, 0, 0], [0, 0, 0]],
            [[2, -1, 0], [1, -1, 0], [-1, 0, 0], [0, 0, 0]],
            [[1, 2, 0], [-1, 0, 0], [3, -1, 0], [0, 0, 0]],
        ][:decisions],
        dtype=np.float32,
    )
    return {
        "priorities": priorities,
        "forbidden": np.zeros((decisions, 4), dtype=np.bool_),
        "actions": np.arange(decisions, dtype=np.int64) % 4,
        "trajectory_indices": np.arange(decisions, dtype=np.int64),
        "normalized_costs": np.zeros(decisions, dtype=np.float32),
        "trajectory_lengths": np.ones(decisions, dtype=np.float32),
    }


def test_router_balance_diagnostics_and_tail_weighting() -> None:
    policy = _policy()
    layer = install_moe_lora(policy, num_experts=2, rank=2, alpha=4)
    configure_moe_lora_training(policy)
    _, uniform = _backward_replay(
        policy,
        _collected(),
        decision_batch_size=2,
        loss_denominator=3,
        entropy_coefficient=0,
        device="cpu",
        router_balance_coefficient=0.1,
        return_diagnostics=True,
    )
    assert uniform["decision_count"] == 3
    assert uniform["token_count"] == 12
    assert uniform["balance_loss_weighted_sum"] == pytest.approx(0.0, abs=1e-7)
    assert uniform["router_weight_sums"] == pytest.approx([6.0, 6.0])

    policy.zero_grad(set_to_none=True)
    with torch.no_grad():
        layer.router.bias.copy_(torch.tensor([3.0, -3.0]))
    _, biased = _backward_replay(
        policy,
        _collected(),
        decision_batch_size=2,
        loss_denominator=3,
        entropy_coefficient=0,
        device="cpu",
        router_balance_coefficient=0.1,
        return_diagnostics=True,
    )
    assert biased["balance_loss_weighted_sum"] > 1.0
    assert layer.router.bias.grad is not None
    assert torch.isfinite(layer.router.bias.grad).all()
    assert layer.router.bias.grad.abs().sum() > 0


def test_non_moe_rejects_nonzero_router_balance() -> None:
    with pytest.raises(ValueError, match="requires an installed"):
        _backward_replay(
            _policy(),
            _collected(2),
            decision_batch_size=2,
            loss_denominator=2,
            entropy_coefficient=0,
            device="cpu",
            router_balance_coefficient=0.1,
        )


def test_nonfinite_scaled_update_advances_scaler_and_recovers() -> None:
    class StatefulScaler:
        def __init__(self) -> None:
            self.awaiting_update = False
            self.update_calls = 0

        def unscale_(self, optimizer) -> None:
            del optimizer
            if self.awaiting_update:
                raise RuntimeError("unscale_() has already been called")
            self.awaiting_update = True

        def step(self, optimizer) -> None:
            if not self.awaiting_update:
                raise RuntimeError("step() called before unscale_()")
            optimizer.step()

        def update(self) -> None:
            if not self.awaiting_update:
                raise RuntimeError("update() called without unscale_()")
            self.awaiting_update = False
            self.update_calls += 1

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    scaler = StatefulScaler()
    parameter.grad = torch.tensor([float("inf")])
    applied, norm = _apply_optimizer_update(
        optimizer,
        (parameter,),
        gradient_clip_norm=1.0,
        grad_scaler=scaler,
    )
    assert not applied
    assert not np.isfinite(norm)
    assert scaler.update_calls == 1

    parameter.grad = torch.tensor([1.0])
    applied, norm = _apply_optimizer_update(
        optimizer,
        (parameter,),
        gradient_clip_norm=1.0,
        grad_scaler=scaler,
    )
    assert applied
    assert np.isfinite(norm)
    assert scaler.update_calls == 2
    assert parameter.item() == pytest.approx(0.9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression")
def test_real_grad_scaler_recovers_after_overflow() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=8.0, growth_interval=2_000
    )

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter * torch.tensor(float("inf"), device="cuda")).backward()
    initial_scale = scaler.get_scale()
    applied, _ = _apply_optimizer_update(
        optimizer,
        (parameter,),
        gradient_clip_norm=1.0,
        grad_scaler=scaler,
    )
    assert not applied
    assert scaler.get_scale() < initial_scale

    optimizer.zero_grad(set_to_none=True)
    before = parameter.detach().clone()
    scaler.scale(parameter.square()).backward()
    applied, norm = _apply_optimizer_update(
        optimizer,
        (parameter,),
        gradient_clip_norm=1.0,
        grad_scaler=scaler,
    )
    assert applied
    assert np.isfinite(norm)
    assert not torch.equal(parameter, before)


def test_moe_adaptation_save_reload_verify_and_merge_rejection(tmp_path) -> None:
    model = _policy()
    base_config = ScaleDiverseOnlineTrainingConfig(
        lookahead_horizon=20,
        embed_dim=16,
        num_encoder_layers=1,
        num_heads=4,
        ffn_dim=32,
        clip_constant=5.0,
    )
    base_path = tmp_path / "base.pt"
    torch.save(
        {
            "policy_architecture": SCALE_DIVERSE_POLICY_V1,
            "online_crp_training_config": asdict(base_config),
            "model_state_dict": model.state_dict(),
        },
        base_path,
    )
    domain = LARGE_SCALE_DOMAIN.name
    train = generate_target_domain(
        domain, split="adapt_train", episode_count=2, master_seed=71
    )
    validation = generate_target_domain(
        domain,
        split="validation",
        episode_count=2,
        master_seed=72,
        excluded_canonical_ids={episode.canonical_id for episode in train},
    )
    train_path = save_episode_manifest(train, tmp_path / "train.json")
    validation_path = save_episode_manifest(validation, tmp_path / "validation.json")
    output = tmp_path / "moe.adapter.pt"
    summary = adapt(
        AdaptationConfig(
            domain=domain,
            target_modules=("stack_projection.2",),
            adapter_kind="moe_lora",
            num_experts=2,
            rank=2,
            alpha=4,
            router_balance_coefficient=0.01,
            epochs=1,
            updates_per_epoch=2,
            max_updates=2,
            batch_size=2,
            rollouts_per_instance=2,
            precision_mode="fp32",
            replay_decision_batch_size=64,
        ),
        base_checkpoint=base_path,
        train_manifest=train_path,
        validation_manifest=validation_path,
        output=output,
    )
    assert summary["applied_updates"] == 2
    assert summary["frozen_base_unchanged"]
    assert torch.load(output, weights_only=False)["format"] == MOE_LORA_ADAPTER_FORMAT
    checks = verify(
        base_checkpoint=base_path,
        validation_manifest=validation_path,
        domain=domain,
        adapter=output,
        device="cpu",
        limit=None,
    )
    assert checks["passed"]
    assert checks["merge_supported"] is False
    H20AdaptationSelector(base_path, adaptation_checkpoint=output)
    with pytest.raises(ValueError, match="cannot be merged"):
        H20AdaptationSelector(
            base_path,
            adaptation_checkpoint=output,
            merge_lora_for_inference=True,
        )


def test_smoke_refuses_existing_output_root(tmp_path) -> None:
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        smoke(base_checkpoint="unused.pt", output_root=tmp_path)


def test_live_adaptation_curve_accepts_epoch_zero_and_router_metrics(tmp_path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    records = [
        {"record_type": "run_start", "adapter_kind": "moe_lora"},
        {
            "record_type": "epoch",
            "epoch": 0,
            "validation_mean_relocations_per_container": 1.1,
            "epoch_mean_sampled_relocations": None,
            "epoch_mean_normalized_loss": None,
            "epoch_router_mean_entropy": None,
            "epoch_router_mean_weights": None,
        },
        {
            "record_type": "epoch",
            "epoch": 1,
            "validation_mean_relocations_per_container": 1.0,
            "epoch_mean_sampled_relocations": 120.0,
            "epoch_mean_normalized_loss": -0.2,
            "epoch_router_mean_entropy": 0.68,
            "epoch_router_mean_weights": [0.52, 0.48],
        },
    ]
    metrics.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output = update_live_adaptation_curve(
        load_adaptation_epochs(metrics), tmp_path / "live.png"
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_live_adaptation_curve_waits_for_first_completed_epoch(
    tmp_path, monkeypatch
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    output = tmp_path / "live.png"
    metrics.write_text(
        json.dumps({"record_type": "run_start", "adapter_kind": "lora"}) + "\n",
        encoding="utf-8",
    )
    running = iter((True, False))

    monkeypatch.setattr(
        "experiments.plot_adaptation_live._process_is_running",
        lambda _pid: next(running),
    )

    def complete_first_epoch(_seconds: float) -> None:
        epoch = {
            "record_type": "epoch",
            "epoch": 0,
            "validation_mean_relocations_per_container": 1.0,
        }
        with metrics.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch) + "\n")

    monkeypatch.setattr(
        "experiments.plot_adaptation_live.time.sleep", complete_first_epoch
    )

    watch(metrics, output, training_pid=123, poll_seconds=0.01)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
