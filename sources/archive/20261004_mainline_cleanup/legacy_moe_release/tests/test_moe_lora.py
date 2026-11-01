from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scrp.moe_lora import (
    MoELoRALinear,
    capture_moe_lora_routing,
    configure_moe_lora_training,
    install_moe_lora,
    load_moe_lora_adapter,
    moe_lora_state_dict,
    save_moe_lora_adapter,
)
from scrp.scale_diverse_policy import ScaleDiverseRelocationPolicy


def _policy(*, embed_dim: int = 16, ffn_dim: int = 32):
    torch.manual_seed(17)
    return ScaleDiverseRelocationPolicy(
        lookahead_horizon=20,
        embed_dim=embed_dim,
        num_encoder_layers=1,
        num_heads=4,
        ffn_dim=ffn_dim,
        clip_constant=5.0,
    )


def _inputs():
    priorities = torch.tensor(
        [[[1, -1, 0], [2, -1, 0], [-1, 0, 0], [0, 0, 0]]],
        dtype=torch.float32,
    )
    forbidden = torch.tensor([[True, False, False, False]])
    return priorities, forbidden


def test_zero_delta_full_policy_and_default_trainable_count() -> None:
    policy = _policy(embed_dim=128, ffn_dim=512).eval()
    priorities, forbidden = _inputs()
    with torch.inference_mode():
        expected = policy.action_log_probabilities(priorities, forbidden)
    install_moe_lora(policy, num_experts=2, rank=8, alpha=16)
    names = configure_moe_lora_training(policy)
    with torch.inference_mode():
        actual = policy.action_log_probabilities(priorities, forbidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sum(p.numel() for p in policy.parameters() if p.requires_grad) == 11_266
    assert set(names) == {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }


def test_dense_expert_formula_and_input_dependent_routing() -> None:
    base = nn.Linear(5, 3)
    layer = MoELoRALinear.from_linear(
        base, num_experts=2, rank=2, alpha=4, dropout=0
    )
    with torch.no_grad():
        layer.experts[0].lora_B.fill_(0.2)
        layer.experts[1].lora_B.fill_(-0.1)
        layer.router.weight.zero_()
        layer.router.bias.zero_()
        layer.router.weight[0, 0] = 2.0
        layer.router.weight[1, 0] = -2.0
    inputs = torch.tensor([[1.0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0]])
    routing = torch.softmax(F.linear(inputs, layer.router.weight, layer.router.bias), -1)
    deltas = torch.stack(
        [expert(inputs) * layer.scaling for expert in layer.experts], dim=1
    )
    expected = F.linear(inputs, layer.weight, layer.bias) + (
        routing.unsqueeze(-1) * deltas
    ).sum(dim=1)
    torch.testing.assert_close(layer(inputs), expected)
    assert not torch.equal(routing[0], routing[1])


def test_multistep_gradients_reach_experts_and_router() -> None:
    policy = _policy()
    layer = install_moe_lora(policy, num_experts=2, rank=2, alpha=4)
    configure_moe_lora_training(policy)
    optimizer = torch.optim.SGD(
        (parameter for parameter in policy.parameters() if parameter.requires_grad), lr=0.2
    )
    inputs = torch.randn(3, 4, 32)
    optimizer.zero_grad(set_to_none=True)
    layer(inputs).square().mean().backward()
    assert all(expert.lora_B.grad is not None for expert in layer.experts)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    layer(inputs).square().mean().backward()
    assert layer.router.weight.grad is not None
    assert torch.isfinite(layer.router.weight.grad).all()
    assert layer.router.weight.grad.abs().sum() > 0
    assert all(
        expert.lora_A.grad is not None
        and torch.isfinite(expert.lora_A.grad).all()
        and expert.lora_A.grad.abs().sum() > 0
        for expert in layer.experts
    )


def test_routing_capture_is_scoped_and_cleared_on_error() -> None:
    policy = _policy()
    layer = install_moe_lora(policy, num_experts=2, rank=2, alpha=4)
    priorities, forbidden = _inputs()
    with capture_moe_lora_routing(policy) as capture:
        policy.action_log_probabilities(priorities, forbidden)
        assert capture.probabilities and capture.probabilities[0].requires_grad
    assert layer._routing_capture is None
    assert capture.probabilities == []
    with pytest.raises(RuntimeError, match="boom"):
        with capture_moe_lora_routing(policy):
            policy.action_log_probabilities(priorities, forbidden)
            raise RuntimeError("boom")
    assert layer._routing_capture is None


def test_round_trip_and_malformed_load_is_atomic(tmp_path) -> None:
    base = _policy()
    base_state = deepcopy(base.state_dict())
    adapted = deepcopy(base)
    install_moe_lora(adapted, num_experts=2, rank=2, alpha=4)
    configure_moe_lora_training(adapted)
    with torch.no_grad():
        for parameter in adapted.parameters():
            if parameter.requires_grad:
                parameter.add_(0.01)
    path = save_moe_lora_adapter(
        tmp_path / "adapter.pt",
        adapted,
        base_checkpoint_sha256="base-hash",
        metadata={"purpose": "test"},
    )
    restored = _policy()
    restored.load_state_dict(base_state)
    load_moe_lora_adapter(
        restored, path, expected_base_checkpoint_sha256="base-hash"
    )
    for name, value in moe_lora_state_dict(adapted).items():
        torch.testing.assert_close(moe_lora_state_dict(restored)[name], value)

    malformed = torch.load(path, weights_only=False)
    malformed["adapter_state_dict"].pop(next(iter(malformed["adapter_state_dict"])))
    bad_path = tmp_path / "bad.pt"
    torch.save(malformed, bad_path)
    untouched = _policy()
    with pytest.raises(ValueError, match="state mismatch"):
        load_moe_lora_adapter(untouched, bad_path)
    assert type(untouched.stack_projection[2]) is nn.Linear


def test_interrupted_save_preserves_existing_checkpoint(tmp_path, monkeypatch) -> None:
    policy = _policy()
    install_moe_lora(policy, num_experts=2, rank=2, alpha=4)
    destination = tmp_path / "adapter.pt"
    previous = b"known-good-checkpoint"
    destination.write_bytes(previous)

    def interrupted_save(payload, stream) -> None:
        del payload
        stream.write(b"partial-new-checkpoint")
        stream.flush()
        raise OSError("injected write interruption")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="injected write interruption"):
        save_moe_lora_adapter(
            destination,
            policy,
            base_checkpoint_sha256="base-hash",
            metadata={"purpose": "atomic-save-test"},
        )
    assert destination.read_bytes() == previous
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression")
def test_cuda_autocast_keeps_router_and_final_scores_fp32() -> None:
    policy = _policy().cuda().eval()
    priorities, forbidden = _inputs()
    priorities = priorities.cuda()
    forbidden = forbidden.cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        baseline = policy.action_log_probabilities(priorities, forbidden)
    install_moe_lora(policy, num_experts=2, rank=2, alpha=4)
    with torch.inference_mode(), capture_moe_lora_routing(policy) as capture:
        with torch.autocast("cuda", dtype=torch.float16):
            adapted = policy.action_log_probabilities(priorities, forbidden)
        routing_dtype = capture.probabilities[0].dtype
    assert routing_dtype == torch.float32
    assert adapted.dtype == torch.float32
    legal = ~forbidden
    torch.testing.assert_close(adapted[legal], baseline[legal], rtol=0, atol=2e-5)
    assert torch.equal(adapted.argmax(dim=1), baseline.argmax(dim=1))
