from __future__ import annotations

import torch

from scrp.lora import (
    configure_trainable_parameters,
    install_lora,
    load_lora_adapter,
    merge_lora_,
    parameter_counts,
    save_lora_adapter,
)
from scrp.scale_diverse_policy import ScaleDiverseRelocationPolicy
from scrp.online_crp import OnlineCRPEpisode, OnlineCRPEnvironment
from scrp.scale_diverse_policy import Online2DPriorityAdapter


TARGETS = ("stack_projection.2", "container_embedding.2")


def _policy() -> ScaleDiverseRelocationPolicy:
    torch.manual_seed(17)
    return ScaleDiverseRelocationPolicy(
        lookahead_horizon=20,
        embed_dim=16,
        num_encoder_layers=1,
        num_heads=4,
        ffn_dim=32,
        clip_constant=5.0,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    priorities = torch.tensor(
        [[[1, -1, 0], [2, -1, 0], [-1, 0, 0], [0, 0, 0]]],
        dtype=torch.float32,
    )
    forbidden = torch.tensor([[True, False, False, False]])
    return priorities, forbidden


def test_zero_delta_lora_matches_original_logits_bit_for_bit() -> None:
    policy = _policy().eval()
    priorities, forbidden = _inputs()
    with torch.inference_mode():
        baseline = policy.action_log_probabilities(priorities, forbidden)

    install_lora(policy, target_modules=TARGETS, rank=4, alpha=8, dropout=0)
    configure_trainable_parameters(policy)
    with torch.inference_mode():
        adapted = policy.action_log_probabilities(priorities, forbidden)

    torch.testing.assert_close(adapted, baseline, rtol=0, atol=0)
    assert all(
        not parameter.requires_grad
        for name, parameter in policy.named_parameters()
        if "lora_" not in name
    )
    # Targets are 32 -> 16 and 8 -> 16 on this small policy.
    assert parameter_counts(policy)["trainable_parameters"] == 4 * (32 + 16 + 8 + 16)


def test_lora_updates_without_changing_frozen_base_weights() -> None:
    policy = _policy()
    install_lora(policy, target_modules=TARGETS, rank=2, alpha=4)
    configure_trainable_parameters(policy)
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if "lora_" not in name
    }
    lora_before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if "lora_" in name
    }
    priorities, forbidden = _inputs()
    loss = -policy.action_log_probabilities(priorities, forbidden)[0, 1]
    optimizer = torch.optim.Adam(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=1e-2,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    for name, before in base_before.items():
        torch.testing.assert_close(dict(policy.named_parameters())[name], before)
    assert any(
        not torch.equal(dict(policy.named_parameters())[name], before)
        for name, before in lora_before.items()
    )


def test_adapter_round_trip_and_merge_preserve_logits(tmp_path) -> None:
    base = _policy().eval()
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    install_lora(base, target_modules=TARGETS, rank=2, alpha=4)
    configure_trainable_parameters(base)
    with torch.no_grad():
        for _, parameter in base.named_parameters():
            if parameter.requires_grad:
                parameter.add_(0.01)
    priorities, forbidden = _inputs()
    with torch.inference_mode():
        expected = base.action_log_probabilities(priorities, forbidden)

    adapter_path = tmp_path / "adapter.pt"
    save_lora_adapter(
        adapter_path,
        base,
        base_checkpoint_sha256="base-hash",
        metadata={"seed": 3, "purpose": "test"},
    )
    reloaded = _policy().eval()
    reloaded.load_state_dict(base_state, strict=True)
    payload = load_lora_adapter(
        reloaded,
        adapter_path,
        expected_base_checkpoint_sha256="base-hash",
    )
    assert payload["lora_config"]["target_modules"] == list(TARGETS)
    with torch.inference_mode():
        round_trip = reloaded.action_log_probabilities(priorities, forbidden)
    torch.testing.assert_close(round_trip, expected, rtol=1e-6, atol=1e-7)

    merge_lora_(reloaded)
    with torch.inference_mode():
        merged = reloaded.action_log_probabilities(priorities, forbidden)
    torch.testing.assert_close(merged, expected, rtol=1e-6, atol=1e-7)


def test_h20_action_is_invariant_to_unrevealed_future_order() -> None:
    common = {
        "num_stacks": 5,
        "max_tiers": 6,
        "initial_stacks": (
            (1, 2, 3, 4, 5, 6),
            (7, 8, 9, 10, 11, 12),
            (13, 14, 15, 16, 17, 18),
            (19, 20, 21, 22, 23),
            (24,),
        ),
        "parameter_group": "future-privacy",
        "metadata": {},
    }
    base = OnlineCRPEpisode(
        episode_id="base",
        retrieval_sequence=tuple(range(1, 25)),
        **common,
    )
    changed_unknown_tail = OnlineCRPEpisode(
        episode_id="changed-tail",
        retrieval_sequence=tuple(range(1, 22)) + (23, 22, 24),
        **common,
    )
    first = OnlineCRPEnvironment(base, lookahead_horizon=20)
    second = OnlineCRPEnvironment(changed_unknown_tail, lookahead_horizon=20)
    assert first.state == second.state
    adapter = Online2DPriorityAdapter(6)
    first_tensor = torch.as_tensor(adapter.build(first.state)).unsqueeze(0)
    second_tensor = torch.as_tensor(adapter.build(second.state)).unsqueeze(0)
    torch.testing.assert_close(first_tensor, second_tensor)
    forbidden = torch.ones((1, 5), dtype=torch.bool)
    forbidden[0, list(first.legal_destinations())] = False
    policy = _policy().eval()
    with torch.inference_mode():
        first_logits = policy.action_log_probabilities(first_tensor, forbidden)
        second_logits = policy.action_log_probabilities(second_tensor, forbidden)
    torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)
