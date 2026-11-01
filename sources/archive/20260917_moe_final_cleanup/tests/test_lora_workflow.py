from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from experiments.adapt_lora import AdaptationConfig, adapt, build_parser
from experiments.evaluate_lora_adaptation import H20AdaptationSelector
from experiments.train_online_crp_scale_diverse import ScaleDiverseOnlineTrainingConfig
from experiments.verify_lora_invariants import _comparison, verify
from scrp.adaptation_data import LARGE_SCALE_DOMAIN, generate_target_domain
from scrp.lora import LORA_ADAPTER_FORMAT, load_lora_adapter
from scrp.online_crp import OnlineCRPEnvironment, save_episode_manifest
from scrp.scale_diverse_policy import (
    SCALE_DIVERSE_POLICY_V1,
    Online2DPriorityAdapter,
    ScaleDiverseRelocationPolicy,
)


def _small_policy() -> ScaleDiverseRelocationPolicy:
    return ScaleDiverseRelocationPolicy(
        lookahead_horizon=20, embed_dim=16, num_encoder_layers=1,
        num_heads=4, ffn_dim=32, clip_constant=5.0,
    )


def test_generic_lora_training_save_reload_and_evaluation(tmp_path) -> None:
    torch.manual_seed(17)
    model = _small_policy()
    base_path = tmp_path / "base.pt"
    base_config = ScaleDiverseOnlineTrainingConfig(
        lookahead_horizon=20, embed_dim=16, num_encoder_layers=1,
        num_heads=4, ffn_dim=32, clip_constant=5.0,
    )
    torch.save({
        "policy_architecture": SCALE_DIVERSE_POLICY_V1,
        "online_crp_training_config": asdict(base_config),
        "model_state_dict": model.state_dict(),
    }, base_path)
    domain = LARGE_SCALE_DOMAIN.name
    train = generate_target_domain(
        domain, split="adapt_train", episode_count=2, master_seed=41
    )
    validation = generate_target_domain(
        domain, split="validation", episode_count=2, master_seed=42,
        excluded_canonical_ids={episode.canonical_id for episode in train},
    )
    train_path = save_episode_manifest(train, tmp_path / "train.json")
    validation_path = save_episode_manifest(validation, tmp_path / "validation.json")
    output = tmp_path / "adapter.pt"
    summary = adapt(
        AdaptationConfig(
            domain=domain, target_modules=("stack_projection.2",),
            rank=2, alpha=4, epochs=1, updates_per_epoch=1,
            batch_size=2, rollouts_per_instance=2, precision_mode="fp32",
            replay_decision_batch_size=128,
        ),
        base_checkpoint=base_path, train_manifest=train_path,
        validation_manifest=validation_path, output=output,
    )
    assert summary["applied_updates"] == 1
    assert summary["skipped_updates"] == 0
    assert summary["frozen_base_unchanged"]
    assert summary["test_manifest_used"] is False
    assert summary["changed_trainable_parameter_names"]
    payload = torch.load(output, weights_only=False)
    assert payload["format"] == LORA_ADAPTER_FORMAT
    assert set(payload["adapter_state_dict"]) == {
        "stack_projection.2.lora_A", "stack_projection.2.lora_B"
    }
    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in torch.load(base_path, weights_only=False)["model_state_dict"].items()
    )
    checks = verify(
        base_checkpoint=base_path, validation_manifest=validation_path,
        domain=domain, adapter=output, device="cpu", limit=None,
    )
    assert checks["passed"]
    assert checks["adapter_round_trip"]["exact_tensor_equality"]
    assert checks["zero_delta_vs_base"]["exact_tensor_equality"]
    environment = OnlineCRPEnvironment(validation[0], lookahead_horizon=20)
    priorities = Online2DPriorityAdapter(validation[0].max_tiers).build(environment.state)
    forbidden = torch.ones(validation[0].num_stacks, dtype=torch.bool)
    forbidden[list(environment.legal_destinations())] = False
    actions = []
    for merge in (False, True):
        selector = H20AdaptationSelector(
            base_path, adaptation_checkpoint=output, merge_lora_for_inference=merge,
        )
        actions.append(selector(
            priorities, forbidden.numpy(), None,
            validation[0].num_stacks, validation[0].max_tiers,
        ))
    assert actions[0] == actions[1]
    assert not forbidden[actions[0]]


def test_adaptation_cli_requires_targets_and_rejects_retired_mode() -> None:
    parser = build_parser()
    required = [
        "--base-checkpoint", "base.pt", "--train-manifest", "train.json",
        "--validation-manifest", "val.json", "--domain", LARGE_SCALE_DOMAIN.name,
        "--output", "adapter.pt",
    ]
    with pytest.raises(SystemExit) as missing:
        parser.parse_args(required)
    assert missing.value.code == 2
    explicit = required + ["--target-modules", "stack_projection.2"]
    assert parser.parse_args(explicit).target_modules == ["stack_projection.2"]
    with pytest.raises(SystemExit) as retired:
        parser.parse_args(explicit + ["--mode", "query_lora"])
    assert retired.value.code == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_verifier_rejects_nonfinite_legal_values_and_changed_mask(bad) -> None:
    first = torch.tensor([[float("-inf"), -0.5, -1.0]])
    second = first.clone()
    second[0, 1] = bad
    assert not _comparison([first], [second])["within_numerical_tolerance"]


def test_generic_loader_rejects_retired_checkpoint_format(tmp_path) -> None:
    checkpoint = tmp_path / "retired.pt"
    torch.save({"format": "mijnrl_post_tanh_residual_v1"}, checkpoint)
    with pytest.raises(ValueError, match="not a MijnRL LoRA adapter"):
        load_lora_adapter(_small_policy(), checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression")
def test_final_scoring_stays_fp32_inside_cuda_autocast(monkeypatch) -> None:
    policy = _small_policy().cuda().eval()
    priorities = torch.tensor(
        [[[1, -1, 0], [2, -1, 0], [-1, 0, 0], [0, 0, 0]]],
        dtype=torch.float32, device="cuda",
    )
    forbidden = torch.tensor([[True, False, False, False]], device="cuda")
    original_einsum = torch.einsum
    observed_dtypes = []

    def record_einsum(*args, **kwargs):
        result = original_einsum(*args, **kwargs)
        if args[0] == "bd,bsd->bs":
            observed_dtypes.append(result.dtype)
        return result

    monkeypatch.setattr(torch, "einsum", record_einsum)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        logits = policy.action_log_probabilities(priorities, forbidden)
    assert observed_dtypes == [torch.float32]
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits[~forbidden]).all()
    assert torch.isneginf(logits[forbidden]).all()
