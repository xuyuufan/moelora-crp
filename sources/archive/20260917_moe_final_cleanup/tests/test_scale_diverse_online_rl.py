from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.online_crp_fair_evaluation import TorchCheckpointSelector, run_episode
from experiments.train_online_crp_scale_diverse import (
    ScaleDiverseOnlineTrainingConfig,
    train,
)
from scrp.online_crp import (
    OnlineCRPEpisode,
    OnlineCRPEnvironment,
    generate_table2_structural_sample,
    save_episode_manifest,
)
from scrp.scale_diverse_policy import (
    Online2DPriorityAdapter,
    ScaleDiverseRelocationPolicy,
)


def _lookahead_episode(sequence=(1, 2, 3, 4)) -> OnlineCRPEpisode:
    return OnlineCRPEpisode(
        episode_id="lookahead",
        num_stacks=3,
        max_tiers=3,
        initial_stacks=((1, 3), (2, 4), ()),
        retrieval_sequence=tuple(sequence),
        parameter_group="test",
        metadata={},
    )


def test_lookahead_reveals_current_plus_h_future_targets_without_leakage() -> None:
    first = OnlineCRPEnvironment(
        _lookahead_episode((1, 2, 3, 4)), lookahead_horizon=1
    )
    same_window = OnlineCRPEnvironment(
        _lookahead_episode((1, 2, 4, 3)), lookahead_horizon=1
    )
    different_window = OnlineCRPEnvironment(
        _lookahead_episode((1, 3, 2, 4)), lookahead_horizon=1
    )
    assert first.state == same_window.state
    assert first.state != different_window.state
    assert first.state.stacks == ((1, 0), (2, 0), ())


def test_h_zero_reveals_only_the_current_target() -> None:
    first = OnlineCRPEnvironment(
        _lookahead_episode((1, 2, 3, 4)), lookahead_horizon=0
    )
    changed_future = OnlineCRPEnvironment(
        _lookahead_episode((1, 3, 2, 4)), lookahead_horizon=0
    )
    assert first.state == changed_future.state
    assert first.state.stacks == ((1, 0), (0, 0), ())


def test_priority_adapter_distinguishes_unknown_containers_from_empty_tiers() -> None:
    environment = OnlineCRPEnvironment(_lookahead_episode(), lookahead_horizon=1)
    tensor = Online2DPriorityAdapter(3).build(environment.state)
    np.testing.assert_array_equal(
        tensor,
        np.asarray(((1, -1, 0), (2, -1, 0), (0, 0, 0)), dtype=np.float32),
    )


def test_policy_accepts_variable_2d_layouts_and_respects_action_mask() -> None:
    policy = ScaleDiverseRelocationPolicy(
        lookahead_horizon=1,
        embed_dim=16,
        num_encoder_layers=1,
        num_heads=4,
        ffn_dim=32,
    )
    priorities = torch.tensor([
        [[1, -1, 0], [2, -1, 0], [0, 0, 0]],
    ], dtype=torch.float32)
    forbidden = torch.tensor([[True, True, False]])
    action, log_probability = policy(priorities, forbidden, greedy=True)
    assert action.tolist() == [2]
    assert torch.isfinite(log_probability).all()

    larger = torch.tensor([
        [[1, -1, 0, 0], [2, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0]],
    ], dtype=torch.float32)
    larger_mask = torch.tensor([[True, False, False, False]])
    probabilities = policy.action_log_probabilities(larger, larger_mask)
    assert probabilities.shape == (1, 4)
    assert torch.isneginf(probabilities[0, 0])


def test_h_zero_policy_keeps_unknown_token_distinct_from_current_target() -> None:
    policy = ScaleDiverseRelocationPolicy(
        lookahead_horizon=0,
        embed_dim=16,
        num_encoder_layers=1,
        num_heads=4,
        ffn_dim=32,
    )
    assert policy.unknown_token.item() == 2.0


def test_tiny_scale_diverse_training_checkpoint_runs_in_evaluator(tmp_path: Path) -> None:
    output = tmp_path / "tiny-scale-diverse.pt"
    validation_manifest = tmp_path / "validation.json"
    save_episode_manifest(
        generate_table2_structural_sample(
            (3, 3), episode_count=1, master_seed=991
        ),
        validation_manifest,
    )
    config = ScaleDiverseOnlineTrainingConfig(
        lookahead_horizon=0,
        embed_dim=16,
        num_encoder_layers=1,
        num_heads=4,
        ffn_dim=32,
        epochs=1,
        steps_per_epoch=1,
        scales_per_step=1,
        instances_per_scale=1,
        rollouts_per_instance=2,
        seed=71,
    )
    train(
        config,
        output,
        device="cpu",
        validation_manifest=validation_manifest,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["policy_architecture"] == "scale_diverse_online_2d_v1"
    assert checkpoint["training_history"][0]["optimizer_updates"] == 1
    epoch_one = tmp_path / "tiny-scale-diverse_epochs" / "tiny-scale-diverse_epoch_001.pt"
    assert epoch_one.exists()
    epoch_one_checkpoint = torch.load(epoch_one, map_location="cpu", weights_only=False)
    assert epoch_one_checkpoint["completed_epoch"] == 1
    metrics = tmp_path / "tiny-scale-diverse_epoch_metrics.jsonl"
    metric_records = [
        json.loads(line)
        for line in metrics.read_text(encoding="utf-8").splitlines()
    ]
    assert metric_records[0]["validation_macro_reward"] is not None
    assert "S3-W3" in metric_records[0]["validation_per_scale"]
    assert len(metric_records[0]["validation_episode_relocations"]) == 1
    assert (tmp_path / "tiny-scale-diverse.best.pt").exists()
    assert (tmp_path / "tiny-scale-diverse_live_reward_curve.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )

    selector = TorchCheckpointSelector(output, device="cpu")
    result = run_episode(_lookahead_episode(), "RL", selector)
    assert result.relocations >= 0
    assert selector.lookahead_horizon == 0

    resumed_config = ScaleDiverseOnlineTrainingConfig(
        lookahead_horizon=0,
        embed_dim=16,
        num_encoder_layers=1,
        num_heads=4,
        ffn_dim=32,
        epochs=2,
        steps_per_epoch=1,
        scales_per_step=1,
        instances_per_scale=1,
        rollouts_per_instance=2,
        seed=71,
    )
    train(
        resumed_config,
        output,
        device="cpu",
        resume=output,
        validation_manifest=validation_manifest,
    )
    resumed = torch.load(output, map_location="cpu", weights_only=False)
    assert len(resumed["training_history"]) == 2
    assert resumed["training_engine"] == "batched_sampling_replay_live_v2"
    epoch_two = tmp_path / "tiny-scale-diverse_epochs" / "tiny-scale-diverse_epoch_002.pt"
    assert epoch_two.exists()
    epoch_two_checkpoint = torch.load(epoch_two, map_location="cpu", weights_only=False)
    assert len(epoch_two_checkpoint["training_history"]) == 2
    assert epoch_two_checkpoint["completed_epoch"] == 2

    weights_only_output = tmp_path / "weights-only.pt"
    weights_only_config = ScaleDiverseOnlineTrainingConfig(
        lookahead_horizon=0,
        embed_dim=16,
        num_encoder_layers=1,
        num_heads=4,
        ffn_dim=32,
        learning_rate=1e-4,
        epochs=1,
        steps_per_epoch=1,
        scales_per_step=1,
        instances_per_scale=1,
        rollouts_per_instance=2,
        replay_decision_batch_size=32,
        seed=71,
    )
    train(
        weights_only_config,
        weights_only_output,
        device="cpu",
        resume=output,
        resume_mode="weights-only",
    )
    weights_only = torch.load(
        weights_only_output, map_location="cpu", weights_only=False
    )
    assert weights_only["completed_epoch"] == 1
    assert weights_only["online_crp_training_config"]["learning_rate"] == 1e-4
    assert weights_only["optimizer_state_dict"]["param_groups"][0]["lr"] == 1e-4
