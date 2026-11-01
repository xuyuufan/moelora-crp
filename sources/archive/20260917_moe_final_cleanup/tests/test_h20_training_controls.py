from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from experiments.train_online_crp_scale_diverse import (
    LOSS_WEIGHTING_LEGACY,
    ScaleDiverseOnlineTrainingConfig,
    _backward_replay,
    _finite_or_none,
    _strict_json,
    _validate_full_resume_config,
    build_balanced_scale_schedule,
    entropy_coefficient_for_epoch,
    update_early_stopping_state,
)
from experiments.online_crp_fair_evaluation import paired_bootstrap_mean_ci
from scrp.online_crp import PAPER_TABLE2_GROUPS


def test_production_schedule_is_balanced_and_distinct() -> None:
    config = ScaleDiverseOnlineTrainingConfig(seed=17)
    schedule = build_balanced_scale_schedule(config, epoch_index=0)

    assert len(schedule) == 105
    assert all(len(step) == len(set(step)) == 4 for step in schedule)
    assert Counter(group for step in schedule for group in step) == {
        group: 20 for group in PAPER_TABLE2_GROUPS
    }
    assert schedule == build_balanced_scale_schedule(config, epoch_index=0)


def test_entropy_schedule_uses_constant_cosine_and_floor() -> None:
    config = ScaleDiverseOnlineTrainingConfig()

    assert entropy_coefficient_for_epoch(config, 1) == pytest.approx(0.002)
    assert 0.0002 < entropy_coefficient_for_epoch(config, 15) < 0.002
    assert entropy_coefficient_for_epoch(config, 30) == pytest.approx(0.0002)
    assert entropy_coefficient_for_epoch(config, 100) == pytest.approx(0.0002)


class _LogitPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

    def action_log_probabilities(
        self, priorities: torch.Tensor, forbidden: torch.Tensor
    ) -> torch.Tensor:
        del forbidden
        return torch.log_softmax(self.logits, dim=0).expand(priorities.shape[0], -1)


def test_entropy_produces_gradient_when_advantages_are_zero() -> None:
    policy = _LogitPolicy()
    collected = {
        "priorities": np.zeros((2, 1, 1), dtype=np.float32),
        "forbidden": np.zeros((2, 2), dtype=np.bool_),
        "actions": np.asarray([0, 1], dtype=np.int64),
        "trajectory_indices": np.asarray([0, 1], dtype=np.int64),
        "normalized_costs": np.zeros(2, dtype=np.float32),
        "trajectory_lengths": np.ones(2, dtype=np.float32),
    }

    loss = _backward_replay(
        policy,
        collected,
        decision_batch_size=2,
        loss_denominator=2,
        entropy_coefficient=0.01,
        device="cpu",
    )

    assert loss < 0
    assert policy.logits.grad is not None
    assert torch.linalg.vector_norm(policy.logits.grad).item() > 0


def test_replay_gives_short_and_long_trajectories_equal_total_weight() -> None:
    policy = _LogitPolicy()
    with torch.no_grad():
        policy.logits.zero_()
    collected = {
        "priorities": np.zeros((3, 1, 1), dtype=np.float32),
        "forbidden": np.zeros((3, 2), dtype=np.bool_),
        "actions": np.asarray([0, 1, 1], dtype=np.int64),
        "trajectory_indices": np.asarray([0, 1, 1], dtype=np.int64),
        "normalized_costs": np.ones(2, dtype=np.float32),
        "trajectory_lengths": np.asarray([1, 2], dtype=np.float32),
    }

    _backward_replay(
        policy,
        collected,
        decision_batch_size=3,
        loss_denominator=2,
        entropy_coefficient=0.0,
        device="cpu",
    )

    assert policy.logits.grad is not None
    torch.testing.assert_close(policy.logits.grad, torch.zeros_like(policy.logits.grad))


def test_early_stopping_cannot_trigger_before_epoch_55_and_improvement_resets() -> None:
    config = ScaleDiverseOnlineTrainingConfig(early_stopping_enabled=True)
    reference = -math.inf
    stale = 0
    stopped = False
    for epoch in range(1, 55):
        reward = -34.0 if epoch == 1 else -34.01
        reference, stale, stopped = update_early_stopping_state(
            epoch_number=epoch,
            validation_reward=reward,
            reference_reward=reference,
            epochs_without_improvement=stale,
            config=config,
        )
        assert not stopped

    reference, stale, stopped = update_early_stopping_state(
        epoch_number=55,
        validation_reward=-34.01,
        reference_reward=reference,
        epochs_without_improvement=stale,
        config=config,
    )
    assert stopped
    assert stale == 25

    reference, stale, stopped = update_early_stopping_state(
        epoch_number=40,
        validation_reward=-33.94,
        reference_reward=-34.0,
        epochs_without_improvement=9,
        config=config,
    )
    assert reference == pytest.approx(-33.94)
    assert stale == 0
    assert not stopped


def test_disabled_early_stopping_never_increments_patience() -> None:
    config = ScaleDiverseOnlineTrainingConfig(early_stopping_enabled=False)
    reference, stale, stopped = update_early_stopping_state(
        epoch_number=100,
        validation_reward=-99.0,
        reference_reward=-30.0,
        epochs_without_improvement=4,
        config=config,
    )
    assert (reference, stale, stopped) == (-30.0, 4, False)


def test_full_resume_rejects_changed_optimizer_configuration() -> None:
    config = ScaleDiverseOnlineTrainingConfig()
    checkpoint_config = asdict(config)
    checkpoint_config["learning_rate"] = 1e-4

    with pytest.raises(ValueError, match="weights-only"):
        _validate_full_resume_config(config, checkpoint_config)

    # Extending the epoch limit is explicitly compatible with full resume.
    _validate_full_resume_config(replace(config, epochs=120), asdict(config))


def test_legacy_h20_checkpoint_config_is_migrated_without_changing_semantics() -> None:
    config = ScaleDiverseOnlineTrainingConfig(
        loss_weighting=LOSS_WEIGHTING_LEGACY,
        lookahead_horizon=20,
    )
    checkpoint_config = asdict(config)
    for field in ("loss_weighting",):
        checkpoint_config.pop(field)

    _validate_full_resume_config(config, checkpoint_config)


def test_nonfinite_diagnostics_are_serialized_as_json_null() -> None:
    record = {"gradient_norm": _finite_or_none(float("nan"))}
    encoded = _strict_json(record)

    assert json.loads(encoded) == {"gradient_norm": None}
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_paired_bootstrap_interval_is_deterministic_and_supports_improvement() -> None:
    differences = [-1.0, -2.0, -1.0, -3.0, -2.0]
    first = paired_bootstrap_mean_ci(differences, samples=2_000, seed=7)
    second = paired_bootstrap_mean_ci(differences, samples=2_000, seed=7)

    assert first == second
    assert first["upper"] < 0
