from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.train_online_crp_scale_diverse import (
    _collect_scale_rollouts_scalar,
    _collect_scale_rollouts_vectorized,
)
from scrp.online_crp import OnlineCRPEpisode, OnlineCRPEnvironment
from scrp.scale_diverse_policy import (
    Online2DPriorityAdapter,
    ScaleDiverseRelocationPolicy,
)
from scrp.vectorized_online_crp import VectorizedOnlineCRPEnvironment


def _episodes() -> tuple[OnlineCRPEpisode, OnlineCRPEpisode]:
    common = {
        "num_stacks": 4,
        "max_tiers": 4,
        "parameter_group": "vector-test",
        "metadata": {},
    }
    return (
        OnlineCRPEpisode(
            episode_id="vector-a",
            initial_stacks=((4, 1, 6), (2, 7), (5, 3, 8), ()),
            retrieval_sequence=(1, 2, 3, 4, 5, 6, 7, 8),
            **common,
        ),
        OnlineCRPEpisode(
            episode_id="vector-b",
            initial_stacks=((8, 2), (4, 1, 7), (5, 6, 3), ()),
            retrieval_sequence=(1, 3, 2, 4, 5, 6, 7, 8),
            **common,
        ),
    )


def _assert_matches_scalar_batch(
    vector: VectorizedOnlineCRPEnvironment,
    scalars: list[OnlineCRPEnvironment],
) -> None:
    state = vector.state
    priorities = state.priorities.detach().cpu().numpy()
    legal = state.legal_destination_mask.detach().cpu().numpy()
    forbidden = state.forbidden_mask.detach().cpu().numpy()
    retrievals = state.retrieval_count.detach().cpu().numpy()
    relocations = state.relocation_count.detach().cpu().numpy()
    crane_positions = state.crane_position.detach().cpu().numpy()
    crane_times = state.crane_time.detach().cpu().numpy()
    terminated = state.terminated.detach().cpu().numpy()

    for row, scalar in enumerate(scalars):
        expected = Online2DPriorityAdapter(scalar.max_tiers).build(scalar.state)
        np.testing.assert_array_equal(priorities[row], expected)
        expected_legal = np.zeros(vector.num_stacks, dtype=np.bool_)
        expected_legal[list(scalar.legal_destinations())] = True
        np.testing.assert_array_equal(legal[row], expected_legal)
        np.testing.assert_array_equal(forbidden[row], ~expected_legal)
        assert int(retrievals[row]) == scalar.state.retrieval_count
        assert int(relocations[row]) == scalar.relocation_count
        expected_position = -1 if scalar.crane_position is None else scalar.crane_position
        assert int(crane_positions[row]) == expected_position
        assert float(crane_times[row]) == pytest.approx(scalar.crane_time)
        assert bool(terminated[row]) == scalar.state.terminated


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_vectorized_transitions_match_scalar_environment_step_by_step(
    device: str,
) -> None:
    episodes = _episodes()
    rollouts = 3
    lookahead = 3
    vector = VectorizedOnlineCRPEnvironment(
        episodes,
        rollouts_per_episode=rollouts,
        lookahead_horizon=lookahead,
        device=device,
    )
    scalars = [
        OnlineCRPEnvironment(episode, lookahead_horizon=lookahead)
        for episode in episodes
        for _ in range(rollouts)
    ]

    assert vector.priorities.device.type == device
    assert vector.episode_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert vector.rollout_indices.tolist() == [0, 1, 2, 0, 1, 2]
    _assert_matches_scalar_batch(vector, scalars)

    transition = 0
    while not all(environment.state.terminated for environment in scalars):
        actions = torch.zeros(vector.batch_size, dtype=torch.int64, device=device)
        active_before = []
        for row, environment in enumerate(scalars):
            if environment.state.terminated:
                continue
            active_before.append(row)
            legal = environment.legal_destinations()
            destination = legal[(transition + row) % len(legal)]
            actions[row] = destination
            environment.step(destination)

        _, rewards, done = vector.step(actions)
        expected_rewards = torch.zeros(vector.batch_size, dtype=torch.float32)
        expected_rewards[active_before] = -1.0
        torch.testing.assert_close(rewards.cpu(), expected_rewards)
        assert done.cpu().tolist() == [
            environment.state.terminated for environment in scalars
        ]
        _assert_matches_scalar_batch(vector, scalars)
        transition += 1
        assert transition < 1_000

    assert vector.relocation_count.cpu().tolist() == [
        environment.relocation_count for environment in scalars
    ]


def test_online_observation_reveals_current_plus_h_future_targets() -> None:
    base = OnlineCRPEpisode(
        episode_id="window-a",
        num_stacks=3,
        max_tiers=3,
        initial_stacks=((1, 3), (2, 4), ()),
        retrieval_sequence=(1, 2, 3, 4),
        parameter_group="window",
        metadata={},
    )
    swapped_unknowns = OnlineCRPEpisode(
        episode_id="window-b",
        num_stacks=3,
        max_tiers=3,
        initial_stacks=((1, 3), (2, 4), ()),
        retrieval_sequence=(1, 2, 4, 3),
        parameter_group="window",
        metadata={},
    )
    vector = VectorizedOnlineCRPEnvironment(
        (base, swapped_unknowns), lookahead_horizon=1
    )
    torch.testing.assert_close(vector.priorities[0], vector.priorities[1])
    torch.testing.assert_close(
        vector.priorities[0],
        torch.tensor([[1.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_vectorized_h_zero_reveals_only_current_target() -> None:
    episode = OnlineCRPEpisode(
        episode_id="h0",
        num_stacks=3,
        max_tiers=3,
        initial_stacks=((1, 3), (2, 4), ()),
        retrieval_sequence=(1, 2, 3, 4),
        parameter_group="window",
        metadata={},
    )
    vector = VectorizedOnlineCRPEnvironment((episode,), lookahead_horizon=0)
    torch.testing.assert_close(
        vector.priorities[0],
        torch.tensor([[1.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_vectorized_free_retrieval_handles_mixed_container_counts() -> None:
    common = {
        "num_stacks": 4,
        "max_tiers": 4,
        "parameter_group": "mixed-count",
        "metadata": {},
    }
    episodes = (
        OnlineCRPEpisode(
            episode_id="short",
            initial_stacks=((3, 1), (2, 4), (), ()),
            retrieval_sequence=(1, 2, 3, 4),
            **common,
        ),
        OnlineCRPEpisode(
            episode_id="long",
            initial_stacks=((5, 1), (2, 6), (3, 4), ()),
            retrieval_sequence=(1, 2, 3, 4, 5, 6),
            **common,
        ),
    )
    vector = VectorizedOnlineCRPEnvironment(
        episodes, rollouts_per_episode=2, lookahead_horizon=3
    )
    scalars = [
        OnlineCRPEnvironment(episode, lookahead_horizon=3)
        for episode in episodes
        for _ in range(2)
    ]
    _assert_matches_scalar_batch(vector, scalars)
    transitions = 0
    while not all(environment.state.terminated for environment in scalars):
        actions = torch.zeros(vector.batch_size, dtype=torch.int64)
        for row, environment in enumerate(scalars):
            if environment.state.terminated:
                continue
            destination = environment.legal_destinations()[0]
            actions[row] = destination
            environment.step(destination)
        vector.step(actions)
        _assert_matches_scalar_batch(vector, scalars)
        transitions += 1
        assert transitions < 100


def test_active_only_actions_and_illegal_actions() -> None:
    episode = OnlineCRPEpisode(
        episode_id="mixed-termination",
        num_stacks=3,
        max_tiers=3,
        initial_stacks=((1,), (2,), ()),
        retrieval_sequence=(1, 2),
        parameter_group="mixed",
        metadata={},
    )
    # Both targets are initially on top and reset therefore terminates without
    # charging a relocation.  Calling step on an all-done batch is a no-op.
    vector = VectorizedOnlineCRPEnvironment((episode,), rollouts_per_episode=2)
    assert vector.terminated.tolist() == [True, True]
    _, rewards, done = vector.step(torch.empty(0, dtype=torch.int64))
    assert rewards.tolist() == [0.0, 0.0]
    assert done.tolist() == [True, True]

    active_episode = _episodes()[0]
    active = VectorizedOnlineCRPEnvironment((active_episode,))
    source = int(active.source_stack_indices[0])
    with pytest.raises(ValueError, match="illegal destination"):
        active.step(torch.tensor([source]))


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_vectorized_rollout_collection_matches_scalar_replay_data(
    device: str,
) -> None:
    policy = ScaleDiverseRelocationPolicy(
        lookahead_horizon=3,
        embed_dim=16,
        num_encoder_layers=1,
        num_heads=4,
        ffn_dim=32,
        validate_inputs=False,
    ).to(device)

    def reseed() -> None:
        torch.manual_seed(20260901)
        if device == "cuda":
            torch.cuda.manual_seed_all(20260901)

    reseed()
    scalar = _collect_scale_rollouts_scalar(
        policy,
        _episodes(),
        rollouts_per_instance=3,
        lookahead_horizon=3,
        device=device,
    )
    reseed()
    vectorized = _collect_scale_rollouts_vectorized(
        policy,
        _episodes(),
        rollouts_per_instance=3,
        lookahead_horizon=3,
        device=device,
        validate_actions=True,
    )

    for key in (
        "priorities",
        "forbidden",
        "actions",
        "trajectory_indices",
        "normalized_costs",
        "trajectory_lengths",
        "costs",
    ):
        actual = vectorized[key].detach().cpu().numpy()
        np.testing.assert_allclose(actual, scalar[key], rtol=0.0, atol=1e-6)
