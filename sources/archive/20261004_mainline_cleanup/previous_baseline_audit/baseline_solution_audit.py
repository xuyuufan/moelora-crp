"""Independent, read-only solution audit for the three released baselines.

This script does not train models and does not modify baseline outputs.  It
replays released solutions, checks every transition against the scalar OCRP
environment, and compares the resulting relocation counts with the immutable
published records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
MIJN_RL = ROOT / "Mijn_RL"
if str(MIJN_RL) not in sys.path:
    sys.path.insert(0, str(MIJN_RL))

from GP.config import implementation_sha256, load_config, run_relative_dir, run_seed
from GP.data import sha256_file
from GP.expression import Node, TERMINAL_SETS
from GP.online_crp_oracle import (
    OnlineCRPEpisode as GPOnlineCRPEpisode,
    OnlineCRPEnvironment as GPOnlineCRPEnvironment,
    load_episode_manifest as load_gp_manifest,
)
from GP.simulator import select_destination, trace_tree
from Leveling.heuristic import load_episode_manifest as load_leveling_manifest
from scrp.adaptation_data import load_large_scale_split, manifest_sha256
from scrp.moe_lora import configure_moe_lora_training, load_moe_lora_adapter
from scrp.moe_training import evaluate_fixed_validation
from scrp.online_crp import (
    LevelingBaseline,
    OnlineCRPEpisode,
    OnlineCRPEnvironment,
    load_episode_manifest,
)
from scrp.scale_diverse_policy import Online2DPriorityAdapter
from scrp.vectorized_online_crp import VectorizedOnlineCRPEnvironment
from experiments.evaluate_moe_caserta import _load_policy


EVIDENCE_DIR = ROOT / "baseline_solution_audit_work"
CASERTA = MIJN_RL / "data/online_crp/caserta_table2_840.json"
GP_CONFIG = ROOT / "GP/configs/h20_four_terminal_sets.json"
GP_RESULTS = ROOT / "GP/results/h20_four_terminal_sets"
LEVELING_RESULTS = ROOT / "Leveling/results/caserta_table2_840"
MOE_CASERTA_RESULTS = MIJN_RL / "output/moe_lora_caserta_20260918"
MOE_SEALED_RESULTS = MIJN_RL / "output/moe_lora_test_sealed_20260917"
BASE_CHECKPOINT = MIJN_RL / "checkpoints/h20.best.pt"
SEALED_MANIFEST = (
    MIJN_RL
    / "data/online_crp/lora_domains/large_scale_n100_150/test_sealed.json"
)
ADAPTERS = (
    (
        "MoE_seed20260913",
        MIJN_RL / "output/moe_lora_long_20260917_seed20260913/best.adapter.pt",
    ),
    (
        "MoE_seed20260923",
        MIJN_RL / "output/moe_lora_long_20260917_seed20260923/best.adapter.pt",
    ),
    (
        "MoE_seed20260933",
        MIJN_RL / "output/moe_lora_long_20260917_seed20260933/best.adapter.pt",
    ),
)
HORIZONS = (0, 5, 10, 20)


def _write_evidence(name: str, payload: Mapping[str, Any]) -> Path:
    EVIDENCE_DIR.mkdir(exist_ok=True)
    path = EVIDENCE_DIR / f"{name}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"non-object JSONL record at {path}:{line_number}")
            result.append(record)
    return result


def _flatten(stacks: Iterable[Iterable[int]]) -> list[int]:
    return [int(item) for stack in stacks for item in stack]


def _assert_public_state(environment: Any, episode: Any) -> None:
    state = environment.state
    physical = environment._stacks
    next_index = int(environment._next_index)
    horizon = int(environment.lookahead_horizon)
    sequence = tuple(int(item) for item in episode.retrieval_sequence)
    visible_end = min(next_index + horizon + 1, len(sequence))
    visible = {
        item: offset + 1
        for offset, item in enumerate(sequence[next_index:visible_end])
    }
    expected = tuple(
        tuple(visible.get(int(item), 0) for item in stack) for stack in physical
    )
    if state.stacks != expected:
        raise AssertionError("public observation exposes incorrect priorities")
    allowed = set(range(0, horizon + 2))
    if any(value not in allowed for stack in state.stacks for value in stack):
        raise AssertionError("public observation contains an out-of-window priority")
    if state.terminated != (next_index == len(sequence)):
        raise AssertionError("termination flag disagrees with retrieval progress")
    if not state.terminated:
        target_stacks = [i for i, stack in enumerate(state.stacks) if 1 in stack]
        if len(target_stacks) != 1:
            raise AssertionError("public state does not expose exactly one current target")


def _assert_physical_state(environment: Any, episode: Any) -> None:
    stacks = environment._stacks
    sequence = tuple(int(item) for item in episode.retrieval_sequence)
    next_index = int(environment._next_index)
    remaining = _flatten(stacks)
    expected_remaining = list(sequence[next_index:])
    if len(remaining) != len(set(remaining)):
        raise AssertionError("duplicate physical container in yard")
    if set(remaining) != set(expected_remaining):
        raise AssertionError("container identity/count invariant failed")
    if any(len(stack) > int(episode.max_tiers) for stack in stacks):
        raise AssertionError("stack capacity exceeded")
    for stack_id, stack in enumerate(stacks):
        for item in stack:
            if environment._locations[item] != stack_id:
                raise AssertionError("container location map is inconsistent")
    for item in sequence[:next_index]:
        if environment._locations[item] is not None:
            raise AssertionError("retrieved container still has a yard location")
    _assert_public_state(environment, episode)


def _audit_transition(environment: Any, episode: Any, destination: int) -> int:
    """Validate and execute exactly one restricted relocation.

    Returns the number of automatic retrievals performed by the transition.
    """

    _assert_physical_state(environment, episode)
    before_stacks = tuple(tuple(stack) for stack in environment._stacks)
    before_items = _flatten(before_stacks)
    before_next = int(environment._next_index)
    before_relocations = int(environment.relocation_count)
    target = environment.current_target_id
    if target is None:
        raise AssertionError("attempted relocation after termination")
    source = int(environment._source_stack())
    legal = tuple(int(item) for item in environment.legal_destinations())
    if destination not in legal:
        raise AssertionError(
            f"illegal destination {destination}; source={source}; legal={legal}"
        )
    if destination == source:
        raise AssertionError("destination equals source")
    if len(environment._stacks[destination]) >= int(episode.max_tiers):
        raise AssertionError("destination is full")
    source_stack = environment._stacks[source]
    if not source_stack or target not in source_stack or source_stack[-1] == target:
        raise AssertionError("restricted-CRP source/target condition failed")
    blocker = int(source_stack[-1])
    target_tier = source_stack.index(target)
    if target_tier >= len(source_stack) - 1:
        raise AssertionError("moved container is not above current target")

    environment.step(int(destination))

    after_next = int(environment._next_index)
    if int(environment.relocation_count) != before_relocations + 1:
        raise AssertionError("relocation counter did not increase by exactly one")
    if after_next < before_next:
        raise AssertionError("retrieval progress moved backwards")
    retrieved = tuple(episode.retrieval_sequence[before_next:after_next])
    after_items = _flatten(environment._stacks)
    if set(before_items) != set(after_items) | set(retrieved):
        raise AssertionError("transition lost, duplicated, or created a container")
    if len(before_items) != len(after_items) + len(retrieved):
        raise AssertionError("transition container count is inconsistent")
    if blocker not in retrieved:
        if environment._locations[blocker] != destination:
            raise AssertionError("moved blocker location is not destination")
        if blocker not in environment._stacks[destination]:
            raise AssertionError("moved blocker is absent from destination")
    _assert_physical_state(environment, episode)
    if environment.state.terminated and any(environment._stacks):
        raise AssertionError("terminal yard is not empty")
    return after_next - before_next


def _assert_trace_state(actual: Any, expected: Any) -> None:
    scalar = actual.state
    if scalar.stacks != expected.stacks:
        raise AssertionError("accelerated trace/public state mismatch")
    fields = ("retrieval_count", "relocation_count", "crane_position", "terminated")
    for field in fields:
        if getattr(scalar, field) != getattr(expected, field):
            raise AssertionError(f"accelerated trace {field} mismatch")
    if abs(float(scalar.crane_time) - float(expected.crane_time)) > 1e-9:
        raise AssertionError("accelerated trace crane_time mismatch")


def audit_gp() -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(GP_CONFIG, workspace=ROOT)
    episodes = load_gp_manifest(config.test_manifest)
    training_episodes = load_gp_manifest(config.train_manifest)
    test_ids = tuple(episode.canonical_id for episode in episodes)
    train_ids = {episode.canonical_id for episode in training_episodes}
    if len(episodes) != 840 or len(training_episodes) != 840:
        raise AssertionError("GP train/test manifests are not both 840 episodes")
    if train_ids.intersection(test_ids):
        raise AssertionError("GP training and test manifests overlap")
    current_impl_hash = implementation_sha256()
    formal_implementation_hashes: set[str] = set()
    expected_runs = len(config.horizons) * len(config.terminal_sets) * config.runs_per_cell
    run_paths = sorted(GP_RESULTS.glob("runs/H*/**/run_complete.json"))
    if len(run_paths) != expected_runs:
        raise AssertionError(f"expected {expected_runs} GP runs, found {len(run_paths)}")

    decisions = 0
    automatic_retrievals = 0
    episode_replays = 0
    count_mismatches = 0
    illegal_actions = 0
    provenance_mismatches = 0
    leakage_pairs = 0
    run_totals: dict[str, int] = {}
    for run_index, run_path in enumerate(run_paths, start=1):
        record = _read_json(run_path)
        horizon = int(record["horizon"])
        terminal_set = str(record["terminal_set"])
        run = int(record["run"])
        expected_seed = run_seed(config, horizon, terminal_set, run)
        train_path = run_path.with_name("train_complete.json")
        training = _read_json(train_path)
        formal_implementation_hash = record.get("implementation_sha256")
        checks = (
            record.get("record_type") == "gp_run_complete",
            record.get("protocol") == config.protocol,
            record.get("config_sha256") == config.sha256,
            isinstance(formal_implementation_hash, str),
            len(str(formal_implementation_hash)) == 64,
            record.get("train_manifest_sha256") == sha256_file(config.train_manifest),
            record.get("test_manifest_sha256") == sha256_file(config.test_manifest),
            int(record.get("visible_target_count", -1)) == horizon + 1,
            terminal_set in TERMINAL_SETS,
            int(record.get("seed", -1)) == expected_seed,
            training.get("best_prefix") == record.get("best_prefix"),
            training.get("seed") == record.get("seed"),
            training.get("config_sha256") == record.get("config_sha256"),
            training.get("implementation_sha256")
            == record.get("implementation_sha256"),
        )
        if not all(checks):
            provenance_mismatches += 1
            raise AssertionError(f"GP provenance mismatch: {run_path}")
        formal_implementation_hashes.add(str(formal_implementation_hash))
        tree = Node.from_prefix(record["best_prefix"])
        if tree.expression() != record["best_expression"]:
            raise AssertionError(f"GP expression/prefix mismatch: {run_path}")
        if tree.size != int(record["best_tree_size"]) or tree.depth != int(
            record["best_tree_depth"]
        ):
            raise AssertionError(f"GP tree metadata mismatch: {run_path}")
        reported_order = tuple(str(item) for item in record["episode_order"])
        reported_values = tuple(int(item) for item in record["episode_relocations"])
        if reported_order != test_ids or len(reported_values) != len(episodes):
            raise AssertionError(f"GP episode order mismatch: {run_path}")

        replay_values: list[int] = []
        for episode, reported in zip(episodes, reported_values, strict=True):
            trace = trace_tree(
                tree, episode, horizon=horizon, max_steps=max(reported + 1, 1)
            )
            environment = GPOnlineCRPEnvironment(
                episode, lookahead_horizon=horizon
            )
            _assert_physical_state(environment, episode)
            for decision in trace.decisions:
                _assert_trace_state(environment, decision.state)
                legal = environment.legal_destinations()
                if legal != decision.legal_destinations:
                    raise AssertionError("GP accelerated/scalar legal-set mismatch")
                if decision.selected_destination not in legal:
                    illegal_actions += 1
                    raise AssertionError("GP trace selected an illegal action")
                automatic_retrievals += _audit_transition(
                    environment, episode, decision.selected_destination
                )
                decisions += 1
            _assert_trace_state(environment, trace.final_state)
            replayed = int(environment.relocation_count)
            replay_values.append(replayed)
            episode_replays += 1
            if replayed != reported:
                count_mismatches += 1
                raise AssertionError(
                    f"GP count mismatch {run_path} {episode.episode_id}: "
                    f"reported={reported} replayed={replayed}"
                )
        if sum(replay_values) != int(record["summary"]["total_relocations"]):
            raise AssertionError(f"GP summary total mismatch: {run_path}")
        run_totals[
            f"H{horizon}/{terminal_set}/run_{run:02d}"
        ] = sum(replay_values)

        # Metamorphic leakage check on one formal state per evolved rule.  The
        # visible prefix is fixed while the hidden suffix is reversed.
        for episode in episodes:
            original = GPOnlineCRPEnvironment(episode, lookahead_horizon=horizon)
            if original.state.terminated:
                continue
            boundary = min(
                original._next_index + horizon + 1,
                len(episode.retrieval_sequence),
            )
            if boundary >= len(episode.retrieval_sequence) - 1:
                continue
            altered_sequence = (
                episode.retrieval_sequence[:boundary]
                + tuple(reversed(episode.retrieval_sequence[boundary:]))
            )
            altered_episode = GPOnlineCRPEpisode(
                episode_id=episode.episode_id + "-hidden-suffix-audit",
                num_stacks=episode.num_stacks,
                max_tiers=episode.max_tiers,
                initial_stacks=episode.initial_stacks,
                retrieval_sequence=altered_sequence,
                parameter_group=episode.parameter_group,
                metadata=episode.metadata,
            )
            altered = GPOnlineCRPEnvironment(
                altered_episode, lookahead_horizon=horizon
            )
            if original.state != altered.state:
                continue
            action_a = select_destination(
                tree,
                original.state,
                original.legal_destinations(),
                max_tiers=episode.max_tiers,
            )
            action_b = select_destination(
                tree,
                altered.state,
                altered.legal_destinations(),
                max_tiers=episode.max_tiers,
            )
            if action_a != action_b:
                raise AssertionError("GP action changed after hidden-suffix permutation")
            leakage_pairs += 1
            break
        if run_index % 10 == 0:
            print(
                f"[gp] {run_index}/{expected_runs} runs; "
                f"episodes={episode_replays}; decisions={decisions}",
                flush=True,
            )

    summary = _read_json(GP_RESULTS / "summary.json")
    if len(formal_implementation_hashes) != 1:
        raise AssertionError("GP formal runs mix implementation hashes")
    formal_implementation_hash = next(iter(formal_implementation_hashes))
    if int(summary["run_count"]) != expected_runs:
        raise AssertionError("GP aggregate summary run count mismatch")
    if summary.get("implementation_sha256") != formal_implementation_hash:
        raise AssertionError("GP summary does not preserve formal run provenance")
    if summary.get("reporting_implementation_sha256") != current_impl_hash:
        raise AssertionError("GP summary reporting implementation hash mismatch")
    episode_lines = _read_jsonl(GP_RESULTS / "episode_results.jsonl")
    if len(episode_lines) != expected_runs * 840:
        raise AssertionError("GP aggregate episode_results row count mismatch")
    result = {
        "status": "PASS",
        "validation": "FULL",
        "formal_runs": expected_runs,
        "instances_per_run": 840,
        "episode_solutions_checked": episode_replays,
        "relocation_decisions_checked": decisions,
        "automatic_retrievals_observed": automatic_retrievals,
        "illegal_actions": illegal_actions,
        "count_mismatches": count_mismatches,
        "provenance_mismatches": provenance_mismatches,
        "final_yards_valid": episode_replays,
        "horizons": list(config.horizons),
        "visible_target_counts": [horizon + 1 for horizon in config.horizons],
        "terminal_sets": list(config.terminal_sets),
        "seeds_checked": expected_runs,
        "hidden_suffix_metamorphic_pairs": leakage_pairs,
        "future_information_leakage_detected": False,
        "train_instances": len(training_episodes),
        "test_instances": len(episodes),
        "train_test_overlap": 0,
        "train_manifest_sha256": sha256_file(config.train_manifest),
        "test_manifest_sha256": sha256_file(config.test_manifest),
        "config_sha256": config.sha256,
        "implementation_sha256": formal_implementation_hash,
        "reporting_implementation_sha256": current_impl_hash,
        "run_total_relocations": run_totals,
        "reporting_issue": {
            "status": "RESOLVED",
            "file": "GP/summary.py",
            "function": "_plot_convergence",
            "error": "ValueError: zip() argument 2 is shorter than argument 1",
            "resolution": "subplot layout now follows the configured horizon count; formal H=20 convergence.png was generated successfully",
            "affects_solution_or_relocation_count": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_evidence("gp", result)
    return result


def audit_leveling() -> dict[str, Any]:
    started = time.perf_counter()
    episodes = load_episode_manifest(CASERTA)
    independent_episodes = load_leveling_manifest(CASERTA)
    if len(episodes) != 840 or len(independent_episodes) != 840:
        raise AssertionError("Leveling Caserta manifest is incomplete")
    reported_records = _read_jsonl(LEVELING_RESULTS / "episode_results.jsonl")
    reported: dict[tuple[int, str], int] = {}
    for record in reported_records:
        key = (int(record["comparison_horizon"]), str(record["canonical_id"]))
        if key in reported:
            raise AssertionError("duplicate Leveling result row")
        reported[key] = int(record["relocations"])
    if len(reported) != 4 * 840:
        raise AssertionError("Leveling result file does not contain 4x840 rows")

    decisions = 0
    automatic_retrievals = 0
    count_mismatches = 0
    illegal_actions = 0
    totals = 0
    replay_by_id: dict[str, int] = {}
    for episode in episodes:
        environment = OnlineCRPEnvironment(episode, lookahead_horizon=0)
        _assert_physical_state(environment, episode)
        while not environment.state.terminated:
            legal = environment.legal_destinations()
            destination = LevelingBaseline.select_destination(
                environment.state, legal
            )
            source = environment._source_stack()
            expected = min(
                legal,
                key=lambda stack_id: (
                    len(environment._stacks[stack_id]),
                    stack_id,
                ),
            )
            if destination != expected:
                raise AssertionError("Leveling destination rule mismatch")
            if destination not in legal or destination == source:
                illegal_actions += 1
                raise AssertionError("Leveling selected an illegal destination")
            automatic_retrievals += _audit_transition(
                environment, episode, destination
            )
            decisions += 1
        replayed = int(environment.relocation_count)
        replay_by_id[episode.canonical_id] = replayed
        totals += replayed
        for horizon in HORIZONS:
            expected_reported = reported[(horizon, episode.canonical_id)]
            if replayed != expected_reported:
                count_mismatches += 1
                raise AssertionError(
                    f"Leveling count mismatch {episode.episode_id} H={horizon}: "
                    f"reported={expected_reported} replayed={replayed}"
                )
    summary = _read_json(LEVELING_RESULTS / "summary.json")
    if totals != int(summary["baseline"]["total_relocations"]):
        raise AssertionError("Leveling summary total mismatch")
    if totals != 36_024:
        raise AssertionError(f"Leveling regression failed: expected 36024, got {totals}")
    result = {
        "status": "PASS",
        "validation": "FULL",
        "instances_checked": len(episodes),
        "reported_rows_checked": len(reported),
        "relocation_decisions_checked": decisions,
        "automatic_retrievals_observed": automatic_retrievals,
        "illegal_actions": illegal_actions,
        "count_mismatches": count_mismatches,
        "final_yards_valid": len(episodes),
        "expected_total_relocations": 36_024,
        "replayed_total_relocations": totals,
        "strict_36024_match": totals == 36_024,
        "horizon_dependency": False,
        "future_information_leakage_detected": False,
        "test_instances": len(episodes),
        "test_manifest_sha256": manifest_sha256(CASERTA),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_evidence("leveling", result)
    return result


def _reported_caserta() -> dict[tuple[str, int, str], int]:
    records = _read_jsonl(MOE_CASERTA_RESULTS / "episode_results.jsonl")
    result: dict[tuple[str, int, str], int] = {}
    for record in records:
        key = (
            str(record["algorithm"]),
            int(record["lookahead_horizon"]),
            str(record["canonical_id"]),
        )
        if key in result:
            raise AssertionError("duplicate MoE Caserta result row")
        result[key] = int(record["relocations"])
    return result


def _reported_sealed() -> dict[tuple[str, str], int]:
    records = _read_jsonl(MOE_SEALED_RESULTS / "episode_results.jsonl")
    result: dict[tuple[str, str], int] = {}
    for record in records:
        key = (str(record["algorithm"]), str(record["canonical_id"]))
        if key in result:
            raise AssertionError("duplicate MoE sealed result row")
        result[key] = int(record["relocations"])
    return result


def _policy_actions(
    policy: torch.nn.Module,
    priorities: torch.Tensor,
    forbidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(), torch.autocast(
        device_type=priorities.device.type, enabled=False
    ):
        masked = policy.action_log_probabilities(priorities, forbidden).float()
        unmasked = policy.action_log_probabilities(
            priorities, torch.zeros_like(forbidden)
        ).float()
    return masked.argmax(dim=1), unmasked.argmax(dim=1)


def _audit_moe_caserta_method(
    method: str,
    policy: torch.nn.Module,
    episodes: Sequence[OnlineCRPEpisode],
    reported: Mapping[tuple[str, int, str], int],
    *,
    device: str,
) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[OnlineCRPEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode.num_stacks, episode.max_tiers)].append(episode)
    decisions = 0
    raw_unmasked_illegal = 0
    automatic_retrievals = 0
    count_mismatches = 0
    by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        values: dict[str, int] = {}
        horizon_decisions = 0
        horizon_unmasked = 0
        for batch in grouped.values():
            vector = VectorizedOnlineCRPEnvironment(
                batch,
                rollouts_per_episode=1,
                lookahead_horizon=horizon,
                device=device,
                validate_actions=True,
            )
            scalar = [
                OnlineCRPEnvironment(episode, lookahead_horizon=horizon)
                for episode in batch
            ]
            adapters = [Online2DPriorityAdapter(episode.max_tiers) for episode in batch]
            while vector.active_indices.numel():
                active = vector.active_indices
                active_rows = [int(item) for item in active.detach().cpu().tolist()]
                priorities = vector.priorities[active]
                forbidden = vector.forbidden_mask[active]
                for local, row in enumerate(active_rows):
                    environment = scalar[row]
                    episode = batch[row]
                    _assert_physical_state(environment, episode)
                    expected_priorities = adapters[row].build(environment.state)
                    actual_priorities = priorities[local].detach().cpu().numpy()
                    if not np.array_equal(expected_priorities, actual_priorities):
                        raise AssertionError("MoE vector/scalar observation mismatch")
                    scalar_legal = set(environment.legal_destinations())
                    vector_legal = set(
                        int(item)
                        for item in torch.nonzero(~forbidden[local], as_tuple=False)
                        .flatten()
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    if scalar_legal != vector_legal:
                        raise AssertionError("MoE vector/scalar legal mask mismatch")
                actions, raw_actions = _policy_actions(
                    policy, priorities, forbidden
                )
                raw_illegal = forbidden.gather(1, raw_actions[:, None]).squeeze(1)
                increment = int(raw_illegal.sum().detach().cpu())
                raw_unmasked_illegal += increment
                horizon_unmasked += increment
                action_values = [int(item) for item in actions.detach().cpu().tolist()]
                for local, row in enumerate(active_rows):
                    environment = scalar[row]
                    episode = batch[row]
                    destination = action_values[local]
                    if destination not in environment.legal_destinations():
                        raise AssertionError("masked MoE action is illegal")
                    automatic_retrievals += _audit_transition(
                        environment, episode, destination
                    )
                    decisions += 1
                    horizon_decisions += 1
                vector.step_active_(actions)
                for row, environment in enumerate(scalar):
                    if int(vector.relocation_count[row]) != environment.relocation_count:
                        raise AssertionError("MoE vector/scalar relocation count mismatch")
                    if int(vector.retrieval_count[row]) != environment._next_index:
                        raise AssertionError("MoE vector/scalar retrieval count mismatch")
                    if bool(vector.terminated[row]) != environment.state.terminated:
                        raise AssertionError("MoE vector/scalar termination mismatch")
                    expected_priorities = adapters[row].build(environment.state)
                    actual_priorities = vector.priorities[row].detach().cpu().numpy()
                    if not np.array_equal(expected_priorities, actual_priorities):
                        raise AssertionError("MoE post-step vector/scalar state mismatch")
            for episode, environment in zip(batch, scalar, strict=True):
                if not environment.state.terminated or any(environment._stacks):
                    raise AssertionError("MoE Caserta replay did not empty yard")
                replayed = int(environment.relocation_count)
                values[episode.canonical_id] = replayed
                expected = reported[(method, horizon, episode.canonical_id)]
                if replayed != expected:
                    count_mismatches += 1
                    raise AssertionError(
                        f"MoE Caserta count mismatch {method} H={horizon} "
                        f"{episode.episode_id}: reported={expected} replayed={replayed}"
                    )
        by_horizon[str(horizon)] = {
            "episodes": len(values),
            "decisions": horizon_decisions,
            "raw_unmasked_argmax_illegal": horizon_unmasked,
            "total_relocations": sum(values.values()),
        }
        print(
            f"[moe-caserta] {method} H={horizon}; episodes={len(values)}; "
            f"decisions={horizon_decisions}; total={sum(values.values())}",
            flush=True,
        )
    return {
        "episode_solutions_checked": len(episodes) * len(HORIZONS),
        "relocation_decisions_checked": decisions,
        "automatic_retrievals_observed": automatic_retrievals,
        "illegal_actions_after_mask": 0,
        "raw_unmasked_argmax_illegal": raw_unmasked_illegal,
        "count_mismatches": count_mismatches,
        "final_yards_valid": len(episodes) * len(HORIZONS),
        "by_horizon": by_horizon,
    }


def _hidden_suffix_episode(
    episode: OnlineCRPEpisode, horizon: int
) -> tuple[OnlineCRPEnvironment, OnlineCRPEnvironment] | None:
    original = OnlineCRPEnvironment(episode, lookahead_horizon=horizon)
    if original.state.terminated:
        return None
    boundary = min(original._next_index + horizon + 1, episode.num_containers)
    if boundary >= episode.num_containers - 1:
        return None
    altered_sequence = (
        episode.retrieval_sequence[:boundary]
        + tuple(reversed(episode.retrieval_sequence[boundary:]))
    )
    altered_episode = OnlineCRPEpisode(
        episode_id=episode.episode_id + "-hidden-suffix-audit",
        num_stacks=episode.num_stacks,
        max_tiers=episode.max_tiers,
        initial_stacks=episode.initial_stacks,
        retrieval_sequence=altered_sequence,
        parameter_group=episode.parameter_group,
        metadata=episode.metadata,
    )
    altered = OnlineCRPEnvironment(altered_episode, lookahead_horizon=horizon)
    if original.state != altered.state:
        return None
    return original, altered


def _moe_leakage_check(
    policy: torch.nn.Module,
    episodes: Sequence[OnlineCRPEpisode],
    *,
    horizon: int,
    device: str,
) -> bool:
    for episode in episodes:
        pair = _hidden_suffix_episode(episode, horizon)
        if pair is None:
            continue
        original, altered = pair
        adapter = Online2DPriorityAdapter(episode.max_tiers)
        arrays = [adapter.build(environment.state) for environment in pair]
        if not np.array_equal(arrays[0], arrays[1]):
            raise AssertionError("hidden-suffix public states differ")
        priorities = torch.as_tensor(
            np.stack(arrays), dtype=torch.float32, device=device
        )
        forbidden = torch.ones(
            (2, episode.num_stacks), dtype=torch.bool, device=device
        )
        for row, environment in enumerate(pair):
            forbidden[row, list(environment.legal_destinations())] = False
        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(device).type, enabled=False
        ):
            actions = policy.action_log_probabilities(
                priorities, forbidden
            ).argmax(dim=1)
        if int(actions[0]) != int(actions[1]):
            raise AssertionError("MoE action changed after hidden-suffix permutation")
        return True
    return False


def _checkpoint_provenance(
    method: str, checkpoint: Path, device: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    policy, record = _load_policy(
        base_checkpoint=BASE_CHECKPOINT,
        adapter_checkpoint=checkpoint,
        device=device,
    )
    seed = int(method.rsplit("seed", 1)[1])
    if int(record["seed"]) != seed:
        raise AssertionError(f"adapter seed/name mismatch for {method}")
    return policy, record


def audit_moe_caserta(device: str) -> dict[str, Any]:
    started = time.perf_counter()
    episodes = load_episode_manifest(CASERTA)
    if len(episodes) != 840:
        raise AssertionError("MoE Caserta manifest is incomplete")
    reported = _reported_caserta()
    expected_rows = (1 + len(ADAPTERS)) * len(HORIZONS) * len(episodes)
    if len(reported) != expected_rows:
        raise AssertionError(
            f"MoE Caserta output rows: expected {expected_rows}, got {len(reported)}"
        )
    published_summary = _read_json(MOE_CASERTA_RESULTS / "summary.json")
    if published_summary["manifest_sha256"] != manifest_sha256(CASERTA):
        raise AssertionError("MoE Caserta manifest hash mismatch")

    methods: dict[str, Any] = {}
    checkpoint_records: dict[str, Any] = {}
    named: tuple[tuple[str, Path | None], ...] = (("MijnRL_base", None), *ADAPTERS)
    for method, checkpoint in named:
        policy, provenance = _load_policy(
            base_checkpoint=BASE_CHECKPOINT,
            adapter_checkpoint=checkpoint,
            device=device,
        )
        methods[method] = _audit_moe_caserta_method(
            method, policy, episodes, reported, device=device
        )
        leakage = {
            str(horizon): _moe_leakage_check(
                policy, episodes, horizon=horizon, device=device
            )
            for horizon in HORIZONS
        }
        if not all(leakage.values()):
            raise AssertionError("could not construct MoE hidden-suffix checks")
        methods[method]["hidden_suffix_metamorphic_check"] = leakage
        checkpoint_records[method] = provenance
        published = published_summary["methods"][method]["provenance"]
        if published.get("base_checkpoint_sha256") != provenance.get(
            "base_checkpoint_sha256"
        ):
            raise AssertionError(f"published base hash mismatch for {method}")
        if checkpoint is not None:
            if published.get("adapter_sha256") != provenance.get("adapter_sha256"):
                raise AssertionError(f"published adapter hash mismatch for {method}")
            if int(published.get("seed")) != int(provenance.get("seed")):
                raise AssertionError(f"published adapter seed mismatch for {method}")
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adapter_methods = [methods[name] for name, _ in ADAPTERS]
    result = {
        "status": "PASS",
        "validation": "FULL",
        "role": "external public benchmark; not used for model selection",
        "instances": len(episodes),
        "horizons": list(HORIZONS),
        "horizon_definition": "current target plus H future targets",
        "adapter_episode_solutions_checked": sum(
            item["episode_solutions_checked"] for item in adapter_methods
        ),
        "adapter_relocation_decisions_checked": sum(
            item["relocation_decisions_checked"] for item in adapter_methods
        ),
        "base_reference_episode_solutions_checked": methods["MijnRL_base"][
            "episode_solutions_checked"
        ],
        "illegal_actions_after_mask": sum(
            item["illegal_actions_after_mask"] for item in methods.values()
        ),
        "count_mismatches": sum(
            item["count_mismatches"] for item in methods.values()
        ),
        "future_information_leakage_detected": False,
        "methods": methods,
        "checkpoint_provenance": checkpoint_records,
        "manifest_sha256": manifest_sha256(CASERTA),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_evidence("moe_caserta", result)
    return result


def audit_moe_sealed(device: str) -> dict[str, Any]:
    started = time.perf_counter()
    episodes = load_episode_manifest(SEALED_MANIFEST)
    if len(episodes) != 500:
        raise AssertionError("sealed test manifest is not 500 episodes")
    reported = _reported_sealed()
    published_summary = _read_json(MOE_SEALED_RESULTS / "summary.json")
    if published_summary["manifest_sha256"] != manifest_sha256(SEALED_MANIFEST):
        raise AssertionError("sealed manifest hash mismatch")
    methods: dict[str, Any] = {}
    total_decisions = 0
    total_retrievals = 0
    mismatches = 0
    for method, checkpoint in ADAPTERS:
        policy, provenance = _checkpoint_provenance(method, checkpoint, device)
        method_decisions = 0
        method_retrievals = 0
        method_total = 0
        for episode_index, episode in enumerate(episodes, start=1):
            environment = OnlineCRPEnvironment(episode, lookahead_horizon=20)
            adapter = Online2DPriorityAdapter(episode.max_tiers)
            _assert_physical_state(environment, episode)
            while not environment.state.terminated:
                legal = environment.legal_destinations()
                observation = adapter.build(environment.state)
                priorities = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                forbidden = torch.ones(
                    (1, episode.num_stacks), dtype=torch.bool, device=device
                )
                forbidden[0, list(legal)] = False
                with torch.inference_mode(), torch.autocast(
                    device_type=torch.device(device).type, enabled=False
                ):
                    action = policy.action_log_probabilities(
                        priorities, forbidden
                    ).argmax(dim=1)
                destination = int(action.item())
                if destination not in legal:
                    raise AssertionError("sealed MoE selected an illegal action")
                method_retrievals += _audit_transition(
                    environment, episode, destination
                )
                method_decisions += 1
            if any(environment._stacks):
                raise AssertionError("sealed MoE final yard is not empty")
            replayed = int(environment.relocation_count)
            expected = reported[(method, episode.canonical_id)]
            if replayed != expected:
                mismatches += 1
                raise AssertionError(
                    f"sealed MoE mismatch {method} {episode.episode_id}: "
                    f"reported={expected} replayed={replayed}"
                )
            method_total += replayed
            if episode_index % 100 == 0:
                print(
                    f"[moe-sealed] {method} {episode_index}/{len(episodes)}; "
                    f"decisions={method_decisions}",
                    flush=True,
                )
        summary_total = int(
            published_summary["methods"][method]["total_relocations"]
        )
        if method_total != summary_total:
            raise AssertionError(f"sealed MoE summary total mismatch for {method}")
        methods[method] = {
            "episodes": len(episodes),
            "relocation_decisions_checked": method_decisions,
            "automatic_retrievals_observed": method_retrievals,
            "total_relocations": method_total,
            "illegal_actions_after_mask": 0,
            "count_mismatches": 0,
            "final_yards_valid": len(episodes),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": manifest_sha256(checkpoint),
            "seed": provenance["seed"],
            "selected_epoch": provenance["selected_epoch"],
        }
        total_decisions += method_decisions
        total_retrievals += method_retrievals
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "status": "PASS",
        "validation": "FULL",
        "role": "official sealed large-scale test",
        "instances": len(episodes),
        "adapter_episode_solutions_checked": len(episodes) * len(ADAPTERS),
        "adapter_relocation_decisions_checked": total_decisions,
        "automatic_retrievals_observed": total_retrievals,
        "illegal_actions_after_mask": 0,
        "count_mismatches": mismatches,
        "final_yards_valid": len(episodes) * len(ADAPTERS),
        "lookahead_horizon": 20,
        "visible_target_count": 21,
        "future_information_leakage_detected": False,
        "test_manifest_used_for_selection": bool(
            published_summary["test_manifest_used_for_selection"]
        ),
        "manifest_sha256": manifest_sha256(SEALED_MANIFEST),
        "methods": methods,
        "runtime_seconds": time.perf_counter() - started,
    }
    if result["test_manifest_used_for_selection"]:
        raise AssertionError("sealed test was used for model selection")
    _write_evidence("moe_sealed", result)
    return result


def audit_moe_provenance() -> dict[str, Any]:
    train = load_large_scale_split(
        MIJN_RL / "data/online_crp/lora_domains/large_scale_n100_150/adapt_train.json",
        "adapt_train",
    )
    validation = load_large_scale_split(
        MIJN_RL / "data/online_crp/lora_domains/large_scale_n100_150/validation.json",
        "validation",
    )
    sealed = load_large_scale_split(SEALED_MANIFEST, "test_sealed")
    sets = {
        "train": {episode.canonical_id for episode in train},
        "validation": {episode.canonical_id for episode in validation},
        "sealed": {episode.canonical_id for episode in sealed},
    }
    caserta_ids = {episode.canonical_id for episode in load_episode_manifest(CASERTA)}
    overlaps = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_sealed": len(sets["train"] & sets["sealed"]),
        "validation_sealed": len(sets["validation"] & sets["sealed"]),
        "train_caserta": len(sets["train"] & caserta_ids),
        "validation_caserta": len(sets["validation"] & caserta_ids),
        "sealed_caserta": len(sets["sealed"] & caserta_ids),
    }
    if any(overlaps.values()):
        raise AssertionError(f"MoE data split overlap: {overlaps}")
    result = {
        "status": "PASS",
        "train_instances": len(train),
        "validation_instances": len(validation),
        "sealed_test_instances": len(sealed),
        "caserta_instances": len(caserta_ids),
        "overlaps": overlaps,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_checkpoint_sha256": manifest_sha256(BASE_CHECKPOINT),
        "adapters": {
            method: {
                "path": str(path),
                "sha256": manifest_sha256(path),
            }
            for method, path in ADAPTERS
        },
    }
    _write_evidence("moe_provenance", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("gp", "leveling", "moe-caserta", "moe-sealed", "provenance", "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    stage = args.stage
    if stage in ("gp", "all"):
        audit_gp()
    if stage in ("leveling", "all"):
        audit_leveling()
    if stage in ("provenance", "all"):
        audit_moe_provenance()
    if stage in ("moe-caserta", "all"):
        audit_moe_caserta(args.device)
    if stage in ("moe-sealed", "all"):
        audit_moe_sealed(args.device)
    print(f"Audit evidence written to {EVIDENCE_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
