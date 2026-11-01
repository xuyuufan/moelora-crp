"""Contract checks for the immutable artifacts shipped with the paper code."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from scrp.adaptation_data import load_large_scale_split, manifest_sha256


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "checkpoints/h20.best.pt"
DOMAIN = ROOT / "data/online_crp/lora_domains/large_scale_n100_150"
SEEDS_AND_BEST_EPOCHS = ((20260913, 2), (20260923, 2), (20260933, 4))


def test_released_splits_are_fixed_and_disjoint() -> None:
    expected_counts = {"adapt_train": 1_024, "validation": 200, "test_sealed": 500}
    identifier_sets: dict[str, set[str]] = {}
    for split, expected_count in expected_counts.items():
        episodes = load_large_scale_split(DOMAIN / f"{split}.json", split)
        assert len(episodes) == expected_count
        identifier_sets[split] = {episode.canonical_id for episode in episodes}
    assert identifier_sets["adapt_train"].isdisjoint(identifier_sets["validation"])
    assert identifier_sets["adapt_train"].isdisjoint(identifier_sets["test_sealed"])
    assert identifier_sets["validation"].isdisjoint(identifier_sets["test_sealed"])


def test_released_adapters_match_base_and_final_protocol() -> None:
    base_hash = manifest_sha256(BASE)
    for seed, best_epoch in SEEDS_AND_BEST_EPOCHS:
        run_dir = ROOT / f"output/moe_lora_long_20260917_seed{seed}"
        adapter = torch.load(
            run_dir / "best.adapter.pt", map_location="cpu", weights_only=False
        )
        config = adapter["moe_lora_config"]
        metadata = adapter["metadata"]
        assert adapter["format"] == "mijnrl_moe_lora_adapter_v1"
        assert adapter["base_checkpoint_sha256"] == base_hash
        assert config["target_module"] == "stack_projection.2"
        assert (config["num_experts"], config["rank"], config["alpha"]) == (2, 8, 16.0)
        assert metadata["adaptation_config"]["seed"] == seed
        assert metadata["completed_epoch"] == best_epoch
        assert metadata["test_manifest_used"] is False

        records = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert records[-1]["record_type"] == "run_complete"
        assert records[-1]["best_epoch"] == best_epoch


def test_caserta_evaluation_is_complete_and_matches_historical_base() -> None:
    episodes = json.loads(
        (ROOT / "data/online_crp/caserta_table2_840.json").read_text(encoding="utf-8")
    )
    assert len(episodes) == 840
    assert len({episode["parameter_group"] for episode in episodes}) == 21

    summary = json.loads(
        (ROOT / "output/moe_lora_caserta_20260918/summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["historical_base_totals_matched"] is True
    assert summary["evaluation_horizons"] == [0, 5, 10, 20]
    assert set(summary["methods"]) == {
        "MijnRL_base",
        "MoE_seed20260913",
        "MoE_seed20260923",
        "MoE_seed20260933",
    }
    expected_totals = {"0": 41_277, "5": 30_907, "10": 26_639, "20": 23_993}
    base = summary["methods"]["MijnRL_base"]["horizons"]
    assert {horizon: base[horizon]["total_relocations"] for horizon in base} == expected_totals


def test_caserta_group_comparison_covers_all_groups() -> None:
    report = json.loads(
        (ROOT / "output/moe_lora_caserta_20260918/group_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["horizons"] == [0, 5, 10, 20]
    assert len(report["groups"]) == 21
    assert len(report["seed_detail"]) == 84
    assert {row["episodes"] for row in report["groups"]} == {40}
    assert report["overall"]["leveling_total_relocations"] == 36_024
    assert report["overall"]["leveling_horizon_invariant"] is True
    assert (
        report["overall"]["h0_improved_groups"],
        report["overall"]["h0_worse_groups"],
    ) == (2, 19)
    for horizon in (5, 10, 20):
        assert report["overall"][f"h{horizon}_improved_groups"] == 21
        assert report["overall"][f"h{horizon}_worse_groups"] == 0
