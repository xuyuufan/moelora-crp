"""Read this run's validation records and evaluate in-memory router ablations.

Run from Mijn_RL. This script never trains or reads a sealed test manifest.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from experiments.train_online_crp_scale_diverse import (
    _configure_precision,
    evaluate_fixed_validation,
)
from scrp.adaptation import load_adapter, load_h20_base_policy
from scrp.adaptation_data import load_and_validate_target_domain
from scrp.lora import sha256_file
from scrp.moe_lora import configure_moe_lora_training


def main() -> None:
    review = Path(__file__).resolve().parent
    output = review / "analysis.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    run = review.parent
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    epochs = [row for row in rows if row.get("record_type") == "epoch"]
    metadata = rows[0]
    episodes = load_and_validate_target_domain(
        PROJECT / metadata["validation_manifest"],
        domain=metadata["adaptation_config"]["domain"],
        allowed_splits=("validation",),
    )
    ids = [episode.canonical_id for episode in episodes]
    containers = np.asarray([episode.num_containers for episode in episodes])
    baseline = np.asarray([epochs[0]["validation_episode_relocations"][key] for key in ids])
    rng = np.random.default_rng(20260917)
    bootstrap_indices = rng.integers(0, len(ids), size=(20000, len(ids)))

    def comparison(values, reference):
        delta = np.asarray(values) - np.asarray(reference)
        normalized = delta / containers
        return {
            "mean_relocations": float(np.mean(values)),
            "mean_r_per_n": float(np.mean(np.asarray(values) / containers)),
            "mean_delta_relocations": float(delta.mean()),
            "total_delta_relocations": int(delta.sum()),
            "mean_delta_r_per_n": float(normalized.mean()),
            "paired_bootstrap_r_per_n_ci95": np.quantile(
                normalized[bootstrap_indices].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "wins_ties_losses": [int((delta < 0).sum()), int((delta == 0).sum()), int((delta > 0).sum())],
        }

    epoch_results = []
    for row in epochs:
        values = [row["validation_episode_relocations"][key] for key in ids]
        result = {key: value for key, value in row.items() if key != "validation_episode_relocations"}
        result["comparison_vs_epoch0"] = comparison(values, baseline)
        result["relative_r_per_n_reduction_percent"] = 100 * (
            1 - float(np.mean(np.asarray(values) / containers)) / float(np.mean(baseline / containers))
        )
        epoch_results.append(result)

    hashes = {}
    for key in ("base_checkpoint", "train_manifest", "validation_manifest"):
        actual = sha256_file(PROJECT / metadata[key])
        assert actual == metadata[key + "_sha256"], key
        hashes[key] = actual
    for name in ("best.adapter.pt", "latest.adapter.pt"):
        hashes[name] = sha256_file(run / name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _configure_precision("fp32", device)
    model, _, base_hash = load_h20_base_policy(PROJECT / metadata["base_checkpoint"], device=device)
    payload = load_adapter(model, run / "best.adapter.pt", expected_base_checkpoint_sha256=base_hash)
    configure_moe_lora_training(model)
    model.eval()
    layer = model.stack_projection[2]
    probabilities = []

    def observe_router(module, inputs, outputs):
        del outputs
        with torch.autocast(device_type=device, enabled=False):
            gates = F.softmax(F.linear(inputs[0].float(), module.router.weight.float(), module.router.bias.float()), dim=-1)
        probabilities.append(gates.detach().reshape(-1, 2).cpu())

    hook = layer.register_forward_hook(observe_router)
    print("Evaluating trained best adapter on 200 validation episodes", flush=True)
    learned = evaluate_fixed_validation(model, episodes, lookahead_horizon=20, device=device)
    hook.remove()
    gates = torch.cat(probabilities).numpy()
    mean_gates = gates.mean(axis=0, dtype=np.float64)
    router_stats = {
        "token_count": len(gates),
        "mean_weights": mean_gates.tolist(),
        "mean_entropy": float((-(gates * np.log(np.maximum(gates, 1e-8))).sum(axis=1)).mean()),
        "maximum_entropy": math.log(2),
        "expert0_probability_quantiles": dict(zip(
            ("min", "p01", "p10", "p50", "p90", "p99", "max"),
            np.quantile(gates[:, 0], [0, .01, .1, .5, .9, .99, 1]).tolist(),
        )),
        "fraction_max_gate_gt_0_7": float((gates.max(axis=1) > .7).mean()),
        "argmax_counts": np.bincount(gates.argmax(axis=1), minlength=2).tolist(),
    }
    print("Trained best mean relocations:", learned["validation_micro_mean_relocations"], "routing:", router_stats, flush=True)
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.bias.zero_()
    print("Evaluating fixed 50/50 routing (in memory only)", flush=True)
    uniform = evaluate_fixed_validation(model, episodes, lookahead_horizon=20, device=device)
    with torch.no_grad():
        layer.router.bias.copy_(torch.as_tensor(mean_gates, device=device, dtype=torch.float32).log())
    print("Evaluating fixed mean routing (in memory only)", flush=True)
    constant = evaluate_fixed_validation(model, episodes, lookahead_horizon=20, device=device)

    best_epoch = payload["metadata"]["completed_epoch"]
    logged_best = [next(row for row in epochs if row["epoch"] == best_epoch)["validation_episode_relocations"][key] for key in ids]
    learned_costs = [learned["validation_episode_relocations"][key] for key in ids]
    ablations = {}
    for name, validation in (("learned", learned), ("uniform", uniform), ("fixed_mean", constant)):
        values = [validation["validation_episode_relocations"][key] for key in ids]
        ablations[name] = {
            "comparison_vs_epoch0": comparison(values, baseline),
            "comparison_vs_learned": comparison(values, learned_costs),
            "episode_relocations": validation["validation_episode_relocations"],
        }
    shapes = []
    for shape in sorted({(episode.num_stacks, episode.max_tiers) for episode in episodes}):
        indices = [i for i, episode in enumerate(episodes) if (episode.num_stacks, episode.max_tiers) == shape]
        before = baseline[indices]
        after = np.asarray(logged_best)[indices]
        shapes.append({
            "shape": list(shape), "episodes": len(indices),
            "base_r_per_n": float((before / containers[indices]).mean()),
            "best_r_per_n": float((after / containers[indices]).mean()),
            "wins_ties_losses": [int((after < before).sum()), int((after == before).sum()), int((after > before).sum())],
        })
    result = {
        "run": str(run.relative_to(PROJECT)), "device": device, "precision": "fp32",
        "epochs": epoch_results, "best_epoch": best_epoch,
        "logged_best_matches_reevaluation": bool(np.array_equal(logged_best, learned_costs)),
        "router_stats_on_best_validation_trajectories": router_stats,
        "router_ablations": ablations, "best_by_shape": shapes,
        "hashes": hashes, "run_complete": rows[-1],
        "bootstrap_seed": 20260917, "bootstrap_repetitions": 20000,
        "statistical_scope": "Exploratory paired intervals on the model-selection validation set; no correction for selection or multiple comparisons; one training seed.",
        "sealed_test_read": False,
    }
    for name in ("best.adapter.pt", "latest.adapter.pt"):
        assert sha256_file(run / name) == hashes[name]
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"logged_best_matches": result["logged_best_matches_reevaluation"], "ablations": {name: {key: value for key, value in data.items() if key != "episode_relocations"} for name, data in ablations.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
