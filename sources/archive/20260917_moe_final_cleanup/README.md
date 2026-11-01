# Mijn_RL final MoE cleanup archive

Date: 2026-09-17

Source workspace: `D:\DSAI\containter_new\Mijn_RL`

This archive contains 44 moved files plus this README (7.949 MiB total) while retaining the final three-seed rank-8 MoE-LoRA result, its large-scale data, sealed-test outputs, paper sources/PDFs, and required code.

## Archived material

- `data/`: the unused `high_density_n50_70` target domain, Caserta Table 2 benchmark data, and the old 420-instance H20 validation manifest.
- `output/`: 2x-rank-16 and no-router-balance MoE ablations; non-selected `latest.adapter.pt` files; live curves and ordinary stdout logs; the seed-20260913 validation review bundle; multi-seed queue status.
- `docs/interface_cleanup.md`: interface-cleanup history, not paper content.
- `experiments/`: the target-manifest generator and old base-training reward-curve plotter.
- `tests/`: six non-MoE-specific tests plus `test_moe_lora_integration.py`, which is the integration wrapper for the removed smoke runner.

## Kept active

- `checkpoints/h20.best.pt`
- `data/online_crp/lora_domains/large_scale_n100_150/`
- Three final `best.adapter.pt` files and their `metrics.jsonl`
- Final sealed-test raw results and summary
- Paper TeX, bibliography, rendered PDFs, and sealed-test figures
- MoE implementation, training/evaluation dependencies, engineering tools, and the standalone MoE unit tests

The archived relative paths mirror their former paths under `Mijn_RL`. Restore a file by moving it back to the same relative path after checking that the destination is unoccupied.
