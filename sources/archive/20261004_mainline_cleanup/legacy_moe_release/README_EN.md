# MijnRL: H=20 Dynamic MoE-LoRA for Online Container Relocation

[中文说明](README.md)

This repository is the minimal reproducible release for the final paper experiment. It contains only the selected dynamic MoE-LoRA method, immutable data, released checkpoints, sealed-test results, and general paper material. Generic LoRA, smoke tests, old generators, Caserta/high-density interfaces, and superseded training entry points are outside the release protocol.

## 1. Experiment definition

- Task: the Online Container Relocation Problem.
- Observation: the current target and at most 20 revealed future targets (`H=20`). The complete retrieval order remains private.
- Base: the frozen `scale_diverse_online_2d_v1` policy in `checkpoints/h20.best.pt`.
- Domain: `large_scale_n100_150`, with 100–150 containers and fill rates from 0.60 to 0.80.
- MoE-LoRA: replace only `stack_projection.2`; two dense rank-8 experts, `alpha=16`, and an FP32 dynamic router, totaling 11,266 trainable parameters.
- Primary metric: compute `R/N` per episode, then take the mean. This is not `sum(R)/sum(N)`.
- Isolation: `adapt_train` updates parameters, `validation` selects the best epoch, and `test_sealed` is opened only after all training is complete. The trainer has no test-manifest argument.

The fixed splits contain 1,024 training, 200 validation, and 500 sealed-test episodes. Their canonical IDs are pairwise disjoint.

## 2. Repository layout

```text
checkpoints/h20.best.pt                      frozen H=20 base model
data/.../large_scale_n100_150/               fixed train/validation/test splits
scrp/                                        environment, policy, MoE, training core
experiments/train_moe_lora.py                formal single-seed training
experiments/evaluate_moe_lora.py             FP32 greedy sealed evaluation
experiments/evaluate_moe_caserta.py          four-horizon Caserta evaluation
experiments/reproduce.py                     three-seed end-to-end entry point
experiments/verify_release.py                SHA-256 release verification
output/moe_lora_long_20260917_seed*/          released adapters and metrics
output/moe_lora_test_sealed_20260917/         released raw results and summary
output/moe_lora_caserta_20260918/             Caserta 840-episode evaluation
output/pdf/                                   compiled general problem-definition PDFs
figures/gen_fig_moe_sealed_test.py            regenerate the paper figure
docs/                                        general paper definition and references
tests/                                       MoE and release-contract regressions
```

## 3. Installation

Verified stack: Windows 11, Python 3.12.13, PyTorch 2.12.0+cu126, CUDA 12.6, cuDNN 91002, NumPy 2.4.4, Matplotlib 3.11.2, and pytest 9.1.0. Formal results were produced on an NVIDIA GeForce RTX 4060 Laptop GPU.

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`. A CUDA GPU is required for formal reproduction. CPU execution can validate the pipeline, but backend-level numerical differences may cause some greedy trajectories to diverge.

Verify the downloaded artifacts first:

```powershell
python -B -m experiments.verify_release
```

Expected output: `verified 30 release artifacts`.

## 4. Fast end-to-end check

This runs one seed, one optimizer update, and four sealed episodes. It validates train/save/reload/evaluate behavior; it is not a paper result.

```powershell
python -B -m experiments.reproduce `
  --quick-check `
  --device cuda `
  --precision-mode fp16 `
  --output-root output/reproduction_quick
```

The output directory must not already exist. A successful run creates:

```text
output/reproduction_quick/
  seed20260913/best.adapter.pt
  seed20260913/latest.adapter.pt
  seed20260913/metrics.jsonl
  sealed_test/summary.json
  sealed_test/episode_results.jsonl
  reproduction.json
```

## 5. Full three-seed reproduction

```powershell
python -B -m experiments.reproduce `
  --device cuda `
  --precision-mode fp16 `
  --output-root output/reproduction_full
```

The command trains seeds `20260913`, `20260923`, and `20260933`, selects each best adapter by validation mean `R/N`, evaluates all 500 sealed episodes, and generates the PNG/PDF figure. Each seed took about 48–49 minutes on the tested RTX 4060 Laptop GPU, or roughly 2.4 hours for training alone.

The formal hyperparameters are fixed in code so an altered run cannot accidentally be presented as a reproduction:

| Setting | Value |
|---|---:|
| epochs | 5 |
| batch size | 16 |
| rollouts / instance | 8 |
| replay decision batch | 2048 |
| learning rate | 3e-4 |
| entropy coefficient | 0.002 |
| router balance coefficient | 0.01 |
| gradient clip norm | 1.0 |
| training precision | FP16 |
| validation / test precision | FP32 greedy |

Train a single seed with:

```powershell
python -B -m experiments.train_moe_lora `
  --seed 20260913 `
  --device cuda `
  --precision-mode fp16 `
  --output-dir output/reproduce_seed20260913
```

Re-evaluate the three released adapters without retraining:

```powershell
python -B -m experiments.evaluate_moe_lora `
  --device cuda `
  --output-dir output/recheck_released_sealed
```

Evaluation always performs dynamic MoE routing. The adapter cannot be statically merged into the base linear layer.

## 6. Released reference results

Validation-selected epochs:

| seed | best epoch | validation mean R/N |
|---:|---:|---:|
| 20260913 | 2 | 0.928947 |
| 20260923 | 2 | 0.928233 |
| 20260933 | 4 | 0.928554 |

Released results on 500 sealed episodes:

| Method | mean R/N | mean relocations |
|---|---:|---:|
| MoE seed 20260913 | 0.920944 | 116.326 |
| MoE seed 20260923 | 0.919516 | 116.148 |
| MoE seed 20260933 | 0.919412 | 116.136 |
| frozen base | 0.946317 | 119.584 |
| Leveling | 1.189675 | 150.268 |

Different GPUs, drivers, or math libraries can introduce tiny floating-point differences that alter a later greedy trajectory when logits are nearly tied. Use the tested CUDA stack for strict episode-level comparison; cross-platform reproduction should compare aggregate metrics and conclusions.

## 7. External Caserta benchmark

After freezing the final three adapters, we evaluated them on all 840 original Caserta Table 2 instances without retraining or checkpoint selection. Run:

```powershell
python -B -m experiments.evaluate_moe_caserta `
  --device cuda `
  --output-dir output/moe_lora_caserta_20260918
```

Values below are episode-wise mean `R/N`. Parentheses report the three-seed MoE mean relative to the base; lower is better.

| Method | H=0 | H=5 | H=10 | H=20 |
|---|---:|---:|---:|---:|
| frozen base | 1.247424 | 0.890973 | 0.788302 | 0.733953 |
| MoE seed 20260913 | 1.076285 | 0.853679 | 0.783186 | 0.734343 |
| MoE seed 20260923 | 1.083585 | 0.855367 | 0.784189 | 0.734237 |
| MoE seed 20260933 | 1.078470 | 0.854207 | 0.783943 | 0.734512 |
| three-seed MoE mean | 1.079447 (-13.47%) | 0.854418 (-4.10%) | 0.783773 (-0.57%) | 0.734364 (+0.06%) |

Base totals at the four horizons were 41,277, 30,907, 26,639, and 23,993 relocations, exactly matching the historical Caserta evaluation. All three MoE runs improve H=0, H=5, and H=10. H=20 is effectively tied but slightly worse; all three paired bootstrap intervals cross zero, so the data do not support an H=20 Caserta improvement claim. This is an external public benchmark, not a sealed test or model-selection source.

The paper figure comparing only Leveling and MoE is `figures/fig_moe_caserta_horizons.pdf`; regenerate it with `figures/gen_fig_moe_caserta_horizons.py`.

For a group-by-group comparison over all 21 Caserta groups (40 instances per group), run:

```powershell
python -B -m experiments.summarize_moe_caserta_groups
```

The main table is `output/moe_lora_caserta_20260918/caserta_21_group_comparison.xlsx`. The same directory also contains `group_comparison.md`, `group_comparison.csv`, `group_seed_detail.csv`, and the machine-readable `group_comparison.json`. These outputs compare Leveling with the three final MoE seeds at H=0, 5, 10, and 20. Leveling uses only public stack heights, so its complete action sequence is horizon-invariant; the script rechecks this property on all 840 instances.

The publication-ready three-line table is `docs/tables/tab_moe_caserta_groups.tex`, with a rendered preview at `output/pdf/tab_moe_caserta_groups.pdf`. Run `python -B -m figures.gen_table_moe_caserta_groups` to regenerate the LaTeX source from the group-level results.

## 8. Tests and outputs

```powershell
python -B -m pytest -q -p no:cacheprovider
```

- `best.adapter.pt`: selected by validation mean `R/N`; used for final testing.
- `latest.adapter.pt`: last completed epoch; not used for the paper result.
- `metrics.jsonl`: epoch 0, each training epoch, and `run_complete`, including seeds, routing diagnostics, timing, and validation scores.
- `episode_results.jsonl`: relocations and runtime for every method/episode pair.
- `summary.json`: aggregate metrics, paired comparisons, and bootstrap intervals.

See [docs/README_EN.md](docs/README_EN.md) for the paper-source index. Historical code and engineering notes were moved outside the repository to `D:\DSAI\containter_new\archive`; they are not imported by the release workflow.
