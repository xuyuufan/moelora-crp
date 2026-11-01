# 2026-09-17 MoE focus archive

This archive keeps non-MoE experiment outputs out of the active `output/`
directory after selecting MoE-LoRA as the continuing research direction.
Nothing in this operation was deleted.

## Archived experiment outputs

| Previous path | Archived path | Files | Bytes |
| --- | --- | ---: | ---: |
| `output/caserta_horizon_eval_20260904` | `archive/20260917_moe_focus/output/caserta_horizon_eval_20260904` | 4 | 1,196,629 |
| `output/h20_large_scale_lora_20260915_retry1` | `archive/20260917_moe_focus/output/h20_large_scale_lora_20260915_retry1` | 24 | 2,554,520 |
| `output/h20_large_scale_lora_confirm_20260916` | `archive/20260917_moe_focus/output/h20_large_scale_lora_confirm_20260916` | 20 | 2,488,029 |
| `output/h20_original` | `archive/20260917_moe_focus/output/h20_original` | 4 | 3,740,447 |
| `output/lora_stack_r16_20260917_seed20260913` | `archive/20260917_moe_focus/output/lora_stack_r16_20260917_seed20260913` | 6 | 530,568 |

Total: 58 files, 10,510,193 bytes.

## Kept active

- All `output/moe_lora_*` run directories and MoE multi-seed queue logs/status.
- `checkpoints/h20.best.pt`, because it is the shared base checkpoint used by
  MoE-LoRA.
- `data/online_crp/lora_domains/large_scale_n100_150`, source code, tests, and
  current documentation.
- `output/pdf`, because it contains current paper/documentation renderings, not
  an experiment run.

## Restoration

To restore an archived run, move its directory from this archive's `output/`
directory back to the repository's active `output/` directory. Check that the
destination does not already exist before restoring it.
