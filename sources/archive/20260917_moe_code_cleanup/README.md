# MijnRL MoE code cleanup archive

Archived on 2026-09-17 from `D:\DSAI\containter_new\Mijn_RL`.

These files were removed from the active release because they implement the
superseded generic LoRA workflow, original base-policy training, broad legacy
evaluation/benchmark tooling, or engineering planning notes. They are kept
here only as history and are not part of the MoE-LoRA reproduction path.

The relative directory layout below matches the original repository. The
active public workflow is now:

1. `experiments/train_moe_lora.py`
2. `experiments/evaluate_moe_lora.py`
3. `experiments/reproduce.py`

Validation before archival: seven focused MoE unit tests passed, released
CUDA quick evaluation matched all 20 corresponding historical episode results,
and the one-update CUDA reproduction pipeline completed end to end.
