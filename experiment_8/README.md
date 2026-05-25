# Experiment 8: Teacher Subspace Regularization

This experiment regularizes student weight matrices toward the teacher's principal left and right singular subspaces.

`subspace_reg.py` loads the teacher checkpoint, computes top-rank SVD bases for selected attention and MLP matrices, and penalizes the part of each student matrix that falls outside those subspaces:

- left residual: `W - U @ (U.T @ W)`
- right residual: `W - (W @ V) @ V.T`

The loss is added to normal cross-entropy with a warmup-scaled `SUBSPACE_LAMBDA`.

## How this came from experiment 7

Experiment 7 tested teacher SVD as an initialization-only prior. Experiment 8 tested the complementary reg-only ablation: do not copy teacher weights into the student, but keep applying pressure toward the teacher's learned subspaces during training.

This separates "good starting point" from "useful persistent constraint."

## What changed from experiment 7

- Added teacher subspace extraction from the same teacher checkpoint.
- Added a persistent auxiliary loss.
- Added logging for:
  - `ce_loss`
  - `subspace_loss`
  - `subspace_left`
  - `subspace_right`
  - active `subspace_lambda`

## How the teacher prior is created

The teacher prior is again a trained checkpoint, but this experiment uses it to define subspaces rather than initial weights. For each selected teacher matrix, the script runs SVD and keeps the top left and right singular vector bases.

Those bases represent the teacher's main input/output directions for the selected attention and MLP matrices.

## How the teacher is loaded into the experiment

`SUBSPACE_TEACHER_PATH` points at the checkpoint. `subspace_reg.py` extracts top-rank bases for shape-compatible student parameters and stores those bases as frozen reference tensors.

At each training step, the student matrix is projected onto the teacher's left and right subspaces. The loss penalizes the residual outside those subspaces, with `SUBSPACE_WARMUP_STEPS` optionally ramping the strength from zero to `SUBSPACE_LAMBDA`.

## Code changes from `train_gpt.py`

`../train_gpt.py` is the baseline comparison script. The meaningful changes in `experiment_8/subspace_reg.py` are:

- Added `SUBSPACE_REG_ENABLED`, `SUBSPACE_TEACHER_PATH`, `SUBSPACE_RANK`, `SUBSPACE_ATTENTION`, `SUBSPACE_MLP`, `SUBSPACE_LAMBDA`, left/right weights, and warmup controls.
- Added teacher checkpoint loading and SVD basis extraction.
- Added `compute_teacher_subspace_loss(...)` for left and right residual penalties.
- Added `subspace_reg_multiplier(...)` for warmup-scaled regularization.
- Changed the training objective to `CE + scaled_subspace_loss` when enabled.
- Logged CE, total subspace loss, left/right components, and active regularization strength.

## Important files

- `subspace_reg.py`: experiment script.

## How this led to experiment 9

After testing init-only and reg-only, the next logical ablation was to combine them.

That led to experiment 9: teacher SVD initialization plus teacher-subspace regularization in the same run.
