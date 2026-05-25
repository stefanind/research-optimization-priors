# Experiment 15: Big Teacher Transformation KD

## What this directory is doing

This directory contains the active implementation of big-teacher transformation KD.

`bigteacher_transformation_kd.py` compares the teacher and student transformation across configurable layer spans:

- `TRANSKD_STUDENT_BEFORE_LAYER`
- `TRANSKD_STUDENT_AFTER_LAYER`
- `TRANSKD_TEACHER_BEFORE_LAYER`
- `TRANSKD_TEACHER_AFTER_LAYER`

It supports:

- `TRANSKD_MODE=delta_rel`: match changes in token relation matrices.
- `TRANSKD_MODE=delta_dir`: match residual update direction structure.
- `TRANSKD_MODE=both`: combine both signals.

## How this came from experiment 14

Experiment 14 tested output-logit KD from a bigger teacher. Experiment 15 asks whether a more internal signal is better: not the teacher's final answer, and not just one hidden state, but the transformation the teacher applies between two hidden states.

The early transformation-KD runs exposed scale and diagnostic issues, especially around raw direction norms and normalized relation deltas. The current script includes clearer normalization controls and diagnostics for those cases.

## What changed from experiment 14

- Switched from logit KL to transformation-level hidden KD.
- Captured two hidden states from both student and teacher.
- Compared before/after changes rather than static hidden geometry.
- Added `TRANSKD_NORMALIZE_DELTA` for cosine-style normalized relation-delta matching.
- Added `TRANSKD_DIR_USE_RMSNORM` for computing direction deltas after RMSNorm.
- Logged raw and normalized direction norms separately.
- Logged relation norm ratios and direction norm ratios so scale mismatches are visible.
- Kept separate large-teacher shape controls.

## How the teacher signal is created

The teacher signal is produced from a frozen teacher checkpoint on each training batch. The script captures two hidden states from the student and two hidden states from the teacher:

- before layer for the span
- after layer for the span

It then compares the transformation across that span. `delta_rel` compares changes in token relation matrices before vs. after the span. `delta_dir` compares the relational structure of the residual update direction, optionally after RMSNorm.

## How the teacher is loaded into the experiment

`TRANSKD_TEACHER_PATH` points at the teacher checkpoint, defaulting to `./large_teacher.pt`. `bigteacher_transformation_kd.py` constructs the teacher with separate architecture controls, loads the weights, freezes the model, and evaluates it under `torch.no_grad()`.

`forward_hiddens(input_ids, hidden_layers)` returns the requested before/after hidden states. Student and teacher can use different layer endpoints through the `TRANSKD_STUDENT_*` and `TRANSKD_TEACHER_*` controls.

## Code changes from `train_gpt.py`

`../train_gpt.py` is the baseline comparison script. The meaningful changes in `experiment_15/bigteacher_transformation_kd.py` are:

- Added `TRANSKD_MODE`, `TRANSKD_TEACHER_PATH`, `TRANSKD_LAMBDA`, per-mode lambdas, span endpoints, token subsampling controls, normalization controls, and separate teacher shape controls.
- Added `forward_hiddens(...)` to capture multiple hidden states in one forward pass.
- Added `delta_relation_loss(...)` for before/after relation-matrix changes.
- Added `residual_direction_relation_loss(...)` for residual update direction geometry.
- Added `transformation_kd_loss(...)` to combine `delta_rel`, `delta_dir`, or both.
- Constructed and froze a larger teacher model with independently configured shape.
- Changed the training objective to `CE + TRANSKD_LAMBDA * transformation_KD`.
- Logged relation norms, direction norms, norm ratios, and per-mode KD components.

## Important files

- `../train_gpt.py`: baseline script used for comparison against this and later experiments.
- `bigteacher_transformation_kd.py`: active implementation.
- `logs/transkd_delta_rel_s6_8_t6_8_lam001.txt`
- `logs/transkd_delta_rel_s6_8_t6_8_lam001_norm.txt`
- `logs/transkd_delta_rel_s6_8_t6_8_lam01.txt`
- `logs/transkd_delta_rel_s6_8_t6_8_lam01_norm.txt`
- `logs/transkd_delta_rel_s6_8_t6_8_lam1.txt`
- `logs/transkd_delta_rel_s6_8_t6_8_lam10.txt`
- `logs/transkd_delta_dir_s6_8_t6_8_lam001.txt`
- `logs/transkd_delta_dir_s6_8_t6_8_lam01.txt`

## Current role in the logical flow

This is the latest transformation-KD experiment in the sequence. The path to here is:

1. Start with fixed data priors.
2. Try trainable data-prior modules.
3. Move to teacher weight priors.
4. Compare teacher hidden geometry, embedding-anchor geometry, and logits.
5. Refine the teacher signal down to transformations between layers.

The next decision should come from controlled comparisons between the best logit KD, relational KD, and transformation KD settings under the same budget.
