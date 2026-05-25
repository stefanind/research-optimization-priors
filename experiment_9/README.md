# Experiment 9: Teacher SVD Init Plus Subspace Regularization

This experiment combines the two teacher-weight priors from experiments 7 and 8:

- initialize selected student matrices with rank-limited teacher SVD reconstructions
- keep a persistent subspace regularizer active during training

`init_plus_reg.py` contains both the `SVD_INIT_*` controls and the `SUBSPACE_*` controls. It applies teacher SVD initialization first, then extracts teacher principal subspaces and trains with the auxiliary subspace loss.

## How this came from experiment 8

Experiment 7 asked whether teacher geometry helps as initialization. Experiment 8 asked whether teacher geometry helps as an ongoing constraint. Experiment 9 closes the ablation loop by testing the combination.

This is the "init + reg" condition from the original teacher-subspace plan.

## What changed from experiment 8

- Added the SVD initialization path back into the subspace-regularized script.
- Applied initialization before training and before regularization.
- Logged both initialized tensor names and regularized tensor names.

## How the teacher prior is created

This experiment uses one teacher checkpoint in two ways:

- truncated SVD reconstructions initialize selected student matrices
- top teacher singular-vector bases define a persistent subspace regularizer

Both pieces come from the teacher's attention and MLP weight matrices, so this is a combined static-weight teacher prior.

## How the teacher is loaded into the experiment

`init_plus_reg.py` first loads `SVD_INIT_PATH` to copy rank-limited reconstructions into shape-compatible student parameters. It then loads or reuses the teacher source for `SUBSPACE_TEACHER_PATH` and extracts top-rank left/right bases for regularization.

The order matters: initialization happens before training begins, then the subspace loss stays active during training with the configured warmup and lambda.

## Code changes from `train_gpt.py`

`../train_gpt.py` is the baseline comparison script. The meaningful changes in `experiment_9/init_plus_reg.py` are:

- Combined the `SVD_INIT_*` controls from experiment 7 with the `SUBSPACE_*` controls from experiment 8.
- Added teacher SVD reconstruction and copy-in before optimizer/training.
- Added teacher subspace basis extraction for the same selected matrix families.
- Added persistent subspace loss on top of cross-entropy.
- Logged both initialized tensors and regularized tensors, so the init-only and reg-only pieces can be audited together.

## Important files

- `../train_gpt.py`: baseline script used for comparison against this and later experiments.
- `init_plus_reg.py`: combined init-plus-regularization script.

## How this led to experiment 10

Experiments 7 through 9 focused on teacher weight geometry. The next question was whether the best teacher signal is not in the weights at all, but in the hidden representations produced on actual data.

That led to experiment 10: relational KD on teacher and student hidden states.
