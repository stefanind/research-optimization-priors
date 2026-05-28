# Experiment 15: Big Teacher Transformation KD

This directory contains the active implementation of big teacher transformation KD.

`bigteacher_transformation_kd.py` compares the teacher and student transformation across configurable layer spans:

- `TRANSKD_STUDENT_BEFORE_LAYER`
- `TRANSKD_STUDENT_AFTER_LAYER`
- `TRANSKD_TEACHER_BEFORE_LAYER`
- `TRANSKD_TEACHER_AFTER_LAYER`

It supports:

- `TRANSKD_MODE=delta_rel`: match changes in token relation matrices.
- `TRANSKD_MODE=delta_dir`: match residual update direction structure.
- `TRANSKD_MODE=both`: combine both signals.

Experiment 15 asks whether a more internal signal is better: not the teacher's final answer, and not just one hidden state, but the transformation the teacher applies between two hidden states.


## How the teacher signal is created

The teacher signal is produced from a frozen teacher checkpoint on each training batch. The script captures two hidden states from the student and two hidden states from the teacher:

- before layer for the span
- after layer for the span

It then compares the transformation across that span. `delta_rel` compares changes in token relation matrices before vs. after the span. `delta_dir` compares the relational structure of the residual update direction, optionally after RMSNorm.

## Code changes from `train_gpt.py`

`../train_gpt.py` is the baseline comparison script. The meaningful changes in `experiment_15/bigteacher_transformation_kd.py` are:

- Added `TRANSKD_MODE`, `TRANSKD_TEACHER_PATH`, `TRANSKD_LAMBDA`, per-mode lambdas, span endpoints, token subsampling controls, normalization controls, and separate teacher shape controls.
- Added `forward_hiddens(...)` to capture multiple hidden states in one forward pass.
- Added `delta_relation_loss(...)` for before/after relation matrix changes.
- Added `residual_direction_relation_loss(...)` for residual update direction geometry.
- Added `transformation_kd_loss(...)` to combine `delta_rel`, `delta_dir`, or both.
- Constructed and froze a larger teacher model with independently configured shape.
- Changed the training objective to `CE + TRANSKD_LAMBDA * transformation_KD`.
- Logged relation norms, direction norms, norm ratios, and per-mode KD components.

## How this led to Experiment 16  

Between layer transforms, I noticed huge average norms for each token hidden state coming from the teacher. For instance, outside layer 8, the avg token norm being written into the residual stream was 4,000,000. I didn't fully understand this signal so I wanted to explore further. This is why experiment 16 is more of a diagnostic.
