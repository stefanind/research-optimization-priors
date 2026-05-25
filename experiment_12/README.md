# Experiment 12: Classic Same-Size Logit KD

This experiment implements classic teacher-student logit distillation. `classic_logit_kd.py` loads a frozen teacher with the same architecture as the student and adds KL between softened teacher logits and student logits:

`CE(student, targets) + LOGIT_KD_LAMBDA * KL(teacher_logits/T || student_logits/T)`

The temperature is controlled by `LOGIT_KD_TEMP`.

## How this came from experiment 11

Experiment 10 tested hidden relational geometry. Experiment 11 tested hidden-to-embedding anchor distributions. Experiment 12 established the standard output distribution baseline so the hidden state methods could be compared against a familiar teacher signal.

This acts sort of a diagnostic to see if the techniques beat classic logit kd.

## What changed from experiment 11

- Removed hidden-state relational or anchor losses as the main path.
- Used teacher logits directly.
- Added `LOGIT_KD_TEACHER_PATH`, `LOGIT_KD_LAMBDA`, and `LOGIT_KD_TEMP`.

## How the teacher signal is created

The teacher signal is the teacher's output distribution over the vocabulary for the same batch of input tokens. The teacher checkpoint is trained beforehand, then frozen during this experiment.

Temperature `LOGIT_KD_TEMP` softens both distributions before KL, making lower-ranked teacher preferences visible instead of only the hard next-token target.

## How the teacher is loaded into the experiment

`LOGIT_KD_TEACHER_PATH` points at a same-size checkpoint. `classic_logit_kd.py` constructs the same architecture as the student, loads the teacher weights, freezes the teacher, and evaluates it under `torch.no_grad()`.

The student and teacher both expose `forward_logits(...)`. The student CE is computed from its logits, and the KD term is `KL(teacher_logits / T || student_logits / T) * T^2`.

## Code changes from `train_gpt.py`

`../train_gpt.py` is the baseline comparison script. The meaningful changes in `experiment_12/classic_logit_kd.py` are:

- Added `LOGIT_KD_TEACHER_PATH`/`TEACHER_PATH`, `LOGIT_KD_LAMBDA`, and `LOGIT_KD_TEMP`.
- Added `GPT.forward_logits(...)` so logits can be reused for CE and KD.
- Added teacher checkpoint loading, freezing, and eval-mode setup.
- Added `logit_kl_distill_loss(...)`.
- Used a direct student forward path for logit KD instead of the compiled CE-only model path.
- Changed the training objective to `CE + LOGIT_KD_LAMBDA * logit_KD`.
- Logged `logit_kd` separately from CE.

## Important files

- `classic_logit_kd.py`: same-size logit KD script.

## How this led to experiment 13

Classic logit KD didn't perform nearly as well as I hoped. During the first 600 steps of optimization, it performed much worse than the techniques I developed in experiments 10 and 11. This made be suspicious. I concluded that the teacher isn't big enough to provide a strong signal that the student could rely on. I also wanted to see if the techniques from 10 and 11 performed better when the signal is given from a larger teacher. 
