# Experiment 11: Activation Embedding KL

This experiment distills hidden states through their similarity to teacher token embeddings.

`teacher_embd_kl.py` captures selected hidden layers from student and teacher. For each hidden vector, it computes similarities to the teacher's token embedding matrix, turns those similarities into a soft distribution, and applies a KL loss between the student and teacher distributions.

This can be read as anchor-neighborhood distillation: the student should place each hidden state near the same teacher-token anchors as the teacher does.

## How this came from experiment 10

Experiment 10 compared hidden states by their pairwise geometry within a batch or sequence. Experiment 11 asked whether a more stable reference frame would help: instead of comparing each token only to other current tokens, compare it to the teacher's whole embedding vocabulary.

That keeps the teacher signal representation-level, but changes the coordinate system.

## What changed from experiment 10

- Added `EMBED_KD_LAYER_PAIRS`.
- Added `EMBED_KD_LAMBDA`, `EMBED_KD_TEMP`, and `EMBED_KD_TOPK`.
- Used teacher token embeddings as anchors.
- Added optional relative magnitude matching through `EMBED_MAG_LAMBDA`.

## How the teacher signal is created

The signal comes from a frozen teacher checkpoint plus the teacher's token embedding matrix. For each selected layer pair, the script asks: which teacher-token anchors does this hidden state look closest to?

That turns each hidden vector into a distribution over teacher vocabulary anchors, then distills that distribution with KL.

## How the teacher is loaded into the experiment

`TEACHER_PATH` points at the checkpoint. `teacher_embd_kl.py` loads and freezes the teacher, captures selected hidden states from both student and teacher, and reads `teacher_model.tok_emb.weight` as the anchor matrix.

For each hidden vector, both student and teacher similarities are computed against the same normalized teacher-anchor matrix. `EMBED_KD_TOPK` can restrict the KL to the teacher's strongest anchors. `EMBED_MAG_LAMBDA` optionally adds relative norm matching.

## Code changes from `train_gpt.py`

`../train_gpt.py` is the baseline comparison script. The meaningful changes in `experiment_11/teacher_embd_kl.py` are:

- Added `TEACHER_PATH`, `EMBED_KD_LAYER_PAIRS`, `EMBED_KD_LAMBDA`, `EMBED_KD_TEMP`, `EMBED_KD_TOPK`, and `EMBED_MAG_LAMBDA`.
- Added `forward_with_states(...)` to capture selected student and teacher hidden states.
- Loaded and froze the teacher checkpoint.
- Added `activation_embedding_kd_loss(...)` using the teacher token embedding matrix as fixed anchors.
- Added optional top-k teacher-anchor selection and relative magnitude matching.
- Changed the training objective to `CE + embedding_anchor_KD + optional_magnitude_loss`.
- Logged anchor KL and magnitude loss separately.

## Important files

- `teacher_embd_kl.py`: activation embedding KD script.

## How this led to experiment 12

Experiments 10 and 11 explored hidden-state teacher signals. I noticed that both show some preliminary results, but neither one better than the other. This is when I wanted to test the techniques against a baseline that is proven to work, classic logit kd. 

That led to experiment 12.
