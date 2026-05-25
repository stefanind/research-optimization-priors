# Experiment 0: Baseline and Reference Runs

This directory is the reference point for the whole sequence. It keeps the unmodified GPT training script and the logs used to decide whether later ideas were helping.

The baseline model is the small GPT setup:

- 9 transformer layers
- width 512
- 8 attention heads with 4 KV heads
- 1024-token SentencePiece vocabulary
- tied embeddings
- Muon for matrix parameters and Adam for embeddings/scalars
- 10 minute training cap for normal comparison runs

The small teacher checkpoint used by experiments 6-12 lives at `./small_teacher.pt`; the logs in this folder document the baseline runs that produced the reference model family.

## Baseline code relationship

The root `../train_gpt.py` is the canonical baseline used to compare against every later experiment. The copy in this directory preserves that baseline training setup alongside the logs from baseline runs.

Later experiment scripts are best read as modified copies of `../train_gpt.py`: they keep the same tokenizer, dataloading style, GPT block structure, optimizer split, validation loop, and checkpoint/logging conventions, then add one research intervention at a time.

The baseline has no data-prior loader, teacher checkpoint loader, auxiliary KD loss, or extra adapter path. Its training objective is plain next-token cross-entropy, so it is the control condition for asking whether each added signal actually helps.

## Important files

- `train_gpt.py`: baseline training script.
- `teacher_9l_512_10m.txt`: main real baseline comparison log.
- `teacher_12l_768_60m.txt`: larger teacher model run log kept for comparison with later larger-teacher experiments.
