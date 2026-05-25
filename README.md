# Optimization Priors for Small Language Models

This repository contains independent research on optimization priors, which are either teacher- or data-derived signals for transformers.

The project began as an exploration of OpenAI's Parameter Golf challenge, and it continues to use the same `train_gpt.py` baseline, tokenizer workflow, and FineWeb data pipeline as a controlled experimental substrate. It is not an official OpenAI project and is not affiliated with or endorsed by OpenAI.

## Overall Research Question

What kinds of prior signals can help a small language model learn or optimize more effectively?

This project began with a narrower question: can we embed useful data statistics into a model before training, so the model does not have to spend capacity rediscovering them from scratch?

Over the experiments, that question broadened into a study of teacher-assisted optimization: what is the most useful signal a teacher model can provide to a smaller student? The experiments compare data-derived priors, weight and subspace initialization, hidden-state geometry, transformation matching, and classic logit distillation as different ways of biasing the student toward better learning.


## Repository Layout

- `train_gpt.py`: canonical baseline training script taken from OpenAI Parameter Golf.
- `data/`: scripts and metadata for downloading/rebuilding the FineWeb-derived shards and tokenizer.
- `experiment_0_baseline/`: control runs.
- `experiment_1` ... `experiment_17`: research interventions, notes, and logs.

## Flow to the Experiments

1. Experiments 1-5 use data priors.
2. Experiments 7-9 use teacher weight priors.
3. Experiments 6, and 10-15 use teacher directional transforms, hidden geometry, embedding-anchor geometry, and logits.
4. Experiment 16 is a diagnostic due to Experiment 15 to decide how to go further.

## Provenance

This work builds on:
- OpenAI Parameter Golf baseline code, MIT license.
- Parameter Golf FineWeb export, ODC-By 1.0.
- Hugging Face FineWeb, ODC-By 1.0.
- modded-nanogpt lineage, MIT license.
