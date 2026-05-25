# Experiment 2: Bigram Prior KL Regularization

This experiment turns the bigram prior from experiment 1 into a teacher distribution. Instead of adding bigram logits to the model logits, `bigram_kl.py` adds an auxiliary KL term:

`CE(model, targets) + weight(t) * KL(P_bigram_teacher || P_model)`

The prior still decays over training, but now it guides the model through the loss rather than directly changing the predictions.

## How this came from experiment 1

Experiment 1 tested the strongest possible data-prior intervention: direct injection. The issue is that once this was removed or the decay is stronger, the loss would jump back up and so the model had to learn this info anyways. I thought it would learn it indirectly along with more nuance information the data provides. 

Therefore, instead of adding it straight into the logits, I added a KL term. 

## What changed from experiment 1

- Removed direct prior-logit injection as the main mechanism.
- Added `KL(P_bigram_teacher || P_model)`.
- Added logging for separate loss components:
  - `train_total`
  - `train_ce`
  - `train_raw_kl`
  - `train_kl`
  - `kl_coeff`


## Code changes from `train_gpt.py`

`../train_gpt.py` is the baseline comparison script. The meaningful changes in `experiment_2/bigram_kl.py` are:

- Added `BIGRAM_PRIOR_PATH`, `PRIOR_KL_MAX_STEP`, and `PRIOR_KL_WEIGHT_START`.
- Added a `BigramPrior` loader for the sparse `.npz` prior.
- Added `get_prior_kl_weight(...)` for scheduled KL strength.
- Changed `GPT.forward(...)` to accept `prior_kl_weight` and return total, CE, and KL losses.
- Computed `KL(P_bigram_teacher || P_model)` with `F.kl_div(..., log_target=True)`.
- Added the KL term to cross-entropy during training only when the prior and weight are active.
- Logged `prior_kl_loss`, `prior_kl_weight`, and whether the prior path is active.

## Important files

- `../data/bigram_prior_extract.py`: creates the sparse smoothed bigram prior file.
- `bigram_kl.py`: experiment script.

## How this led to experiment 3

Only in hindsight do I realize that this experiment is basically the exact same thing as learning it through baseline training, i.e., adding a KL term for a bigram distribution is exactly the same thing as simply letting the model learn this information within the first 100-200 training steps.

As a result, I asked myself if there was another way to do a form of injection but have it built into the model. I decided a very quick test would be having simple unigram statistics embedded into the output head. I chose unigram over the bigram because of memory efficiency, i.e., the bigram table would add 1M parameters instead of 1024 for the unigram. Even if a bigram is helpful, it would add too much memory to be feasible. 
