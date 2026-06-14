# Teacher Delta Activation Audit

## Metadata

- **model_config**:
  - `vocab_size`: `1024`
  - `num_layers`: `12`
  - `model_dim`: `768`
  - `num_heads`: `12`
  - `num_kv_heads`: `4`
  - `mlp_mult`: `2`
  - `seq_len`: `1024`
  - `audit_batch_tokens`: `65536`
  - `audit_num_batches`: `50`
  - `checkpoint`: `./large_teacher.pt`

## How to read the main metrics

- `constant_energy_fraction`: fraction of expected delta energy explained by the mean write.
- `raw_k90`: number of dimensions needed to explain 90% of raw expected delta energy.
- `centered_k90`: same, but after subtracting the mean write; this is the input-dependent part.
- `raw_pr` / `centered_pr`: participation ratio of raw vs centered energy distributions.

## Most suspicious constant/outlier rows

| component | layer | constant_frac | raw_k90 | centered_k90 | raw_pr | centered_pr | constant_top_dim | raw_top1_frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_delta | 0 | 0.990936 | 1 | 437 | 1.05 | 11.08 | 690 | 0.976385 |
| block_total_delta | 11 | 0.956880 | 1 | 1 | 1.00 | 1.01 | 690 | 0.999729 |
| mlp_delta | 11 | 0.955476 | 1 | 1 | 1.00 | 1.01 | 690 | 0.999758 |
| attn_delta | 1 | 0.882710 | 59 | 459 | 1.46 | 69.33 | 690 | 0.827923 |
| attn_delta | 11 | 0.830309 | 321 | 667 | 1.66 | 457.19 | 690 | 0.776158 |
| mix_delta | 1 | 0.736371 | 21 | 367 | 1.44 | 3.48 | 690 | 0.832648 |
| attn_delta | 8 | 0.674100 | 480 | 630 | 2.95 | 67.02 | 690 | 0.581869 |
| mix_delta | 9 | 0.672390 | 33 | 182 | 1.38 | 2.43 | 690 | 0.852673 |
| attn_delta | 9 | 0.656495 | 507 | 635 | 3.17 | 355.53 | 690 | 0.561373 |
| attn_delta | 7 | 0.650619 | 492 | 632 | 4.06 | 194.82 | 690 | 0.496077 |
| mix_delta | 5 | 0.643657 | 133 | 336 | 1.54 | 3.76 | 690 | 0.806944 |
| attn_delta | 10 | 0.637582 | 514 | 640 | 3.33 | 644.49 | 690 | 0.547936 |
| attn_delta | 6 | 0.568924 | 533 | 621 | 11.29 | 183.77 | 690 | 0.295829 |
| skip_delta | 5 | 0.556694 | 83 | 187 | 2.49 | 15.60 | 690 | 0.631868 |
| mix_delta | 4 | 0.552540 | 1 | 158 | 1.23 | 1.55 | 690 | 0.902927 |
| attn_delta | 3 | 0.548926 | 439 | 533 | 4.55 | 176.54 | 690 | 0.467939 |
| attn_delta | 5 | 0.515664 | 532 | 604 | 10.34 | 394.82 | 690 | 0.309484 |
| skip_delta | 0 | 0.497657 | 409 | 512 | 3.47 | 23.33 | 690 | 0.536345 |
| mix_delta | 11 | 0.477040 | 316 | 458 | 6.53 | 100.89 | 690 | 0.385696 |
| mix_delta | 6 | 0.470491 | 461 | 549 | 4.13 | 30.63 | 690 | 0.491699 |

## Quick interpretation

Use this decision rule:

- If `raw_k90` is tiny but `centered_k90` becomes much larger, the raw signal was mostly a constant/outlier write.
- If both `raw_k90` and `centered_k90` remain tiny, the input-dependent transformation itself is low-dimensional.
- If `constant_energy_fraction` is high in `mlp_delta`, the MLP is likely producing the massive dimension.
- If it is high in `skip_delta`, the skip pathway is importing/amplifying the outlier.
- If it is high in `block_total_delta` but not `attn_delta` or `mlp_delta`, inspect `mix_delta` and skip behavior.
