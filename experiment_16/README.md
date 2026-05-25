# Experiment 17: Teacher Block Norm Diagnostics

This directory is a diagnostic follow-up to the teacher transformation KD work. It does not train a student. Instead, `diagnostics.ipynb` contains a self-contained teacher model definition and instrumentation for inspecting norm flow through a trained teacher checkpoint.

The notebook/script path reports per-layer quantities such as:

- residual stream norms before and after each block
- skip-path update norms
- attention and MLP update norms
- raw block delta norms
- RMS-normalized block delta norms
- control parameter magnitudes such as scale and residual mix values

## How this came from experiment 15

Experiment 15 showed some interesting norm differences between layer transforms, but the logs still raised a scale question: are the large transformation signals coming from real directional changes, residual stream scale growth, skip connections, residual mixing, attention updates, or MLP updates?

Experiment 16 isolates that question by inspecting the teacher itself. It is a diagnostic bridge for deciding how to interpret and normalize teacher transformation signals in future runs.

## Interpretations

The last layer for both small teacher and big teacher have most their magnitude added in and written into the residual stream (23x the magnitude than the previous layer for the big teacher). The layers before have a consistent increase layer over layer. 

Diving in further, I checked to see if the magnitude is added more to any particular token versus others, but found that it is spread evenly enough. Not uniformly, e.g., the top 1% of tokens account for 2.1% of the magnitude.  

Now going even deeper, I checked out to see if within each token's hidden state, if it is spread across the hidden dim or concentrated. 1% of the features received 99.8% of the magnitude increase! I then checked if this behaviour is consistent across layers, i.e., if 1% of the hidden dims of a token get most of the magnitude and it is consistent, but not as strong as the last layer.  