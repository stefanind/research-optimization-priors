import glob
import numpy as np
import argparse
import torch


def load_tokens(file_path):
    header_bytes = 256 * 4
    return np.fromfile(file_path, dtype=np.uint16, offset=header_bytes)


def compute_unigram_probs(pattern, vocab_size, max_files=None, alpha_smooth=0.1):
    """
    One-pass computation of smoothed unigram probabilities P(token=i).

    Returns:
        probs: np.ndarray of shape [vocab_size], dtype float32
    """
    files = sorted(glob.glob(pattern))
    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise ValueError(f"No files matched pattern: {pattern}")

    print(f"Found {len(files)} files")

    counts = np.zeros(vocab_size, dtype=np.int64)
    total_tokens = 0

    for file_idx, file_path in enumerate(files):
        tokens = load_tokens(file_path)
        total_tokens += len(tokens)

        if len(tokens) == 0:
            continue

        counts += np.bincount(tokens.astype(np.int64), minlength=vocab_size)

        if file_idx % 10 == 0:
            print(f"[{file_idx}/{len(files)}] tokens: {total_tokens:,}")

    print(f"Total tokens processed: {total_tokens:,}")

    denom = total_tokens + alpha_smooth * vocab_size
    probs = (counts.astype(np.float32) + alpha_smooth) / denom

    # Safety check
    probs = probs.astype(np.float32)
    probs /= probs.sum(dtype=np.float64)

    return probs


def save_unigram_probs(probs, output_path):
    """
    Save as a torch tensor so your training script can load it with torch.load(...).
    """
    tensor = torch.from_numpy(probs).float()
    torch.save(tensor, output_path)

    print(f"Saved unigram probs to {output_path}")
    print(f"Shape: {tuple(tensor.shape)}")
    print(f"Sum: {tensor.sum().item():.8f}")
    print(f"Min prob: {tensor.min().item():.8e}")
    print(f"Max prob: {tensor.max().item():.8e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output", type=str, default="unigram_probs.pt")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--alpha-smooth", type=float, default=0.1)
    args = parser.parse_args()

    probs = compute_unigram_probs(
        args.pattern,
        vocab_size=args.vocab_size,
        max_files=args.max_files,
        alpha_smooth=args.alpha_smooth,
    )

    save_unigram_probs(probs, args.output)


if __name__ == "__main__":
    main()