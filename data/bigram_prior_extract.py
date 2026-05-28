import glob
import numpy as np
import argparse


def load_tokens(file_path):
    header_bytes = 256 * 4
    return np.fromfile(file_path, dtype=np.uint16, offset=header_bytes)


def compute_bigram_log_probs(pattern, vocab_size, max_files=None):
    """
    One-pass computation of smoothed log P(j | i)
    plus per-row default log-prob for unseen bigrams.
    """
    files = sorted(glob.glob(pattern))
    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise ValueError(f"No files matched pattern: {pattern}")

    print(f"Found {len(files)} files")

    pair_counts = np.zeros(vocab_size * vocab_size, dtype=np.int64)
    row_counts = np.zeros(vocab_size, dtype=np.int64)
    total_tokens = 0

    for file_idx, file_path in enumerate(files):
        tokens = load_tokens(file_path)
        total_tokens += len(tokens)

        if len(tokens) < 2:
            continue

        t0 = tokens[:-1].astype(np.int64)
        t1 = tokens[1:].astype(np.int64)

        row_counts += np.bincount(t0, minlength=vocab_size)

        pair_ids = t0 * vocab_size + t1
        local_counts = np.bincount(pair_ids, minlength=vocab_size * vocab_size)
        pair_counts += local_counts

        if file_idx % 10 == 0:
            print(f"[{file_idx}/{len(files)}] tokens: {total_tokens:,}")

    print(f"Total tokens processed: {total_tokens:,}")

    alpha_smooth = 0.1

    nonzero = pair_counts.nonzero()[0]
    rows = nonzero // vocab_size
    cols = nonzero % vocab_size
    counts = pair_counts[nonzero].astype(np.float32)

    row_totals = row_counts[rows].astype(np.float32) + alpha_smooth * vocab_size
    probs = (counts + alpha_smooth) / row_totals
    log_probs = np.log(probs)
    log_probs = np.clip(log_probs, -10.0, 0.0)

    default_log_probs = np.log(
        alpha_smooth / (row_counts.astype(np.float32) + alpha_smooth * vocab_size)
    )
    default_log_probs = np.clip(default_log_probs, -10.0, 0.0)

    return rows, cols, log_probs, default_log_probs


def save_prior(rows, cols, log_probs, default_log_probs, output_path):
    np.savez(
        output_path,
        rows=rows,
        cols=cols,
        log_probs=log_probs,
        default_log_probs=default_log_probs,
    )

    print(f"Saved bigram prior to {output_path}")
    print(f"Nonzero entries: {len(log_probs):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output", type=str, default="bigram_prior.npz")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    rows, cols, log_probs, default_log_probs = compute_bigram_log_probs(
        args.pattern,
        vocab_size=args.vocab_size,
        max_files=args.max_files,
    )

    save_prior(rows, cols, log_probs, default_log_probs, args.output)


if __name__ == "__main__":
    main()