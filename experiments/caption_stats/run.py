#!/usr/bin/env python3
"""Caption stats — length, vocabulary, TTR, MTLD, word frequencies, word cloud.

Reproduces the caption-quality numbers cited in:
  - Section 5.2 (caption length + vocabulary characterization)
  - Section 7.1b (caption diversity: TTR, MTLD)
  - Figure "word_cloud.jpeg"

Single entry point. Reads data/labels/<md5>/0.json files, writes outputs under
results/caption_stats/. Deterministic with --seed.
"""

import argparse
import csv
import json
import random
import statistics
import string
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "labels"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "caption_stats"

PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, whitespace-split. Matches caption_diversity.py."""
    return text.lower().translate(PUNCT_TABLE).split()


def load_captions(labels_dir: Path) -> list[str]:
    captions = []
    for json_path in sorted(labels_dir.glob("*/0.json")):
        with open(json_path) as f:
            record = json.load(f)
        caption = (record.get("caption") or "").strip()
        if caption:
            captions.append(caption)
    return captions


def per_group_ttr(captions: list[str], group_size: int, rng: random.Random):
    """TTR pooled per group of `group_size` captions, after shuffling."""
    shuffled = captions[:]
    rng.shuffle(shuffled)
    ttrs, tok_per_group, uniq_per_group = [], [], []
    n_groups = len(shuffled) // group_size
    for i in range(n_groups):
        group = shuffled[i * group_size : (i + 1) * group_size]
        tokens = []
        for caption in group:
            tokens.extend(tokenize(caption))
        if not tokens:
            continue
        ttrs.append(len(set(tokens)) / len(tokens))
        tok_per_group.append(len(tokens))
        uniq_per_group.append(len(set(tokens)))
    return ttrs, tok_per_group, uniq_per_group


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).

    Returns the mean of forward and backward MTLD.
    """

    def one_direction(seq: list[str]) -> float:
        factors = 0.0
        window_counts: dict[str, int] = {}
        n = 0
        for tok in seq:
            window_counts[tok] = window_counts.get(tok, 0) + 1
            n += 1
            ttr = len(window_counts) / n
            if ttr <= threshold:
                factors += 1.0
                window_counts = {}
                n = 0
        if n > 0:
            ttr = len(window_counts) / n
            if ttr < 1.0:
                factors += (1 - ttr) / (1 - threshold)
        if factors == 0:
            return float("nan")
        return len(seq) / factors

    fwd = one_direction(tokens)
    bwd = one_direction(list(reversed(tokens)))
    return (fwd + bwd) / 2.0


def write_length_histogram(word_counts, mean_len, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping length histogram", file=sys.stderr)
        return
    plt.figure(figsize=(8, 5))
    plt.hist(word_counts, bins=50, color="#4d7c36", edgecolor="black")
    plt.axvline(mean_len, color="red", linestyle="--", label=f"mean = {mean_len:.1f}")
    plt.axvline(154, color="black", linestyle=":", label="paper claim = 154")
    plt.xlabel("Caption length (words)")
    plt.ylabel("Number of captions")
    plt.title(f"Caption length distribution (n={len(word_counts):,})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_word_cloud(captions: list[str], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from wordcloud import STOPWORDS, WordCloud
    except ImportError:
        print(
            "wordcloud not installed — skipping word cloud "
            "(install with: pip install wordcloud)",
            file=sys.stderr,
        )
        return
    stopwords = set(STOPWORDS)
    stopwords.update(
        ["video", "appears", "depicts", "features", "captures", "showcases"]
    )
    wc = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        colormap="plasma",
        relative_scaling=0.5,
        min_font_size=8,
        max_words=200,
        stopwords=stopwords,
        collocations=False,
    ).generate(" ".join(captions))
    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=100)
    parser.add_argument("--top-n-words", type=int, default=500)
    parser.add_argument("--mtld-threshold", type=float, default=0.72)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading captions from {args.labels_dir}")
    captions = load_captions(args.labels_dir)
    if not captions:
        print("No captions found.", file=sys.stderr)
        return 1
    print(f"Loaded {len(captions):,} captions")

    word_counts = [len(tokenize(c)) for c in captions]

    all_tokens: list[str] = []
    for caption in captions:
        all_tokens.extend(tokenize(caption))
    word_freq = Counter(all_tokens)
    vocab_size = len(word_freq)

    print("Computing per-group TTR ...")
    ttrs, tok_per_grp, uniq_per_grp = per_group_ttr(captions, args.group_size, rng)

    print("Computing MTLD (forward + backward) ...")
    mtld_value = mtld(all_tokens, args.mtld_threshold)

    mean_len = statistics.mean(word_counts)
    summary = {
        "n_captions": len(captions),
        "caption_length_words": {
            "mean": mean_len,
            "median": statistics.median(word_counts),
            "std": statistics.pstdev(word_counts),
            "min": min(word_counts),
            "max": max(word_counts),
        },
        "paper_claimed_mean_length": 154,
        "delta_vs_paper": mean_len - 154,
        "total_tokens": len(all_tokens),
        "vocabulary_size": vocab_size,
        "global_ttr": vocab_size / len(all_tokens),
        "group_ttr": {
            "group_size": args.group_size,
            "n_groups": len(ttrs),
            "mean": statistics.mean(ttrs) if ttrs else None,
            "median": statistics.median(ttrs) if ttrs else None,
            "std": statistics.pstdev(ttrs) if len(ttrs) > 1 else None,
            "tokens_per_group_mean": (
                statistics.mean(tok_per_grp) if tok_per_grp else None
            ),
            "unique_per_group_mean": (
                statistics.mean(uniq_per_grp) if uniq_per_grp else None
            ),
        },
        "mtld": {"value": mtld_value, "threshold": args.mtld_threshold},
        "seed": args.seed,
    }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    word_freq_path = args.output_dir / "word_freq.csv"
    with open(word_freq_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "word", "count", "fraction"])
        for rank, (word, count) in enumerate(
            word_freq.most_common(args.top_n_words), 1
        ):
            writer.writerow([rank, word, count, count / len(all_tokens)])

    write_length_histogram(
        word_counts, mean_len, args.output_dir / "length_histogram.png"
    )
    write_word_cloud(captions, args.output_dir / "word_cloud.png")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
