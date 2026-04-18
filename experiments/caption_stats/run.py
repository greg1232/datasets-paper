#!/usr/bin/env python3
"""Caption stats — length, vocabulary, TTR, MTLD, word frequencies, word cloud.

Reproduces the caption-quality numbers cited in:
  - Section 5.2 (caption length + vocabulary characterization)
  - Section 7.1b (caption diversity: TTR, MTLD) — including side-by-side
    comparison against InternVid and OpenVid-1M samples streamed from
    HuggingFace.

Single entry point. Reads data/labels/<md5>/0.json files, streams comparator
datasets from HF, writes outputs under results/caption_stats/. Deterministic
with --seed. Pass --no-hf to skip the HF comparison arm.
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

HF_COMPARATORS = [
    {
        "name": "InternVid",
        "path": "OpenGVLab/InternVid",
        "subset": "InternVid-10M",
        "split": "FLT",
        "caption_column": "Caption",
    },
    {
        "name": "OpenVid-1M",
        "path": "nkp37/OpenVid-1M",
        "subset": None,
        "split": "train",
        "caption_column": "caption",
    },
]


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, whitespace-split. Matches caption_diversity.py."""
    return text.lower().translate(PUNCT_TABLE).split()


def load_local_captions(labels_dir: Path) -> list[str]:
    captions = []
    for json_path in sorted(labels_dir.glob("*/0.json")):
        with open(json_path) as f:
            record = json.load(f)
        caption = (record.get("caption") or "").strip()
        if caption:
            captions.append(caption)
    return captions


def stream_hf_captions(
    path: str,
    subset: str | None,
    split: str,
    caption_column: str,
    sample_size: int,
    seed: int,
    buffer_size: int = 20_000,
) -> list[str]:
    """Stream up to `sample_size` captions from a HuggingFace dataset."""
    from datasets import load_dataset

    dataset = load_dataset(path, subset, streaming=True, split=split)
    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    captions: list[str] = []
    for i, sample in enumerate(dataset):
        if len(captions) >= sample_size:
            break
        text = sample.get(caption_column, "")
        if isinstance(text, str) and text.strip():
            captions.append(text.strip())
        if i >= sample_size * 4:  # safety bound for sparse streams
            break
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


def compute_stats(
    name: str,
    captions: list[str],
    group_size: int,
    mtld_threshold: float,
    rng: random.Random,
) -> dict:
    """Compute the full set of metrics for one dataset's caption list."""
    if not captions:
        return {"name": name, "n_captions": 0, "note": "no captions loaded"}
    word_counts = [len(tokenize(c)) for c in captions]
    all_tokens: list[str] = []
    for c in captions:
        all_tokens.extend(tokenize(c))
    vocab = Counter(all_tokens)
    ttrs, tok_per_grp, uniq_per_grp = per_group_ttr(captions, group_size, rng)
    return {
        "name": name,
        "n_captions": len(captions),
        "total_tokens": len(all_tokens),
        "vocabulary_size": len(vocab),
        "caption_length_words": {
            "mean": statistics.mean(word_counts),
            "median": statistics.median(word_counts),
            "std": statistics.pstdev(word_counts),
            "min": min(word_counts),
            "max": max(word_counts),
        },
        "global_ttr": len(vocab) / len(all_tokens),
        "group_ttr": {
            "group_size": group_size,
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
        "mtld": {"value": mtld(all_tokens, mtld_threshold), "threshold": mtld_threshold},
    }


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


def write_word_freq_csv(all_tokens: list[str], top_n: int, output_path: Path) -> None:
    word_freq = Counter(all_tokens)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "word", "count", "fraction"])
        for rank, (word, count) in enumerate(word_freq.most_common(top_n), 1):
            writer.writerow([rank, word, count, count / len(all_tokens)])


def write_comparison_csv(stats_list: list[dict], output_path: Path) -> None:
    columns = [
        "Dataset",
        "N Captions",
        "Total Tokens",
        "Vocabulary Size",
        "Mean Length",
        "Median Length",
        "Global TTR",
        "Group TTR (mean)",
        "MTLD",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for s in stats_list:
            if s.get("n_captions", 0) == 0:
                writer.writerow([s["name"], 0, "", "", "", "", "", "", ""])
                continue
            writer.writerow(
                [
                    s["name"],
                    s["n_captions"],
                    s["total_tokens"],
                    s["vocabulary_size"],
                    f"{s['caption_length_words']['mean']:.2f}",
                    s["caption_length_words"]["median"],
                    f"{s['global_ttr']:.4f}",
                    f"{s['group_ttr']['mean']:.4f}"
                    if s["group_ttr"]["mean"] is not None
                    else "",
                    f"{s['mtld']['value']:.2f}",
                ]
            )


def write_comparison_markdown(stats_list: list[dict], output_path: Path) -> None:
    header = (
        "| Dataset | N | Tokens | Vocab | Mean len | Median len | Global TTR | "
        "Group TTR | MTLD |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for s in stats_list:
        if s.get("n_captions", 0) == 0:
            rows.append(f"| {s['name']} | 0 | | | | | | | |")
            continue
        rows.append(
            f"| {s['name']} | {s['n_captions']:,} | {s['total_tokens']:,} | "
            f"{s['vocabulary_size']:,} | "
            f"{s['caption_length_words']['mean']:.1f} | "
            f"{s['caption_length_words']['median']} | "
            f"{s['global_ttr']:.4f} | "
            f"{s['group_ttr']['mean']:.4f} | "
            f"{s['mtld']['value']:.2f} |"
        )
    with open(output_path, "w") as f:
        f.write(header + "\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=100)
    parser.add_argument("--top-n-words", type=int, default=500)
    parser.add_argument("--mtld-threshold", type=float, default=0.72)
    parser.add_argument(
        "--hf-sample-size",
        type=int,
        default=10_000,
        help="Captions to stream per HF comparator (default 10k)",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Skip InternVid / OpenVid-1M comparison arm",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading local captions from {args.labels_dir}")
    ours = load_local_captions(args.labels_dir)
    if not ours:
        print("No local captions found.", file=sys.stderr)
        return 1
    print(f"Loaded {len(ours):,} local captions")

    print("Computing stats for PeoplesVideo ...")
    our_stats = compute_stats(
        "PeoplesVideo (8K subset)",
        ours,
        args.group_size,
        args.mtld_threshold,
        rng,
    )
    our_stats["paper_claimed_mean_length"] = 154
    our_stats["delta_vs_paper"] = our_stats["caption_length_words"]["mean"] - 154

    all_stats = [our_stats]

    if not args.no_hf:
        for cfg in HF_COMPARATORS:
            print(
                f"Streaming {cfg['name']} from HuggingFace "
                f"({args.hf_sample_size:,} captions) ..."
            )
            try:
                captions = stream_hf_captions(
                    cfg["path"],
                    cfg["subset"],
                    cfg["split"],
                    cfg["caption_column"],
                    args.hf_sample_size,
                    args.seed,
                )
                print(f"  collected {len(captions):,} captions")
                stats = compute_stats(
                    cfg["name"],
                    captions,
                    args.group_size,
                    args.mtld_threshold,
                    rng,
                )
            except Exception as exc:  # noqa: BLE001 — graceful continue on HF errors
                print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                stats = {"name": cfg["name"], "n_captions": 0, "error": str(exc)}
            all_stats.append(stats)

    summary = {
        "config": {
            "seed": args.seed,
            "group_size": args.group_size,
            "mtld_threshold": args.mtld_threshold,
            "hf_sample_size": args.hf_sample_size if not args.no_hf else None,
        },
        "datasets": all_stats,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_comparison_csv(all_stats, args.output_dir / "comparison.csv")
    write_comparison_markdown(all_stats, args.output_dir / "comparison.md")

    # Our-dataset-only artefacts (length histogram, word freq, word cloud).
    our_tokens: list[str] = []
    for c in ours:
        our_tokens.extend(tokenize(c))
    write_word_freq_csv(our_tokens, args.top_n_words, args.output_dir / "word_freq.csv")
    write_length_histogram(
        [len(tokenize(c)) for c in ours],
        our_stats["caption_length_words"]["mean"],
        args.output_dir / "length_histogram.png",
    )
    write_word_cloud(ours, args.output_dir / "word_cloud.png")

    print("\n=== Comparison ===")
    with open(args.output_dir / "comparison.md") as f:
        print(f.read())
    print(f"Full summary → {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
