#!/usr/bin/env python3
"""Per-batch inventory and content-category distribution for The People's Video.

Streams a configurable random sample (default 0.1%) of the dataset's
HuggingFace-hosted batch shards, extracts only the per-video ``.info.json``
from each ``.tar.gz`` (skipping the much larger ``.mp4`` payload), and
aggregates two views:

  1. **Dataset inventory.** Per-batch and aggregate source-video counts.
     The full listing across all batches is the source of the 921K-video
     figure reported in §3.3 of the paper.

  2. **Content category distribution.** A breakdown of videos by
     YouTube-supplied ``categories`` and ``language`` fields, used to
     characterise the topical mix of the corpus.

Outputs (under ``--output-dir``, default ``results/category_distribution/``):

  - ``category_sample.csv``        -- raw sampled metadata
  - ``summary.json``               -- aggregate counts (totals, per-batch,
                                      per-category, per-language)
  - ``category_distribution.png``  -- pie chart of major categories

Single entry point. Reads HuggingFace credentials from ``--token`` or the
``HF_TOKEN`` environment variable; never hard-codes tokens.
"""

import argparse
import json
import logging
import os
import random
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from huggingface_hub import HfApi, hf_hub_url  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "category_distribution"

DEFAULT_DATASETS = [f"temp-data-store/batch_{i:04d}" for i in range(11)]
DEFAULT_SAMPLE_FRACTION = 0.001
DEFAULT_SEED = 42
DEFAULT_MAX_WORKERS = 20

logger = logging.getLogger("category_distribution")


def list_tar_gz_files(repo_id: str, token: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    return [f for f in files if f.endswith(".tar.gz")]


def extract_info_json(repo_id: str, tar_name: str, token: str) -> dict | None:
    """Stream a tar from HF Hub and return its ``.info.json`` without ever
    materialising the embedded mp4 payload on disk."""
    url = hf_hub_url(repo_id, tar_name, repo_type="dataset")
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode="r|") as tar:
            for member in tar:
                if member.name.endswith(".info.json"):
                    f = tar.extractfile(member)
                    if f is not None:
                        return json.loads(f.read())
    return None


def _fetch_one(args: tuple) -> dict | None:
    repo_id, tar_name, token = args
    try:
        info = extract_info_json(repo_id, tar_name, token)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed %s/%s: %s", repo_id, tar_name, exc)
        return None
    if info is None:
        return None
    return {
        "categories": info.get("categories") or [],
        "language": info.get("language") or "Unknown",
        "url": info.get("webpage_url", ""),
    }


def collect_metadata(
    datasets: list[str],
    token: str,
    sample_fraction: float,
    seed: int,
    max_workers: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Sample ``sample_fraction`` of tars from each batch and fetch their
    ``info.json`` payloads in parallel.

    Returns the metadata frame and a per-batch total file-count dict.
    """
    rng = random.Random(seed)
    all_tasks: list[tuple[str, str, str]] = []
    per_batch: dict[str, int] = {}

    for repo_id in datasets:
        print(f"Listing files in {repo_id} ...")
        all_files = list_tar_gz_files(repo_id, token)
        total = len(all_files)
        per_batch[repo_id] = total
        sample_size = max(1, int(total * sample_fraction))
        sampled = rng.sample(all_files, sample_size)
        print(f"  {repo_id}: {total:,} files, sampling {sample_size}")
        all_tasks.extend((repo_id, f, token) for f in sampled)

    print(
        f"\nFetching info.json from {len(all_tasks):,} tars "
        f"({max_workers} workers) ..."
    )
    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, task): task for task in all_tasks}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is not None:
                rows.append(result)
            if done % 50 == 0:
                print(f"    [{done}/{len(all_tasks)}] processed ...")

    total_seen = sum(per_batch.values())
    print(
        f"\nTotal samples: {len(rows):,}  (from {total_seen:,} total videos "
        f"across {len(datasets)} batches)"
    )
    return pd.DataFrame(rows), per_batch


def print_summary(df: pd.DataFrame, per_batch: dict[str, int]) -> None:
    total_seen = sum(per_batch.values())
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total videos in dataset (streamed) : {total_seen:,}")
    print(f"Sampled videos                     : {len(df):,}")

    print(f"\n{'-' * 60}")
    print("PER-BATCH FILE COUNT")
    print(f"{'-' * 60}")
    for batch, count in per_batch.items():
        name = batch.split("/")[-1]
        print(f"  {name:<20s}  {count:>10,d}")
    print(f"  {'TOTAL':<20s}  {total_seen:>10,d}")

    if df.empty:
        print("\nNo metadata to analyse.")
        return

    n_videos = len(df)
    cat_series = df["categories"].explode()
    cat_series = cat_series[cat_series.notna() & (cat_series != "")]
    if cat_series.empty:
        cat_series = pd.Series(["Unknown"])
    cat_counts = cat_series.value_counts()
    cat_pct = cat_counts / n_videos * 100

    print(f"\n{'-' * 60}")
    print("CATEGORY BREAKDOWN  (% of videos -- may sum to >100%)")
    print(f"{'-' * 60}")
    for cat in cat_counts.index:
        print(f"  {cat:<30s}  {cat_counts[cat]:>5d}  ({cat_pct[cat]:5.1f}%)")
    print(f"\nUnique categories: {cat_counts.shape[0]}")

    print(f"\n{'-' * 60}")
    print("LANGUAGE BREAKDOWN")
    print(f"{'-' * 60}")
    lang_counts = df["language"].value_counts()
    lang_pct = df["language"].value_counts(normalize=True) * 100
    for lang in lang_counts.index:
        print(f"  {lang:<30s}  {lang_counts[lang]:>5d}  ({lang_pct[lang]:5.1f}%)")
    print(f"\nUnique languages: {df['language'].nunique()}")


def _group_small(counts: pd.Series, threshold_frac: float = 0.02) -> pd.Series:
    """Merge entries below ``threshold_frac`` of total into an ``Other`` bucket."""
    threshold = threshold_frac * counts.sum()
    major = counts[counts >= threshold].copy()
    other_total = counts[counts < threshold].sum()
    if other_total > 0:
        major["Other"] = other_total
    return major


def _save_pie(counts: pd.Series, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    _, _, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title(title, fontsize=14)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_category_pie(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["categories"].explode().value_counts()
    counts = counts[counts.notna() & (counts.index != "")]
    major = _group_small(counts)
    path = out_dir / "category_distribution.png"
    _save_pie(major, "Video Categories (sampled)", path)
    print(f"Saved category pie chart to {path}")


def write_summary_json(
    df: pd.DataFrame, per_batch: dict[str, int], out_path: Path
) -> None:
    cat_series = df["categories"].explode()
    cat_series = cat_series[cat_series.notna() & (cat_series != "")]
    cat_counts = cat_series.value_counts().to_dict() if not cat_series.empty else {}
    lang_counts = df["language"].value_counts().to_dict() if not df.empty else {}

    summary = {
        "total_videos": sum(per_batch.values()),
        "n_batches": len(per_batch),
        "per_batch_counts": per_batch,
        "n_sampled": len(df),
        "category_counts": cat_counts,
        "language_counts": lang_counts,
        "n_unique_categories": len(cat_counts),
        "n_unique_languages": len(lang_counts),
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_path}")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="HuggingFace dataset repos to enumerate (default: "
        "temp-data-store/batch_0000..0010)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help=f"Fraction of tars per batch to sample (default: {DEFAULT_SAMPLE_FRACTION})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token; falls back to HF_TOKEN environment variable",
    )
    args = parser.parse_args()

    if not args.token:
        print(
            "ERROR: HuggingFace token required. "
            "Pass --token or set the HF_TOKEN environment variable.",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir}")

    df, per_batch = collect_metadata(
        args.datasets,
        token=args.token,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        max_workers=args.workers,
    )

    if df.empty:
        print("No metadata extracted; aborting.", file=sys.stderr)
        return 1

    csv_path = args.output_dir / "category_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved sample CSV to {csv_path}")

    write_summary_json(df, per_batch, args.output_dir / "summary.json")
    print_summary(df, per_batch)

    plot_category_pie(df, args.output_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
