#!/usr/bin/env python3
"""Zero-shot text-to-video retrieval + caption length ablation (§7.2a + §7.4).

A lightweight, no-training reproducibility proxy for the paper's downstream
retrieval and caption-length-ablation experiments. Uses the same CLIP ViT-B/32
backbone as §7.1c; no fine-tuning is performed.

Protocol:
  1. For each dataset, sample N clip-caption pairs and compute:
       - per-clip CLIP image embedding (mean-pooled over frames_per_clip)
       - per-caption CLIP text embedding at several length truncations
         (default 132 / 64 / 32 / 17 words, truncated at sentence boundaries)
  2. For each (dataset, length) combination, build the N×N cosine similarity
     matrix between text and image embeddings, rank each caption's candidate
     clips, and report Recall@1 / Recall@5 / Recall@10 / median rank.
  3. Retrieval is evaluated within-dataset (caption i is matched to clip i
     among N candidates). Random-baseline R@1 is 1/N, so R@k figures are
     directly comparable only at matched N.

This captures the same intent as §7.2a (discriminative utility of captions
for text-to-video retrieval) and §7.4 (effect of caption length on retrieval
performance) without requiring fine-tuning of CLIP4Clip or equivalent.
"""

import argparse
import csv
import json
import re
import statistics
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visual_grounding"))
from run import (  # noqa: E402
    fetch_ours,
    fetch_openvid,
    fetch_internvid,
    load_durations,
    load_clip,
    select_device,
    extract_frames_for_pair,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_DIR = REPO_ROOT / "data" / "segments"
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "labels"
DEFAULT_METADATA_CSV = REPO_ROOT / "results" / "clip_durations" / "video_metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "zero_shot_retrieval"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "visual_grounding" / "_comparator_cache"

VALID_DATASETS = {"ours", "openvid", "internvid"}


def truncate_to_words(caption: str, target_words: int) -> str:
    """Truncate to at most `target_words`, at a sentence boundary if possible."""
    sentences = re.split(r"(?<=[.!?])\s+", caption.strip())
    kept: list[str] = []
    word_count = 0
    for s in sentences:
        w = len(s.split())
        if word_count + w <= target_words:
            kept.append(s)
            word_count += w
        else:
            break
    if kept:
        return " ".join(kept)
    # First sentence exceeds target — fall back to word-level truncation.
    return " ".join(caption.split()[:target_words])


def compute_image_embeds(
    pairs: list[dict],
    model,
    preprocess,
    device: str,
    frames_per_clip: int,
    extraction_workers: int,
    batch_size: int,
    durations: dict,
) -> tuple[list[dict], "torch.Tensor"]:
    """Return (kept_pairs, image_embeds) — only pairs with at least one frame
    make it into the output."""
    import torch
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="zsret_") as tmp:
        tmp_root = Path(tmp)
        all_frames: list[tuple[dict, list[Path]]] = [(p, []) for p in pairs]
        with ThreadPoolExecutor(max_workers=extraction_workers) as pool:
            futures = {
                pool.submit(
                    extract_frames_for_pair, p, durations, frames_per_clip, tmp_root
                ): i
                for i, p in enumerate(pairs)
            }
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                all_frames[i] = (pairs[i], fut.result())
                done += 1
                if done % 200 == 0 or done == len(pairs):
                    print(f"    extracted {done}/{len(pairs)}", flush=True)
        kept = [(p, fs) for p, fs in all_frames if fs]
        if not kept:
            return [], None

        flat_images: list[tuple[int, Path]] = []
        for pair_idx, (_, frames) in enumerate(kept):
            for f in frames:
                flat_images.append((pair_idx, f))
        per_pair_sum = [torch.zeros(1, dtype=torch.float32, device=device) for _ in kept]
        per_pair_count = [0] * len(kept)

        with torch.no_grad():
            for i in range(0, len(flat_images), batch_size):
                batch = flat_images[i : i + batch_size]
                imgs, idxs = [], []
                for pair_idx, fp in batch:
                    try:
                        img = Image.open(fp).convert("RGB")
                    except Exception:
                        continue
                    imgs.append(preprocess(img))
                    idxs.append(pair_idx)
                if not imgs:
                    continue
                inp = torch.stack(imgs).to(device)
                emb = model.encode_image(inp)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                for j, pair_idx in enumerate(idxs):
                    if per_pair_count[pair_idx] == 0:
                        per_pair_sum[pair_idx] = emb[j].detach().clone()
                    else:
                        per_pair_sum[pair_idx] = per_pair_sum[pair_idx] + emb[j].detach()
                    per_pair_count[pair_idx] += 1

        image_embeds, valid_pairs = [], []
        for k, c in enumerate(per_pair_count):
            if c == 0:
                continue
            v = per_pair_sum[k] / c
            image_embeds.append(v / v.norm())
            valid_pairs.append(kept[k][0])
        return valid_pairs, torch.stack(image_embeds)


def compute_text_embeds(
    captions: list[str], model, tokenizer, device: str, batch_size: int
) -> "torch.Tensor":
    import torch

    pieces = []
    with torch.no_grad():
        for i in range(0, len(captions), batch_size):
            chunk = captions[i : i + batch_size]
            tokens = tokenizer(chunk).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            pieces.append(emb)
    return torch.cat(pieces, dim=0)


def recall_at_k(ranks: list[int], k: int) -> float:
    return sum(1 for r in ranks if r <= k) / len(ranks) if ranks else 0.0


def compute_retrieval_metrics(sims: "torch.Tensor") -> dict:
    """sims: N×N text-to-image similarity matrix. Correct match for row i is column i.
    Returns R@1/5/10 in both directions + median rank.

    Rank-by-count implementation — avoids argsort edge cases on MPS and
    handles ties by counting items with strictly greater similarity."""
    import torch

    sims_cpu = sims.detach().to("cpu").float()
    n = sims_cpu.shape[0]

    # Text -> Video: for row i, correct match is col i
    t2v_ranks = []
    for i in range(n):
        correct = sims_cpu[i, i].item()
        higher = (sims_cpu[i] > correct).sum().item()
        t2v_ranks.append(higher + 1)

    # Video -> Text: for col j (image j), correct match is row j (text j)
    v2t_ranks = []
    for j in range(n):
        correct = sims_cpu[j, j].item()
        higher = (sims_cpu[:, j] > correct).sum().item()
        v2t_ranks.append(higher + 1)

    return {
        "n": n,
        "t2v_R@1": recall_at_k(t2v_ranks, 1),
        "t2v_R@5": recall_at_k(t2v_ranks, 5),
        "t2v_R@10": recall_at_k(t2v_ranks, 10),
        "t2v_median_rank": statistics.median(t2v_ranks),
        "v2t_R@1": recall_at_k(v2t_ranks, 1),
        "v2t_R@5": recall_at_k(v2t_ranks, 5),
        "v2t_R@10": recall_at_k(v2t_ranks, 10),
        "v2t_median_rank": statistics.median(v2t_ranks),
    }


def write_outputs(results: list[dict], output_dir: Path) -> None:
    # Flatten: one row per (dataset, length)
    rows = []
    for r in results:
        for length, m in r["per_length"].items():
            rows.append(
                {
                    "dataset": r["dataset"],
                    "caption_length_cap": length,
                    "n": m["n"],
                    "t2v_R@1": m["t2v_R@1"],
                    "t2v_R@5": m["t2v_R@5"],
                    "t2v_R@10": m["t2v_R@10"],
                    "t2v_median_rank": m["t2v_median_rank"],
                    "v2t_R@1": m["v2t_R@1"],
                    "v2t_R@5": m["v2t_R@5"],
                    "v2t_R@10": m["v2t_R@10"],
                    "v2t_median_rank": m["v2t_median_rank"],
                    "actual_mean_len": m["actual_mean_len"],
                }
            )
    with open(output_dir / "retrieval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Markdown — one table per dataset
    lines = []
    for r in results:
        lines.append(f"\n## {r['dataset']} (N={r['n_pairs']})\n")
        lines.append(
            "| Cap length (words) | Actual mean | t2v R@1 | t2v R@5 | t2v R@10 | t2v MdR | v2t R@1 | v2t R@5 | v2t R@10 |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for length in sorted(r["per_length"].keys(), reverse=True):
            m = r["per_length"][length]
            lines.append(
                f"| {length} | {m['actual_mean_len']:.1f} | "
                f"{m['t2v_R@1']:.1%} | {m['t2v_R@5']:.1%} | {m['t2v_R@10']:.1%} | "
                f"{int(m['t2v_median_rank'])} | "
                f"{m['v2t_R@1']:.1%} | {m['v2t_R@5']:.1%} | {m['v2t_R@10']:.1%} |"
            )
    with open(output_dir / "comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    # Plot: R@1 vs caption length, one line per dataset
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for r in results:
            lengths = sorted(r["per_length"].keys())
            r1 = [r["per_length"][L]["t2v_R@1"] for L in lengths]
            plt.plot(lengths, r1, marker="o", label=f"{r['dataset']} (n={r['n_pairs']})")
        plt.xlabel("Caption truncation cap (words)")
        plt.ylabel("Text→Video Recall@1")
        plt.title("Zero-shot retrieval vs caption length")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "length_ablation.png", dpi=150)
        plt.close()
    except ImportError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--datasets", default="ours,openvid,internvid")
    parser.add_argument("--sample-size-ours", type=int, default=50)
    parser.add_argument("--sample-size-openvid", type=int, default=50)
    parser.add_argument("--sample-size-internvid", type=int, default=50)
    parser.add_argument("--openvid-shard", default="OpenVidHD/OpenVidHD_part_1.zip")
    parser.add_argument(
        "--length-caps",
        default="132,64,32,17",
        help="Comma-separated max word counts to evaluate",
    )
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--extraction-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in VALID_DATASETS:
            print(f"Unknown dataset: {d}", file=sys.stderr)
            return 2

    length_caps = [int(x) for x in args.length_caps.split(",") if x.strip()]
    length_caps.sort(reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    durations = load_durations(args.metadata_csv)

    model, preprocess, tokenizer = load_clip(args.model, args.pretrained, device)

    results: list[dict] = []

    for name in datasets:
        print(f"\n=== Fetching {name} ===", flush=True)
        if name == "ours":
            pairs = fetch_ours(
                args.sample_size_ours, args.seed, args.labels_dir, args.segments_dir
            )
        elif name == "openvid":
            pairs = fetch_openvid(
                args.sample_size_openvid,
                args.seed,
                args.cache_dir / "openvid",
                args.openvid_shard,
            )
        elif name == "internvid":
            pairs = fetch_internvid(
                args.sample_size_internvid, args.seed, args.cache_dir / "internvid"
            )
        print(f"  {name}: {len(pairs)} pairs", flush=True)
        if not pairs:
            continue

        print(f"\n=== Computing image embeddings ({name}) ===", flush=True)
        kept_pairs, image_embeds = compute_image_embeds(
            pairs, model, preprocess, device,
            args.frames_per_clip, args.extraction_workers,
            args.batch_size, durations,
        )
        n = len(kept_pairs)
        if n == 0:
            continue
        print(f"  {n} pairs with valid image embeds")

        dataset_result = {
            "dataset": name,
            "n_pairs": n,
            "per_length": {},
        }
        for cap in length_caps:
            truncated = [truncate_to_words(p["caption"], cap) for p in kept_pairs]
            actual_mean = statistics.mean(len(t.split()) for t in truncated)
            print(f"\n=== {name} @ {cap} words (actual mean {actual_mean:.1f}) ===", flush=True)
            text_embeds = compute_text_embeds(truncated, model, tokenizer, device, args.batch_size)
            import torch

            sims = text_embeds @ image_embeds.T
            m = compute_retrieval_metrics(sims)
            m["actual_mean_len"] = actual_mean
            dataset_result["per_length"][cap] = m
            print(
                f"  t2v R@1 {m['t2v_R@1']:.2%}  R@5 {m['t2v_R@5']:.2%}  R@10 {m['t2v_R@10']:.2%}  "
                f"MdR {m['t2v_median_rank']}",
                flush=True,
            )
        results.append(dataset_result)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "config": {
                    "model": args.model,
                    "pretrained": args.pretrained,
                    "device": device,
                    "length_caps": length_caps,
                    "seed": args.seed,
                    "sample_sizes": {
                        "ours": args.sample_size_ours,
                        "openvid": args.sample_size_openvid,
                        "internvid": args.sample_size_internvid,
                    },
                },
                "results": results,
            },
            f,
            indent=2,
        )
    write_outputs(results, args.output_dir)

    print("\n=== Comparison ===")
    with open(args.output_dir / "comparison.md") as f:
        print(f.read())
    print(f"Outputs → {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
