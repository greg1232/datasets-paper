#!/usr/bin/env python3
"""Zero-shot temporal grounding proxy (§7.2d).

A no-training reproducibility proxy for the paper's temporal-grounding
training experiment. Instead of fine-tuning a localization model on
untrimmed videos, we test whether our captions carry *temporal structure*
within individual long clips: does the caption's beginning describe the
clip's beginning, and so on?

Protocol:
  1. Filter to long clips (duration >= 10s — the long-clip tier the
     paper's §7.2d was designed around).
  2. For each clip:
       - Extract K evenly-spaced frames (default K=3: begin/middle/end).
       - Split the caption into K sub-spans at sentence boundaries,
         balanced to roughly equal word counts.
       - Compute the K×K CLIP text×image cosine similarity matrix.
  3. Aggregate:
       - diag_mean = mean over i of sim[i, i]
       - offdiag_mean = mean over i!=j of sim[i, j]
       - diag_advantage = diag_mean - offdiag_mean
       - argmax_accuracy = fraction of rows i where argmax_j sim[i, j] == i
       - Shuffle baseline: permute caption-clip pairings, recompute.
         A dataset with strong temporal alignment will show
         diag_advantage >> shuffle_diag_advantage.

Interpretation: a positive diag_advantage and argmax_accuracy > 1/K
indicate the caption carries temporal information that a localization
model could learn to exploit.

Single entry point. Reuses CLIP loader + fetchers from
experiments/visual_grounding/run.py.
"""

import argparse
import csv
import json
import random
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
    probe_duration,
    extract_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_DIR = REPO_ROOT / "data" / "segments"
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "labels"
DEFAULT_METADATA_CSV = REPO_ROOT / "results" / "clip_durations" / "video_metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "temporal_grounding"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "visual_grounding" / "_comparator_cache"

VALID_DATASETS = {"ours", "openvid", "internvid"}


def split_into_balanced_subspans(caption: str, k: int) -> list[str] | None:
    """Split caption into K roughly-equal-word-count sub-spans at sentence
    boundaries. Returns None if the caption has fewer than K words per span."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", caption.strip()) if s.strip()]
    if not sentences:
        return None
    total_words = sum(len(s.split()) for s in sentences)
    if total_words < k * 3:
        return None  # too short to split meaningfully
    target_per_span = total_words / k

    spans: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for s in sentences:
        w = len(s.split())
        if current and current_words + w > target_per_span * 1.4 and len(spans) < k - 1:
            spans.append(current)
            current = [s]
            current_words = w
        else:
            current.append(s)
            current_words += w
    if current:
        spans.append(current)
    # If we ended with fewer than k spans (e.g. one big sentence), fall back
    # to word-level splitting.
    if len(spans) < k:
        words = caption.split()
        chunk = max(1, len(words) // k)
        spans = []
        for i in range(k):
            start = i * chunk
            end = (i + 1) * chunk if i < k - 1 else len(words)
            spans.append([" ".join(words[start:end])])
    return [" ".join(s) for s in spans[:k]]


def extract_temporal_frames(
    pair: dict, durations: dict, k: int, tmp_dir: Path
) -> list[Path]:
    mp4 = pair["mp4_path"]
    dur = durations.get(mp4.name) or probe_duration(mp4)
    if not dur or dur <= 0:
        return []
    timestamps = [dur * (i + 0.5) / k for i in range(k)]
    out_paths: list[Path] = []
    for i, t in enumerate(timestamps):
        op = tmp_dir / f"frame_{i}.jpg"
        if extract_frame(mp4, t, op):
            out_paths.append(op)
    return out_paths if len(out_paths) == k else []


def run_for_dataset(
    name: str,
    pairs: list[dict],
    model,
    preprocess,
    tokenizer,
    device: str,
    k: int,
    min_duration: float,
    batch_size: int,
    extraction_workers: int,
    durations: dict,
    output_dir: Path,
    rng: random.Random,
) -> dict:
    """Run end-to-end. Returns summary dict."""
    import torch
    from PIL import Image

    # Filter to long clips: require duration >= min_duration.
    long_pairs = []
    for p in pairs:
        mp4 = p["mp4_path"]
        d = durations.get(mp4.name) or probe_duration(mp4)
        if d is None or d < min_duration:
            continue
        # Also require caption that can be split into k meaningful spans.
        spans = split_into_balanced_subspans(p["caption"], k)
        if spans is None:
            continue
        p["_spans"] = spans
        p["_duration"] = d
        long_pairs.append(p)

    if not long_pairs:
        return {
            "dataset": name,
            "n_submitted": len(pairs),
            "n_eligible": 0,
            "error": "no eligible long clips with splittable captions",
        }

    # Extract frames.
    with tempfile.TemporaryDirectory(prefix=f"tempgr_{name}_") as tmp:
        tmp_root = Path(tmp)
        per_pair_frames: list[list[Path]] = [[] for _ in long_pairs]

        def work(i: int):
            sub = tmp_root / f"p{i:05d}"
            sub.mkdir()
            return i, extract_temporal_frames(long_pairs[i], durations, k, sub)

        with ThreadPoolExecutor(max_workers=extraction_workers) as pool:
            futures = [pool.submit(work, i) for i in range(len(long_pairs))]
            done = 0
            for fut in as_completed(futures):
                i, frames = fut.result()
                per_pair_frames[i] = frames
                done += 1
                if done % 50 == 0 or done == len(long_pairs):
                    print(f"    extracted {done}/{len(long_pairs)}", flush=True)

        # Drop pairs that failed frame extraction.
        kept = [
            (p, per_pair_frames[i])
            for i, p in enumerate(long_pairs)
            if per_pair_frames[i]
        ]
        if not kept:
            return {
                "dataset": name,
                "n_submitted": len(pairs),
                "n_eligible": len(long_pairs),
                "n_with_frames": 0,
                "error": "frame extraction failed for all clips",
            }

        # Embed frames: shape (N*K, D).
        all_imgs: list = []
        for _, frames in kept:
            for fp in frames:
                all_imgs.append(preprocess(Image.open(fp).convert("RGB")))
        img_batch = torch.stack(all_imgs).to(device)
        image_embeds_chunks = []
        with torch.no_grad():
            for i in range(0, img_batch.shape[0], batch_size):
                emb = model.encode_image(img_batch[i : i + batch_size])
                emb = emb / emb.norm(dim=-1, keepdim=True)
                image_embeds_chunks.append(emb)
        image_embeds = torch.cat(image_embeds_chunks, dim=0)  # (N*K, D)

        # Embed text sub-spans: (N*K, D).
        all_texts: list[str] = []
        for p, _ in kept:
            all_texts.extend(p["_spans"])
        text_embeds_chunks = []
        with torch.no_grad():
            for i in range(0, len(all_texts), batch_size):
                tokens = tokenizer(all_texts[i : i + batch_size]).to(device)
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                text_embeds_chunks.append(emb)
        text_embeds = torch.cat(text_embeds_chunks, dim=0)  # (N*K, D)

    # Compute per-clip K×K similarity and aggregate metrics.
    n = len(kept)
    diag_means: list[float] = []
    offdiag_means: list[float] = []
    argmax_correct_total = 0
    argmax_total = 0
    per_clip: list[dict] = []
    for idx in range(n):
        t = text_embeds[idx * k : (idx + 1) * k]  # (K, D)
        im = image_embeds[idx * k : (idx + 1) * k]  # (K, D)
        sim = (t @ im.T).cpu()  # (K, K) — rows = text spans, cols = frame windows
        diag = sim.diag().mean().item()
        # Off-diagonal: sum - diag-sum, divided by K*(K-1)
        off_sum = sim.sum().item() - sim.diag().sum().item()
        off_mean = off_sum / (k * (k - 1)) if k > 1 else 0.0
        row_argmax = sim.argmax(dim=-1).tolist()
        correct = sum(1 for i, m in enumerate(row_argmax) if m == i)
        diag_means.append(diag)
        offdiag_means.append(off_mean)
        argmax_correct_total += correct
        argmax_total += k
        per_clip.append(
            {
                "video_id": kept[idx][0]["video_id"],
                "duration": kept[idx][0]["_duration"],
                "diag_mean": diag,
                "offdiag_mean": off_mean,
                "diag_advantage": diag - off_mean,
                "argmax_correct": correct,
                "argmax_possible": k,
            }
        )

    # Shuffle baseline: random permutation of clip→caption pairings, recompute
    # diag_advantage. Measures how much diag-advantage comes from actual
    # temporal alignment vs corpus-level similarity.
    perm = list(range(n))
    rng.shuffle(perm)
    shuffle_diag_sum = 0.0
    shuffle_off_sum = 0.0
    for idx in range(n):
        t_idx = perm[idx]
        t = text_embeds[t_idx * k : (t_idx + 1) * k]
        im = image_embeds[idx * k : (idx + 1) * k]
        sim = (t @ im.T).cpu()
        shuffle_diag_sum += sim.diag().mean().item()
        shuffle_off_sum += (sim.sum().item() - sim.diag().sum().item()) / (k * (k - 1) if k > 1 else 1)
    shuffle_diag_mean = shuffle_diag_sum / n
    shuffle_off_mean = shuffle_off_sum / n

    # Per-clip CSV.
    with open(output_dir / f"per_clip_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "duration", "diag_mean", "offdiag_mean", "diag_advantage",
                    "argmax_correct", "argmax_possible"])
        for r in per_clip:
            w.writerow([r["video_id"], f"{r['duration']:.2f}",
                        f"{r['diag_mean']:.4f}", f"{r['offdiag_mean']:.4f}",
                        f"{r['diag_advantage']:.4f}",
                        r["argmax_correct"], r["argmax_possible"]])

    return {
        "dataset": name,
        "n_submitted": len(pairs),
        "n_eligible": len(long_pairs),
        "n_with_frames": len(kept),
        "k": k,
        "min_duration": min_duration,
        "diag_mean": statistics.mean(diag_means),
        "offdiag_mean": statistics.mean(offdiag_means),
        "diag_advantage": statistics.mean(diag_means) - statistics.mean(offdiag_means),
        "argmax_accuracy": argmax_correct_total / argmax_total,
        "argmax_chance": 1.0 / k,
        "shuffle_diag_mean": shuffle_diag_mean,
        "shuffle_offdiag_mean": shuffle_off_mean,
        "shuffle_diag_advantage": shuffle_diag_mean - shuffle_off_mean,
        "net_diag_advantage": (statistics.mean(diag_means) - statistics.mean(offdiag_means))
                              - (shuffle_diag_mean - shuffle_off_mean),
    }


def write_comparison(summaries: list[dict], output_dir: Path) -> None:
    with open(output_dir / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset", "N submitted", "N eligible", "N scored", "K",
            "Diag mean", "Off-diag mean", "Diag advantage",
            "Argmax accuracy", "Argmax chance",
            "Shuffle diag advantage", "Net diag advantage",
        ])
        for s in summaries:
            if "error" in s:
                w.writerow([s["dataset"], s["n_submitted"],
                            s.get("n_eligible", 0), 0, "", "", "", "", "", "",
                            "", ""])
                continue
            w.writerow([
                s["dataset"], s["n_submitted"], s["n_eligible"], s["n_with_frames"], s["k"],
                f"{s['diag_mean']:.4f}", f"{s['offdiag_mean']:.4f}",
                f"{s['diag_advantage']:.4f}",
                f"{s['argmax_accuracy']:.3f}", f"{s['argmax_chance']:.3f}",
                f"{s['shuffle_diag_advantage']:.4f}",
                f"{s['net_diag_advantage']:.4f}",
            ])

    lines = [
        "| Dataset | N scored | K | Diag adv | Argmax acc (chance 1/K) | Shuffle adv | **Net adv** |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        if "error" in s:
            lines.append(f"| {s['dataset']} | 0 | | | | | — ({s['error']}) |")
            continue
        lines.append(
            f"| {s['dataset']} | {s['n_with_frames']} | {s['k']} | "
            f"{s['diag_advantage']:+.4f} | "
            f"{s['argmax_accuracy']:.1%} (chance {s['argmax_chance']:.1%}) | "
            f"{s['shuffle_diag_advantage']:+.4f} | "
            f"**{s['net_diag_advantage']:+.4f}** |"
        )
    with open(output_dir / "comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--datasets", default="ours,openvid")
    parser.add_argument("--sample-size-ours", type=int, default=500)
    parser.add_argument("--sample-size-openvid", type=int, default=200)
    parser.add_argument("--sample-size-internvid", type=int, default=50)
    parser.add_argument("--openvid-shard", default="OpenVidHD/OpenVidHD_part_1.zip")
    parser.add_argument("--k-temporal-bins", type=int, default=3,
                        help="Number of temporal bins (frames + caption sub-spans).")
    parser.add_argument("--min-duration", type=float, default=10.0,
                        help="Only score clips of at least this duration in seconds.")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    durations = load_durations(args.metadata_csv)
    model, preprocess, tokenizer = load_clip(args.model, args.pretrained, device)
    rng = random.Random(args.seed)

    summaries: list[dict] = []
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
        print(f"  {name}: {len(pairs)} pairs (filtering to duration>={args.min_duration}s)", flush=True)
        if not pairs:
            continue
        print(f"\n=== Temporal grounding ({name}) ===", flush=True)
        s = run_for_dataset(
            name, pairs, model, preprocess, tokenizer, device,
            args.k_temporal_bins, args.min_duration, args.batch_size,
            args.extraction_workers, durations, args.output_dir, rng,
        )
        summaries.append(s)
        if "error" not in s:
            print(
                f"  {name}: n={s['n_with_frames']}  diag_adv={s['diag_advantage']:+.4f}  "
                f"argmax_acc={s['argmax_accuracy']:.2%} (chance {s['argmax_chance']:.2%})  "
                f"net_adv={s['net_diag_advantage']:+.4f}",
                flush=True,
            )
        else:
            print(f"  {name}: {s['error']}", flush=True)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "config": {
                    "model": args.model,
                    "pretrained": args.pretrained,
                    "device": device,
                    "k_temporal_bins": args.k_temporal_bins,
                    "min_duration_sec": args.min_duration,
                    "seed": args.seed,
                    "sample_sizes": {
                        "ours": args.sample_size_ours,
                        "openvid": args.sample_size_openvid,
                        "internvid": args.sample_size_internvid,
                    },
                },
                "summaries": summaries,
            },
            f,
            indent=2,
        )
    write_comparison(summaries, args.output_dir)

    print("\n=== Comparison ===")
    with open(args.output_dir / "comparison.md") as f:
        print(f.read())
    print(f"Outputs → {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
