#!/usr/bin/env python3
"""Activity category coverage (§7.3b).

Zero-shot maps each clip to the 200-leaf ActivityNet 1.3 taxonomy via a
shared CLIP ViT-B/32 backbone. For each dataset we report the frequency
distribution of classes, the Gini coefficient of that distribution (a
summary measure of concentration), and the top coverage gaps between
People's Video and each comparator (classes most under-represented in
our corpus vs the comparator, and vice versa).

Single entry point. Reuses fetchers and CLIP loader from
experiments/visual_grounding/run.py. Classes loaded from
activitynet_classes.txt (fetched from the ActivityNet v1.3 taxonomy,
272 nodes → 200 leaves).
"""

import argparse
import csv
import json
import statistics
import subprocess
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "activity_coverage"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "visual_grounding" / "_comparator_cache"
DEFAULT_CLASSES_FILE = Path(__file__).parent / "activitynet_classes.txt"

VALID_DATASETS = {"ours", "openvid", "internvid"}


def load_classes(path: Path) -> list[str]:
    classes = []
    with open(path) as f:
        for line in f:
            name = line.strip()
            if name:
                classes.append(name)
    return classes


def build_class_embeddings(classes, model, tokenizer, device):
    """Text-embed 'a video of {class}' for every class, normalize."""
    import torch

    prompts = [f"a video of {c.lower()}" for c in classes]
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def classify_pairs(
    pairs: list[dict],
    classes: list[str],
    model,
    preprocess,
    tokenizer,
    class_embeds,
    device: str,
    frames_per_clip: int,
    extraction_workers: int,
    batch_size: int,
    durations: dict,
) -> list[tuple[dict, int, float]]:
    """Return list of (pair, class_idx, top1_score). Skips clips that fail
    frame extraction."""
    import torch
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="actcov_") as tmp:
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
                if done % 100 == 0 or done == len(pairs):
                    print(f"    extracted {done}/{len(pairs)}", flush=True)
        kept = [(p, fs) for p, fs in all_frames if fs]
        if not kept:
            return []

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

        image_embeds, valid_idxs = [], []
        for k, c in enumerate(per_pair_count):
            if c == 0:
                continue
            v = per_pair_sum[k] / c
            image_embeds.append(v / v.norm())
            valid_idxs.append(k)
        image_embeds = torch.stack(image_embeds)

        # sims: (N_clips, N_classes)
        sims = image_embeds @ class_embeds.T
        top1 = sims.argmax(dim=-1).cpu().tolist()
        top1_scores = sims.max(dim=-1).values.cpu().tolist()
        return [
            (kept[k][0], top1[j], top1_scores[j])
            for j, k in enumerate(valid_idxs)
        ]


def gini(counts: list[int]) -> float:
    """Gini coefficient for a frequency distribution. 0 = uniform, 1 = all mass on one class."""
    n = len(counts)
    if n == 0:
        return 0.0
    total = sum(counts)
    if total == 0:
        return 0.0
    sorted_counts = sorted(counts)
    cum = 0
    weighted = 0.0
    for i, c in enumerate(sorted_counts, 1):
        cum += c
        weighted += i * c
    # Gini for non-negative series
    return (2 * weighted) / (n * total) - (n + 1) / n


def summarize(
    dataset_name: str,
    classified: list[tuple[dict, int, float]],
    classes: list[str],
) -> dict:
    counts = [0] * len(classes)
    top1_scores = []
    for _, cls_idx, score in classified:
        counts[cls_idx] += 1
        top1_scores.append(score)
    n = sum(counts)
    n_covered = sum(1 for c in counts if c > 0)
    return {
        "dataset": dataset_name,
        "n_classified": n,
        "n_classes_total": len(classes),
        "n_classes_covered": n_covered,
        "coverage_fraction": n_covered / len(classes),
        "gini": gini(counts),
        "mean_top1_score": (
            sum(top1_scores) / len(top1_scores) if top1_scores else 0.0
        ),
        "class_counts": counts,  # length = 200
        "class_fractions": [c / n for c in counts] if n else [0.0] * len(counts),
    }


def write_per_class_csv(
    classes: list[str], summaries: list[dict], output_dir: Path
) -> None:
    """Wide CSV: one row per class, columns = fraction per dataset."""
    ds_names = [s["dataset"] for s in summaries]
    with open(output_dir / "class_fractions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"] + ds_names)
        for i, cls in enumerate(classes):
            row = [cls]
            for s in summaries:
                row.append(f"{s['class_fractions'][i]:.6f}")
            w.writerow(row)


def write_comparison(
    summaries: list[dict], classes: list[str], output_dir: Path
) -> None:
    with open(output_dir / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset",
            "N Classified",
            "Classes Covered (of 200)",
            "Coverage Fraction",
            "Gini",
            "Mean Top-1 Score",
        ])
        for s in summaries:
            w.writerow([
                s["dataset"],
                s["n_classified"],
                s["n_classes_covered"],
                f"{s['coverage_fraction']:.3f}",
                f"{s['gini']:.3f}",
                f"{s['mean_top1_score']:.3f}",
            ])
    lines = [
        "| Dataset | N | Classes covered | Coverage % | Gini | Mean top-1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['dataset']} | {s['n_classified']} | "
            f"{s['n_classes_covered']}/200 | "
            f"{s['coverage_fraction']:.1%} | "
            f"{s['gini']:.3f} | "
            f"{s['mean_top1_score']:.3f} |"
        )
    with open(output_dir / "comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def write_coverage_gaps(
    summaries: list[dict], classes: list[str], output_dir: Path, top_k: int = 15
) -> None:
    """For each pair (ours, comparator), list the top_k classes most
    over-represented in each direction."""
    try:
        ours = next(s for s in summaries if s["dataset"] == "ours")
    except StopIteration:
        return
    out_lines = [
        "# Activity coverage gaps",
        "",
        "For each comparator, the classes most over-represented in the comparator "
        "relative to People's Video, and vice versa. Positive gap = comparator has "
        "more mass on that class than we do; negative = we have more.",
    ]
    for s in summaries:
        if s["dataset"] == ours["dataset"]:
            continue
        gaps = []
        for i, cls in enumerate(classes):
            gap = s["class_fractions"][i] - ours["class_fractions"][i]
            gaps.append((cls, gap, s["class_fractions"][i], ours["class_fractions"][i]))
        gaps.sort(key=lambda x: -x[1])  # descending

        out_lines.append("")
        out_lines.append(f"## {s['dataset']} vs ours")
        out_lines.append("")
        out_lines.append(f"### Classes over-represented in {s['dataset']}")
        out_lines.append("| Class | Comparator fraction | Ours fraction | Gap |")
        out_lines.append("|---|---:|---:|---:|")
        for cls, gap, comp, us in gaps[:top_k]:
            out_lines.append(f"| {cls} | {comp:.3%} | {us:.3%} | +{gap:.3%} |")
        out_lines.append("")
        out_lines.append(f"### Classes over-represented in ours (vs {s['dataset']})")
        out_lines.append("| Class | Comparator fraction | Ours fraction | Gap |")
        out_lines.append("|---|---:|---:|---:|")
        for cls, gap, comp, us in gaps[-top_k:][::-1]:
            out_lines.append(f"| {cls} | {comp:.3%} | {us:.3%} | {gap:.3%} |")
    with open(output_dir / "coverage_gaps.md", "w") as f:
        f.write("\n".join(out_lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--classes-file", type=Path, default=DEFAULT_CLASSES_FILE)
    parser.add_argument("--datasets", default="ours,openvid,internvid")
    parser.add_argument("--sample-size-ours", type=int, default=300)
    parser.add_argument("--sample-size-openvid", type=int, default=100)
    parser.add_argument("--sample-size-internvid", type=int, default=20)
    parser.add_argument("--openvid-shard", default="OpenVidHD/OpenVidHD_part_1.zip")
    parser.add_argument("--frames-per-clip", type=int, default=3)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=32)
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
    classes = load_classes(args.classes_file)
    print(f"Loaded {len(classes)} ActivityNet classes from {args.classes_file}")

    model, preprocess, tokenizer = load_clip(args.model, args.pretrained, device)
    class_embeds = build_class_embeddings(classes, model, tokenizer, device)
    print(f"Built class embeddings: shape={tuple(class_embeds.shape)}")

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
        print(f"  {name}: {len(pairs)} pairs", flush=True)
        if not pairs:
            summaries.append({"dataset": name, "n_classified": 0,
                              "n_classes_total": len(classes),
                              "n_classes_covered": 0,
                              "coverage_fraction": 0.0, "gini": 0.0,
                              "mean_top1_score": 0.0,
                              "class_counts": [0]*len(classes),
                              "class_fractions": [0.0]*len(classes)})
            continue

        print(f"\n=== Classifying {name} ===", flush=True)
        classified = classify_pairs(
            pairs, classes, model, preprocess, tokenizer, class_embeds,
            device, args.frames_per_clip, args.extraction_workers,
            args.batch_size, durations,
        )
        summaries.append(summarize(name, classified, classes))

        # Per-clip CSV
        with open(args.output_dir / f"per_clip_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "top1_class", "top1_score"])
            for pair, cls_idx, score in classified:
                w.writerow([pair["video_id"], classes[cls_idx], f"{score:.4f}"])

    # Save summary (drop the big counts arrays for readability).
    summary_json = {
        "config": {
            "model": args.model,
            "pretrained": args.pretrained,
            "device": device,
            "frames_per_clip": args.frames_per_clip,
            "seed": args.seed,
            "n_classes": len(classes),
            "sample_sizes": {
                "ours": args.sample_size_ours,
                "openvid": args.sample_size_openvid,
                "internvid": args.sample_size_internvid,
            },
        },
        "summaries": [
            {k: v for k, v in s.items() if k not in ("class_counts", "class_fractions")}
            for s in summaries
        ],
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    write_comparison(summaries, classes, args.output_dir)
    write_per_class_csv(classes, summaries, args.output_dir)
    write_coverage_gaps(summaries, classes, args.output_dir)

    print("\n=== Comparison ===")
    with open(args.output_dir / "comparison.md") as f:
        print(f.read())
    print(f"Outputs → {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
