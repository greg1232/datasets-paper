#!/usr/bin/env python3
"""Visual grounding — CLIP cosine alignment between clip frames and captions (§7.1c).

For a random subsample of clip-caption pairs:
  1. Extract N frames per clip via ffmpeg (default: 1 middle frame).
  2. Embed frames with CLIP, mean-pool across frames per clip.
  3. Embed caption with CLIP text encoder.
  4. Report cosine similarity between the pooled image embedding and the text
     embedding.

Paper specifies CLIP ViT-L/14; for M2 laptop compute we default to ViT-B/32
and a 1,000-clip subsample. Flip to larger model / more clips via flags if
you have budget.

Single entry point. Uses open_clip and MPS when available.
"""

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_DIR = REPO_ROOT / "data" / "segments"
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "labels"
DEFAULT_METADATA_CSV = REPO_ROOT / "results" / "clip_durations" / "video_metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "visual_grounding"


def load_clip_pairs(labels_dir: Path, segments_dir: Path) -> list[dict]:
    """Return list of {video_id, caption, mp4_path} for every label with an
    existing mp4 under segments_dir."""
    pairs = []
    for json_path in sorted(labels_dir.glob("*/0.json")):
        with open(json_path) as f:
            record = json.load(f)
        caption = (record.get("caption") or "").strip()
        url = record.get("url") or ""
        if not caption or not url:
            continue
        filename = Path(url).name
        mp4_path = segments_dir / filename
        if mp4_path.exists():
            pairs.append(
                {
                    "video_id": record.get("video_id"),
                    "caption": caption,
                    "mp4_path": mp4_path,
                }
            )
    return pairs


def load_durations(metadata_csv: Path) -> dict:
    """Map filename → duration (seconds). Returns {} if CSV is missing."""
    if not metadata_csv.exists():
        return {}
    out: dict[str, float] = {}
    with open(metadata_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[row["filename"]] = float(row["duration"])
            except (KeyError, ValueError, TypeError):
                continue
    return out


def extract_frame(mp4_path: Path, timestamp: float, out_path: Path) -> bool:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(mp4_path),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out_path),
        ],
        capture_output=True,
        timeout=30,
    )
    return result.returncode == 0 and out_path.exists()


def extract_frames_for_pair(
    pair: dict,
    durations: dict,
    n_frames: int,
    tmp_root: Path,
) -> list[Path]:
    """Extract up to n_frames evenly spaced frames for a pair. Returns list of
    image paths written."""
    mp4 = pair["mp4_path"]
    duration = durations.get(mp4.name)
    if duration is None or duration <= 0:
        # Cheap fallback — ffprobe the duration now.
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(mp4),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        try:
            duration = float(probe.stdout.strip())
        except ValueError:
            return []
    if n_frames == 1:
        timestamps = [duration / 2.0]
    else:
        timestamps = [duration * (i + 0.5) / n_frames for i in range(n_frames)]

    frames: list[Path] = []
    for i, ts in enumerate(timestamps):
        out_path = tmp_root / f"{pair['video_id']}_{i}.jpg"
        if extract_frame(mp4, ts, out_path):
            frames.append(out_path)
    return frames


def select_device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        help="open_clip model name. Default ViT-B-32 for M2 laptop speed; "
        "paper specifies ViT-L-14 which is ~3x slower.",
    )
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
    )
    parser.add_argument("--extraction-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading clip-caption pairs from {args.labels_dir} ∩ {args.segments_dir} ...")
    pairs = load_clip_pairs(args.labels_dir, args.segments_dir)
    if not pairs:
        print("No matched clip-caption pairs found.", file=sys.stderr)
        return 1
    print(f"Matched {len(pairs):,} pairs with existing mp4 files")

    if args.sample_size < len(pairs):
        pairs = rng.sample(pairs, args.sample_size)
        print(f"Subsampled to {len(pairs):,} pairs (seed={args.seed})")

    durations = load_durations(args.metadata_csv)
    if durations:
        print(f"Loaded {len(durations):,} durations from {args.metadata_csv}")

    # Frame extraction — parallelize ffmpeg calls.
    import torch
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="viz_grounding_") as tmp:
        tmp_root = Path(tmp)
        print(f"Extracting up to {args.frames_per_clip} frames per clip ...")
        all_frames: list[tuple[dict, list[Path]]] = [(p, []) for p in pairs]
        with ThreadPoolExecutor(max_workers=args.extraction_workers) as pool:
            futures = {
                pool.submit(
                    extract_frames_for_pair, p, durations, args.frames_per_clip, tmp_root
                ): i
                for i, p in enumerate(pairs)
            }
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                all_frames[i] = (pairs[i], fut.result())
                done += 1
                if done % 100 == 0 or done == len(pairs):
                    print(f"  extracted {done:,} / {len(pairs):,}", flush=True)

        kept = [(p, frames) for p, frames in all_frames if frames]
        print(f"  usable pairs after extraction: {len(kept):,} / {len(pairs):,}")
        if not kept:
            print("No frames extracted.", file=sys.stderr)
            return 1

        # Load CLIP model.
        import open_clip

        device = select_device(args.device)
        # Original CLIP (pretrained=openai) uses QuickGELU; open_clip's default
        # is standard GELU, which produces a warning and slightly wrong
        # activations. Append -quickgelu to the model name to match.
        model_name = args.model
        if args.pretrained == "openai" and not model_name.endswith("-quickgelu"):
            model_name = f"{model_name}-quickgelu"
        print(f"Loading {model_name} / {args.pretrained} on {device} ...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=args.pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()

        # Encode images in batches; mean-pool per clip.
        print("Encoding image frames ...")
        clip_embeddings: list[torch.Tensor] = []
        flat_images: list[tuple[int, Path]] = []
        for pair_idx, (_, frames) in enumerate(kept):
            for f in frames:
                flat_images.append((pair_idx, f))

        per_pair_embed_sum = [
            torch.zeros(1, dtype=torch.float32, device=device) for _ in kept
        ]
        per_pair_embed_count = [0] * len(kept)

        with torch.no_grad():
            for i in range(0, len(flat_images), args.batch_size):
                batch = flat_images[i : i + args.batch_size]
                imgs = []
                idxs = []
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
                    if per_pair_embed_count[pair_idx] == 0:
                        per_pair_embed_sum[pair_idx] = emb[j].detach().clone()
                    else:
                        per_pair_embed_sum[pair_idx] = (
                            per_pair_embed_sum[pair_idx] + emb[j].detach()
                        )
                    per_pair_embed_count[pair_idx] += 1
                if (i // args.batch_size) % 10 == 0:
                    print(
                        f"  images {i + len(imgs):,} / {len(flat_images):,}",
                        flush=True,
                    )

        # Mean-pool and renormalise per clip.
        image_embeds = []
        valid_pair_indices = []
        for k, count in enumerate(per_pair_embed_count):
            if count == 0:
                continue
            vec = per_pair_embed_sum[k] / count
            vec = vec / vec.norm()
            image_embeds.append(vec)
            valid_pair_indices.append(k)
        image_embeds = torch.stack(image_embeds)

        # Encode captions in batches.
        print("Encoding captions ...")
        texts = [kept[k][0]["caption"] for k in valid_pair_indices]
        text_embeds_list = []
        with torch.no_grad():
            for i in range(0, len(texts), args.batch_size):
                batch = texts[i : i + args.batch_size]
                tokens = tokenizer(batch).to(device)
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                text_embeds_list.append(emb)
        text_embeds = torch.cat(text_embeds_list, dim=0)

        # Cosine similarity per clip.
        scores = (image_embeds * text_embeds).sum(dim=-1).cpu().tolist()

    # Per-clip CSV.
    with open(args.output_dir / "per_clip_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "filename", "caption_length_words", "score"])
        for k_idx, score in zip(valid_pair_indices, scores):
            pair, _ = kept[k_idx]
            writer.writerow(
                [
                    pair["video_id"],
                    pair["mp4_path"].name,
                    len(pair["caption"].split()),
                    f"{score:.6f}",
                ]
            )

    summary = {
        "config": {
            "model": args.model,
            "pretrained": args.pretrained,
            "device": select_device(args.device),
            "sample_size": len(pairs),
            "frames_per_clip": args.frames_per_clip,
            "seed": args.seed,
        },
        "n_pairs_requested": len(pairs),
        "n_pairs_scored": len(scores),
        "cosine_similarity": {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "p10": sorted(scores)[int(0.1 * len(scores))] if scores else None,
            "p90": sorted(scores)[int(0.9 * len(scores))] if scores else None,
        },
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Histogram.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=40, color="#4d7c36", edgecolor="black")
        plt.axvline(
            statistics.mean(scores),
            color="red",
            linestyle="--",
            label=f"mean = {statistics.mean(scores):.3f}",
        )
        plt.xlabel("CLIP cosine similarity")
        plt.ylabel("Number of clips")
        plt.title(
            f"Visual grounding ({args.model}/{args.pretrained}, n={len(scores):,})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "alignment_histogram.png", dpi=150)
        plt.close()
    except ImportError:
        pass

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs → {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
