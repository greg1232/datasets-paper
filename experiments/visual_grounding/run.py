#!/usr/bin/env python3
"""Visual grounding — CLIP cosine alignment (§7.1c) with 3-way comparison.

Scores clip-caption alignment on the People's Video 8K subset and, optionally,
on size-matched samples from InternVid (yt-dlp clip segments from YouTube) and
OpenVid-1M (partial-read from HuggingFace shards via remotezip). All three use
the same CLIP model for a fair comparison.

Single entry point. Downloaded video samples are cached under
results/visual_grounding/_comparator_cache/ (gitignored).
"""

import argparse
import csv
import json
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
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "_comparator_cache"

VALID_DATASETS = {"ours", "openvid", "internvid"}


# --------------------------------------------------------------------------- #
# Dataset fetchers
# --------------------------------------------------------------------------- #

def fetch_ours(sample_size: int, seed: int, labels_dir: Path, segments_dir: Path) -> list[dict]:
    pairs = []
    for json_path in sorted(labels_dir.glob("*/0.json")):
        with open(json_path) as f:
            rec = json.load(f)
        caption = (rec.get("caption") or "").strip()
        url = rec.get("url") or ""
        if not caption or not url:
            continue
        mp4 = segments_dir / Path(url).name
        if mp4.exists():
            pairs.append(
                {
                    "dataset": "ours",
                    "video_id": rec.get("video_id"),
                    "caption": caption,
                    "mp4_path": mp4,
                }
            )
    if sample_size < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, sample_size)
    return pairs


def fetch_openvid(
    sample_size: int,
    seed: int,
    cache_dir: Path,
    shard: str = "OpenVidHD/OpenVidHD_part_1.zip",
    caption_csv: str = "data/train/OpenVidHD.csv",
) -> list[dict]:
    """Fetch (video, caption) pairs by:
    1. Download OpenVidHD.csv (286 MB, cached by hf_hub_download) for captions.
    2. List files in an OpenVidHD shard via remotezip (no download).
    3. Intersect + sample, then remotezip-fetch N mp4 files.
    """
    import csv as csvlib
    from huggingface_hub import hf_hub_download, hf_hub_url
    from remotezip import RemoteZip

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"  OpenVid-1M: downloading caption CSV {caption_csv} (cached) ...",
          flush=True)
    csv_path = hf_hub_download(
        "nkp37/OpenVid-1M", caption_csv, repo_type="dataset"
    )
    filename_to_caption: dict[str, str] = {}
    with open(csv_path) as f:
        for row in csvlib.DictReader(f):
            fname = row.get("video")
            caption = (row.get("caption") or "").strip()
            if fname and caption:
                filename_to_caption[fname] = caption
    print(f"    {len(filename_to_caption):,} caption entries in CSV", flush=True)

    url = hf_hub_url("nkp37/OpenVid-1M", shard, repo_type="dataset")
    print(f"  reading central directory of {shard} ...", flush=True)
    with RemoteZip(url) as rz:
        shard_files = {
            Path(n).name: n for n in rz.namelist() if n.endswith(".mp4")
        }
    print(f"    shard has {len(shard_files):,} mp4 files", flush=True)

    matched = [
        (fname, filename_to_caption[fname], shard_files[fname])
        for fname in shard_files
        if fname in filename_to_caption
    ]
    print(f"    {len(matched):,} files have captions", flush=True)
    if not matched:
        return []

    rng = random.Random(seed)
    rng.shuffle(matched)
    matched = matched[:sample_size]

    print(f"  downloading {len(matched)} mp4 segments (cached) ...", flush=True)
    out: list[dict] = []
    with RemoteZip(url) as rz:
        for i, (fname, caption, zip_path) in enumerate(matched, 1):
            local = cache_dir / fname
            if not local.exists() or local.stat().st_size == 0:
                try:
                    data = rz.read(zip_path)
                    local.write_bytes(data)
                except Exception as exc:
                    print(f"    [{i}/{len(matched)}] {fname}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    continue
            out.append(
                {
                    "dataset": "openvid",
                    "video_id": fname.rsplit(".", 1)[0],
                    "caption": caption,
                    "mp4_path": local,
                }
            )
            if i % 20 == 0 or i == len(matched):
                print(f"    {i}/{len(matched)}", flush=True)
    return out

    print(f"  downloading mp4 segments (cached under {cache_dir}) ...", flush=True)
    out: list[dict] = []
    with RemoteZip(url) as rz:
        for i, p in enumerate(to_fetch, 1):
            local = cache_dir / p["filename"]
            if not local.exists() or local.stat().st_size == 0:
                try:
                    data = rz.read(p["shard_path"])
                    local.write_bytes(data)
                except Exception as exc:
                    print(f"    [{i}/{len(to_fetch)}] {p['filename']}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    continue
            out.append(
                {
                    "dataset": "openvid",
                    "video_id": p["video_id"],
                    "caption": p["caption"],
                    "mp4_path": local,
                }
            )
            if i % 20 == 0 or i == len(to_fetch):
                print(f"    {i}/{len(to_fetch)}", flush=True)
    return out


def _ytdlp_fetch(yt_id: str, start: str, end: str, cache_dir: Path) -> Path | None:
    key = f"{yt_id}_{start.replace(':', '')}_{end.replace(':', '')}"
    existing = list(cache_dir.glob(f"{key}.*"))
    if existing:
        return existing[0]
    cmd = [
        ".venv/bin/yt-dlp",
        f"--download-sections=*{start}-{end}",
        "--format=worst[ext=mp4]/worst",
        "--force-keyframes-at-cuts",
        "-o",
        str(cache_dir / f"{key}.%(ext)s"),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        f"https://www.youtube.com/watch?v={yt_id}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    hits = list(cache_dir.glob(f"{key}.*"))
    return hits[0] if hits else None


def fetch_internvid(
    sample_size: int, seed: int, cache_dir: Path, workers: int = 4
) -> list[dict]:
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("  streaming InternVid captions + timestamps ...", flush=True)
    ds = load_dataset(
        "OpenGVLab/InternVid", "InternVid-10M", streaming=True, split="FLT"
    )
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    # Buffer candidates so we can dispatch in parallel.
    candidates: list[dict] = []
    buffer_target = sample_size * 3
    for sample in ds:
        yt_id = sample["YoutubeID"]
        caption = (sample.get("Caption") or "").strip()
        start = (sample.get("Start_timestamp") or "")[:8]
        end = (sample.get("End_timestamp") or "")[:8]
        if not yt_id or not caption or not start or not end:
            continue
        candidates.append(
            {"yt_id": yt_id, "caption": caption, "start": start, "end": end}
        )
        if len(candidates) >= buffer_target:
            break

    print(
        f"  buffered {len(candidates)} candidates; fetching up to {sample_size} "
        f"via yt-dlp (workers={workers}) ...",
        flush=True,
    )
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for c in candidates:
            f = pool.submit(_ytdlp_fetch, c["yt_id"], c["start"], c["end"], cache_dir)
            futures[f] = c
        for fut in as_completed(futures):
            if len(out) >= sample_size:
                break
            c = futures[fut]
            mp4 = fut.result()
            if mp4 is None:
                continue
            out.append(
                {
                    "dataset": "internvid",
                    "video_id": c["yt_id"],
                    "caption": c["caption"],
                    "mp4_path": mp4,
                    "start": c["start"],
                    "end": c["end"],
                }
            )
            if len(out) % 5 == 0:
                print(f"    got {len(out)}/{sample_size}", flush=True)
    print(f"    total collected: {len(out)} "
          f"({100*len(out)/max(len(candidates),1):.0f}% yield)", flush=True)
    return out[:sample_size]


# --------------------------------------------------------------------------- #
# Frame extraction + CLIP scoring (shared across datasets)
# --------------------------------------------------------------------------- #

def load_durations(metadata_csv: Path) -> dict:
    if not metadata_csv.exists():
        return {}
    out: dict[str, float] = {}
    with open(metadata_csv) as f:
        for row in csv.DictReader(f):
            try:
                out[row["filename"]] = float(row["duration"])
            except (KeyError, ValueError, TypeError):
                continue
    return out


def probe_duration(path: Path) -> float | None:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def extract_frame(mp4: Path, timestamp: float, out_path: Path) -> bool:
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(mp4),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out_path),
        ],
        capture_output=True,
        timeout=30,
    )
    return r.returncode == 0 and out_path.exists()


def extract_frames_for_pair(
    pair: dict, durations: dict, n_frames: int, tmp_root: Path
) -> list[Path]:
    mp4 = pair["mp4_path"]
    dur = durations.get(mp4.name) or probe_duration(mp4)
    if not dur or dur <= 0:
        return []
    if n_frames == 1:
        timestamps = [dur / 2.0]
    else:
        timestamps = [dur * (i + 0.5) / n_frames for i in range(n_frames)]
    frames: list[Path] = []
    for i, ts in enumerate(timestamps):
        out_path = tmp_root / f"{pair['dataset']}_{pair['video_id']}_{i}.jpg"
        if extract_frame(mp4, ts, out_path):
            frames.append(out_path)
    return frames


def select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_clip(model_name: str, pretrained: str, device: str):
    import open_clip

    mn = model_name
    if pretrained == "openai" and not mn.endswith("-quickgelu"):
        mn = f"{mn}-quickgelu"
    print(f"Loading CLIP: {mn} / {pretrained} on {device} ...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        mn, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(mn)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


def score_pairs(
    pairs: list[dict],
    model,
    preprocess,
    tokenizer,
    device: str,
    batch_size: int,
    frames_per_clip: int,
    extraction_workers: int,
    durations: dict,
) -> list[tuple[dict, float]]:
    """Returns list of (pair, score) for every pair that produced at least one
    frame. Model/preprocess/tokenizer are passed in — loaded once by caller."""
    import torch
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="viz_") as tmp:
        tmp_root = Path(tmp)
        all_frames: list[tuple[dict, list[Path]]] = [(p, []) for p in pairs]
        with ThreadPoolExecutor(max_workers=extraction_workers) as pool:
            futures = {
                pool.submit(
                    extract_frames_for_pair,
                    p,
                    durations,
                    frames_per_clip,
                    tmp_root,
                ): i
                for i, p in enumerate(pairs)
            }
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                all_frames[i] = (pairs[i], fut.result())
                done += 1
                if done % 100 == 0 or done == len(pairs):
                    print(f"    extracted {done:,}/{len(pairs):,}", flush=True)
        kept = [(p, fs) for p, fs in all_frames if fs]
        if not kept:
            return []

        # Image embeddings, mean-pooled per clip.
        flat_images: list[tuple[int, Path]] = []
        for pair_idx, (_, frames) in enumerate(kept):
            for f in frames:
                flat_images.append((pair_idx, f))
        per_pair_sum = [
            torch.zeros(1, dtype=torch.float32, device=device) for _ in kept
        ]
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

        image_embeds, valid = [], []
        for k, c in enumerate(per_pair_count):
            if c == 0:
                continue
            v = per_pair_sum[k] / c
            image_embeds.append(v / v.norm())
            valid.append(k)
        image_embeds = torch.stack(image_embeds)

        texts = [kept[k][0]["caption"] for k in valid]
        text_embeds_list = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tokens = tokenizer(texts[i : i + batch_size]).to(device)
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                text_embeds_list.append(emb)
        text_embeds = torch.cat(text_embeds_list, dim=0)
        scores = (image_embeds * text_embeds).sum(dim=-1).cpu().tolist()
        return [(kept[k][0], s) for k, s in zip(valid, scores)]


# --------------------------------------------------------------------------- #
# Summary / output
# --------------------------------------------------------------------------- #

def summarize(dataset_name: str, scored: list[tuple[dict, float]]) -> dict:
    if not scored:
        return {"dataset": dataset_name, "n_scored": 0, "error": "no scores"}
    scores = [s for _, s in scored]
    return {
        "dataset": dataset_name,
        "n_scored": len(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "p10": sorted(scores)[int(0.1 * len(scores))],
        "p90": sorted(scores)[int(0.9 * len(scores))],
    }


def write_comparison(summaries: list[dict], output_dir: Path) -> None:
    with open(output_dir / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "N", "Mean", "Median", "Std", "Min", "Max", "P10", "P90"])
        for s in summaries:
            if s.get("n_scored", 0) == 0:
                w.writerow([s["dataset"], 0, "", "", "", "", "", "", ""])
                continue
            w.writerow(
                [
                    s["dataset"],
                    s["n_scored"],
                    f"{s['mean']:.4f}",
                    f"{s['median']:.4f}",
                    f"{s['std']:.4f}",
                    f"{s['min']:.4f}",
                    f"{s['max']:.4f}",
                    f"{s['p10']:.4f}",
                    f"{s['p90']:.4f}",
                ]
            )

    lines = [
        "| Dataset | N | Mean | Median | Std | P10 | P90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        if s.get("n_scored", 0) == 0:
            lines.append(f"| {s['dataset']} | 0 | | | | | |")
            continue
        lines.append(
            f"| {s['dataset']} | {s['n_scored']:,} | {s['mean']:.4f} | "
            f"{s['median']:.4f} | {s['std']:.4f} | {s['p10']:.4f} | {s['p90']:.4f} |"
        )
    with open(output_dir / "comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def write_histogram(summaries: list[dict], all_scored: dict, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"ours": "#4d7c36", "openvid": "#2b6cb0", "internvid": "#b45309"}
    plt.figure(figsize=(9, 5))
    for name, scored in all_scored.items():
        if not scored:
            continue
        scores = [s for _, s in scored]
        plt.hist(
            scores,
            bins=40,
            alpha=0.55,
            label=f"{name} (n={len(scores)}, μ={statistics.mean(scores):.3f})",
            color=colors.get(name, "gray"),
            edgecolor="black",
            linewidth=0.3,
        )
    plt.xlabel("CLIP cosine similarity (ViT-B/32)")
    plt.ylabel("Number of clips")
    plt.title("Visual grounding — three-way comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_histogram.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--datasets",
        default="ours",
        help="Comma-separated list of ours,openvid,internvid (default ours)",
    )
    parser.add_argument("--sample-size-ours", type=int, default=300)
    parser.add_argument("--sample-size-openvid", type=int, default=100)
    parser.add_argument("--sample-size-internvid", type=int, default=20)
    parser.add_argument(
        "--openvid-shard",
        default="OpenVidHD/OpenVidHD_part_1.zip",
        help="HF path of the OpenVid-1M zip shard to pull sample videos from.",
    )
    parser.add_argument("--frames-per-clip", type=int, default=1)
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

    model, preprocess, tokenizer = load_clip(args.model, args.pretrained, device)

    summaries: list[dict] = []
    all_scored: dict[str, list] = {}

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
            summaries.append({"dataset": name, "n_scored": 0, "error": "no pairs"})
            all_scored[name] = []
            continue

        print(f"\n=== Scoring {name} ===", flush=True)
        scored = score_pairs(
            pairs,
            model,
            preprocess,
            tokenizer,
            device,
            args.batch_size,
            args.frames_per_clip,
            args.extraction_workers,
            durations,
        )
        summaries.append(summarize(name, scored))
        all_scored[name] = scored

        # Per-dataset per-clip CSV.
        per_clip = args.output_dir / f"per_clip_scores_{name}.csv"
        with open(per_clip, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "caption_length_words", "score"])
            for pair, s in scored:
                w.writerow(
                    [
                        pair["video_id"],
                        len(pair["caption"].split()),
                        f"{s:.6f}",
                    ]
                )

    # Write top-level summary.
    config = {
        "model": args.model,
        "pretrained": args.pretrained,
        "device": device,
        "frames_per_clip": args.frames_per_clip,
        "seed": args.seed,
        "datasets": datasets,
        "sample_sizes": {
            "ours": args.sample_size_ours,
            "openvid": args.sample_size_openvid,
            "internvid": args.sample_size_internvid,
        },
    }
    top = {"config": config, "summaries": summaries}
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(top, f, indent=2)

    write_comparison(summaries, args.output_dir)
    write_histogram(summaries, all_scored, args.output_dir)

    print("\n=== Comparison ===")
    with open(args.output_dir / "comparison.md") as f:
        print(f.read())
    print(f"Full summary → {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
