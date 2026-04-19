#!/usr/bin/env python3
"""Zero-shot caption quality via reference-model agreement (§7.2b).

A no-training reproducibility proxy for the paper's Video-LLaVA fine-tuning
experiment. For each sampled clip we extract four evenly-spaced frames,
stitch them into a 2x2 grid, and ask Claude Sonnet to produce an
independent "reference" caption. We then compute BLEU-4, METEOR, and
ROUGE-L between the dataset's caption and Claude's reference.

Intent: high reference-model agreement means the dataset caption
describes the content a strong independent captioning model would name
from the same frames. Scores are reported per dataset for direct
side-by-side comparison.

Single entry point. Reuses fetchers from experiments/visual_grounding/
and the grid-stitching helper from experiments/relabel/.

Requires ANTHROPIC_API_KEY in the environment (or --api-key).
"""

import argparse
import base64
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visual_grounding"))
from run import (  # noqa: E402
    fetch_ours,
    fetch_openvid,
    fetch_internvid,
    load_durations,
    probe_duration,
)

# relabel/run.py has the grid-stitching helpers; import it under a distinct
# name because 'run' already resolves to visual_grounding/run.py above.
import importlib.util as _ilu  # noqa: E402

_relabel_spec = _ilu.spec_from_file_location(
    "_relabel_module",
    str(Path(__file__).resolve().parents[1] / "relabel" / "run.py"),
)
_relabel = _ilu.module_from_spec(_relabel_spec)
_relabel_spec.loader.exec_module(_relabel)
build_grid = _relabel.build_grid
GRID_LAYOUTS = _relabel.GRID_LAYOUTS
describe_grid = _relabel.describe_grid

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_DIR = REPO_ROOT / "data" / "segments"
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "labels"
DEFAULT_METADATA_CSV = REPO_ROOT / "results" / "clip_durations" / "video_metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "caption_quality_claude"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "visual_grounding" / "_comparator_cache"

VALID_DATASETS = {"ours", "openvid", "internvid"}

REF_PROMPT = """You are looking at a single image composed of {n} frames sampled in chronological order from the same short video clip, arranged in a {grid_desc} grid (read left-to-right, top-to-bottom: first frame, then subsequent frames in time order).

Write one paragraph (about 100-150 words) describing what the video shows. Describe the setting and location, the people present, their clothing and actions, any objects and on-screen text, and how the scene changes across frames. Quote any legible on-screen text exactly. Describe only what is visible in the frames; do not speculate about audio or off-screen context. Natural declarative prose, no preamble, no bullet points, no references to the grid layout or frame numbers."""


def extract_grid_for_pair(
    pair: dict, durations: dict, n_frames: int, tmp_dir: Path
) -> Path | None:
    """Extract n_frames for the pair and stitch into a single jpg. Returns
    path or None on failure."""
    mp4 = pair["mp4_path"]
    dur = durations.get(mp4.name) or probe_duration(mp4)
    if not dur or dur <= 0:
        return None
    timestamps = [dur * (i + 0.5) / n_frames for i in range(n_frames)]
    frame_paths: list[Path] = []
    for i, t in enumerate(timestamps):
        out = tmp_dir / f"f{i}.jpg"
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(mp4),
             "-frames:v", "1", "-q:v", "3", str(out)],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            frame_paths.append(out)
    if not frame_paths:
        return None
    grid, (rows, cols) = build_grid(frame_paths)
    grid_path = tmp_dir / "grid.jpg"
    grid.save(grid_path, quality=85)
    return grid_path


def ref_caption_one(
    client, model: str, grid_path: Path, n_frames: int, max_retries: int = 2
) -> str | None:
    """Ask Claude for a reference caption on a pre-stitched grid image."""
    b64 = base64.standard_b64encode(grid_path.read_bytes()).decode("utf-8")
    rows, cols = GRID_LAYOUTS.get(n_frames, (1, n_frames))
    prompt = REF_PROMPT.format(n=n_frames, grid_desc=describe_grid(rows, cols))
    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        },
        {"type": "text", "text": prompt},
    ]
    for attempt in range(max_retries + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": content}],
            )
            return "".join(
                b.text for b in msg.content if hasattr(b, "text")
            ).strip()
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                return None
            time.sleep(1.5 * (attempt + 1))


def tokenize_for_metrics(text: str) -> list[str]:
    """Lightweight tokenization for n-gram metrics — lowercase + punctuation
    strip + whitespace split. Keeps tokenization identical between the
    dataset caption and Claude's reference."""
    import re

    text = re.sub(r"[^\w\s']", " ", text.lower())
    return [t for t in text.split() if t]


def compute_metrics(hypothesis: str, reference: str) -> dict:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import single_meteor_score
    from rouge_score import rouge_scorer

    hyp_tokens = tokenize_for_metrics(hypothesis)
    ref_tokens = tokenize_for_metrics(reference)
    if not hyp_tokens or not ref_tokens:
        return {"bleu_4": 0.0, "meteor": 0.0, "rouge_l_f": 0.0}

    bleu = sentence_bleu(
        [ref_tokens],
        hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )
    meteor = single_meteor_score(ref_tokens, hyp_tokens)
    rl = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(
        " ".join(ref_tokens), " ".join(hyp_tokens)
    )["rougeL"]
    return {
        "bleu_4": bleu,
        "meteor": meteor,
        "rouge_l_f": rl.fmeasure,
    }


def run_for_dataset(
    name: str,
    pairs: list[dict],
    client,
    model: str,
    n_frames: int,
    workers: int,
    durations: dict,
    output_dir: Path,
) -> dict:
    """Run end-to-end for one dataset. Returns summary dict."""
    with tempfile.TemporaryDirectory(prefix=f"capq_{name}_") as tmp:
        tmp_root = Path(tmp)

        # Stage 1: extract grids (parallel).
        print(f"    extracting {n_frames}-frame grids ...", flush=True)
        grid_paths: list[Path | None] = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=min(8, workers * 2)) as pool:
            futures = {}
            for i, p in enumerate(pairs):
                sub = tmp_root / f"p{i:06d}"
                sub.mkdir()
                futures[pool.submit(extract_grid_for_pair, p, durations, n_frames, sub)] = i
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                grid_paths[i] = fut.result()
                done += 1
                if done % 25 == 0 or done == len(pairs):
                    print(f"      {done}/{len(pairs)}", flush=True)

        # Stage 2: Claude Sonnet reference captions (parallel).
        print(f"    reference captions via {model} (workers={workers}) ...", flush=True)
        refs: list[str | None] = [None] * len(pairs)

        def work(i: int):
            if grid_paths[i] is None:
                return i, None
            return i, ref_caption_one(client, model, grid_paths[i], n_frames)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i) for i in range(len(pairs))]
            done = 0
            for fut in as_completed(futures):
                i, caption = fut.result()
                refs[i] = caption
                done += 1
                if done % 10 == 0 or done == len(pairs):
                    print(f"      ref {done}/{len(pairs)}", flush=True)

    # Stage 3: compute metrics in-process.
    per_clip: list[dict] = []
    for pair, ref in zip(pairs, refs):
        if ref is None:
            per_clip.append({"video_id": pair["video_id"], "error": "no reference"})
            continue
        m = compute_metrics(pair["caption"], ref)
        per_clip.append(
            {
                "video_id": pair["video_id"],
                "dataset_caption_len_words": len(pair["caption"].split()),
                "reference_caption_len_words": len(ref.split()),
                "bleu_4": m["bleu_4"],
                "meteor": m["meteor"],
                "rouge_l_f": m["rouge_l_f"],
                "reference_caption": ref[:500],
            }
        )
    valid = [r for r in per_clip if "error" not in r]

    # Write per-clip CSV.
    with open(output_dir / f"per_clip_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "video_id",
                "dataset_caption_len_words",
                "reference_caption_len_words",
                "bleu_4",
                "meteor",
                "rouge_l_f",
                "reference_caption",
                "error",
            ]
        )
        for r in per_clip:
            w.writerow(
                [
                    r["video_id"],
                    r.get("dataset_caption_len_words", ""),
                    r.get("reference_caption_len_words", ""),
                    f"{r['bleu_4']:.4f}" if "bleu_4" in r else "",
                    f"{r['meteor']:.4f}" if "meteor" in r else "",
                    f"{r['rouge_l_f']:.4f}" if "rouge_l_f" in r else "",
                    r.get("reference_caption", ""),
                    r.get("error", ""),
                ]
            )
    return {
        "dataset": name,
        "n_submitted": len(pairs),
        "n_valid": len(valid),
        "n_errors": len(per_clip) - len(valid),
        "bleu_4_mean": statistics.mean(r["bleu_4"] for r in valid) if valid else 0.0,
        "meteor_mean": statistics.mean(r["meteor"] for r in valid) if valid else 0.0,
        "rouge_l_f_mean": statistics.mean(r["rouge_l_f"] for r in valid) if valid else 0.0,
        "dataset_caption_len_mean": (
            statistics.mean(r["dataset_caption_len_words"] for r in valid) if valid else 0.0
        ),
        "reference_caption_len_mean": (
            statistics.mean(r["reference_caption_len_words"] for r in valid) if valid else 0.0
        ),
    }


def write_comparison(summaries: list[dict], output_dir: Path) -> None:
    with open(output_dir / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Dataset",
                "N Submitted",
                "N Valid",
                "BLEU-4 (mean)",
                "METEOR (mean)",
                "ROUGE-L F (mean)",
                "Dataset caption mean len",
                "Reference caption mean len",
            ]
        )
        for s in summaries:
            w.writerow(
                [
                    s["dataset"],
                    s["n_submitted"],
                    s["n_valid"],
                    f"{s['bleu_4_mean']:.4f}",
                    f"{s['meteor_mean']:.4f}",
                    f"{s['rouge_l_f_mean']:.4f}",
                    f"{s['dataset_caption_len_mean']:.1f}",
                    f"{s['reference_caption_len_mean']:.1f}",
                ]
            )
    lines = [
        "| Dataset | N | Valid | BLEU-4 | METEOR | ROUGE-L F | Dataset len | Ref len |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['dataset']} | {s['n_submitted']} | {s['n_valid']} | "
            f"{s['bleu_4_mean']:.4f} | {s['meteor_mean']:.4f} | {s['rouge_l_f_mean']:.4f} | "
            f"{s['dataset_caption_len_mean']:.1f} | {s['reference_caption_len_mean']:.1f} |"
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
    parser.add_argument("--datasets", default="ours,openvid,internvid")
    parser.add_argument("--sample-size-ours", type=int, default=100)
    parser.add_argument("--sample-size-openvid", type=int, default=50)
    parser.add_argument("--sample-size-internvid", type=int, default=20)
    parser.add_argument("--openvid-shard", default="OpenVidHD/OpenVidHD_part_1.zip")
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model for reference captions. Sonnet 4.6 by default for "
        "higher caption fidelity than Haiku.",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in VALID_DATASETS:
            print(f"Unknown dataset: {d}", file=sys.stderr)
            return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    durations = load_durations(args.metadata_csv)

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

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
            continue
        print(f"\n=== Computing reference captions ({name}) ===", flush=True)
        s = run_for_dataset(
            name, pairs, client, args.model, args.frames_per_clip,
            args.workers, durations, args.output_dir,
        )
        summaries.append(s)
        print(
            f"  {name}: BLEU-4 {s['bleu_4_mean']:.3f}  METEOR {s['meteor_mean']:.3f}  "
            f"ROUGE-L {s['rouge_l_f_mean']:.3f}",
            flush=True,
        )

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "config": {
                    "model": args.model,
                    "frames_per_clip": args.frames_per_clip,
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
