#!/usr/bin/env python3
"""Clip duration distribution — reproduces §5.1 histogram + Table 1's 11.2s avg.

Walks every .mp4 under data/segments/, runs ffprobe to extract duration and
resolution, bins durations into the paper's five buckets (0-2/2-4/4-10/10-20/
>20s), and writes summary statistics, a CSV metadata table, a histogram, and a
pie chart matching the one in figure "clip_durations.png".

Single entry point. Ported from llm-caption/infra/paper/ffprobe_analysis_metadata.py
and aggregate_metadata.py, with stdlib-only ffprobe invocation (no ffmpeg-python
dep) and parallelism via ThreadPoolExecutor.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_DIR = REPO_ROOT / "data" / "segments"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "clip_durations"

# Paper's bins (§5.1, figure clip_durations.png).
BIN_EDGES = [(0, 2), (2, 4), (4, 10), (10, 20), (20, float("inf"))]
BIN_LABELS = ["0-2s", "2-4s", "4-10s", "10-20s", ">20s"]
PAPER_PCTS = [37, 22, 23, 8, 10]  # reference distribution from §5.1
PAPER_MEAN_SEC = 11.2              # Table 1, Len_Clip column

PIE_COLORS = ["#4d7c36", "#6ca056", "#8bc34a", "#aed581", "#c5e1a5"]


def probe(path: Path) -> dict:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        meta = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"filename": path.name, "error": str(exc)}

    fmt = meta.get("format", {})
    streams = meta.get("streams", [])
    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )

    stem = path.stem
    if "-Scene-" in stem:
        video_name, clip_number = stem.rsplit("-Scene-", 1)
    else:
        video_name, clip_number = stem, ""

    record = {
        "filename": path.name,
        "video_name": video_name,
        "clip_number": clip_number,
        "format_duration": _safe_float(fmt.get("duration")),
        "size_bytes": _safe_int(fmt.get("size")),
        "bit_rate": _safe_int(fmt.get("bit_rate")),
    }
    if video_stream is not None:
        record["duration"] = _safe_float(
            video_stream.get("duration") or fmt.get("duration")
        )
        record["width"] = video_stream.get("width")
        record["height"] = video_stream.get("height")
        record["codec_name"] = video_stream.get("codec_name")
        record["r_frame_rate"] = video_stream.get("r_frame_rate")
        record["nb_frames"] = _safe_int(video_stream.get("nb_frames"))
    else:
        record["duration"] = record["format_duration"]
    return record


def _safe_float(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(x):
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def bin_index(duration: float) -> int:
    for i, (lo, hi) in enumerate(BIN_EDGES):
        if lo <= duration < hi:
            return i
    return len(BIN_EDGES) - 1


def write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    columns = [
        "video_name",
        "clip_number",
        "duration",
        "width",
        "height",
        "filename",
        "size_bytes",
        "bit_rate",
        "codec_name",
        "r_frame_rate",
        "nb_frames",
        "format_duration",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def write_histogram(durations: list[float], mean: float, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping histogram", file=sys.stderr)
        return
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=60, range=(0, 60), color="#4d7c36", edgecolor="black")
    plt.axvline(mean, color="red", linestyle="--", label=f"mean = {mean:.2f}s")
    plt.axvline(PAPER_MEAN_SEC, color="black", linestyle=":", label=f"paper = {PAPER_MEAN_SEC}s")
    plt.xlabel("Clip duration (seconds, clipped to 60s)")
    plt.ylabel("Number of clips")
    plt.title(f"Clip duration distribution (n={len(durations):,})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_pie(counts: list[int], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(6, 6))
    plt.pie(
        counts,
        labels=BIN_LABELS,
        autopct="%1.0f%%",
        startangle=90,
        colors=PIE_COLORS,
    )
    plt.title("Clip Durations")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.segments_dir.glob("**/*.mp4"))
    if not files:
        print(f"No .mp4 files in {args.segments_dir}", file=sys.stderr)
        return 1
    print(f"Probing {len(files):,} clips with {args.workers} workers ...")

    records: list[dict] = [None] * len(files)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe, p): i for i, p in enumerate(files)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            records[i] = fut.result()
            done += 1
            if done % 500 == 0 or done == len(files):
                print(f"  {done:,} / {len(files):,}", flush=True)

    write_csv(records, args.output_dir / "video_metadata.csv")

    durations = [r["duration"] for r in records if r.get("duration") is not None]
    errors = [r for r in records if r.get("error")]
    print(f"Collected durations: {len(durations):,}  (errors: {len(errors)})")

    counts = [0] * len(BIN_EDGES)
    for d in durations:
        counts[bin_index(d)] += 1
    total = sum(counts)
    pcts = [100.0 * c / total for c in counts] if total else [0.0] * len(counts)

    widths = [r["width"] for r in records if r.get("width")]
    heights = [r["height"] for r in records if r.get("height")]

    mean_sec = statistics.mean(durations) if durations else 0.0
    summary = {
        "n_clips": len(durations),
        "n_errors": len(errors),
        "mean_seconds": mean_sec,
        "median_seconds": statistics.median(durations) if durations else 0.0,
        "std_seconds": statistics.pstdev(durations) if len(durations) > 1 else 0.0,
        "min_seconds": min(durations) if durations else None,
        "max_seconds": max(durations) if durations else None,
        "paper_mean_seconds": PAPER_MEAN_SEC,
        "delta_vs_paper_mean": mean_sec - PAPER_MEAN_SEC,
        "bins": [
            {
                "label": BIN_LABELS[i],
                "count": counts[i],
                "fraction": pcts[i] / 100.0,
                "percent": pcts[i],
                "paper_percent": PAPER_PCTS[i],
                "delta_vs_paper_percent": pcts[i] - PAPER_PCTS[i],
            }
            for i in range(len(BIN_EDGES))
        ],
        "resolution": {
            "width_mean": statistics.mean(widths) if widths else None,
            "height_mean": statistics.mean(heights) if heights else None,
            "width_median": statistics.median(widths) if widths else None,
            "height_median": statistics.median(heights) if heights else None,
            "n_720p": sum(1 for h in heights if h == 720),
            "n_1080p": sum(1 for h in heights if h == 1080),
        },
    }

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_histogram(durations, mean_sec, args.output_dir / "clip_duration_histogram.png")
    write_pie(counts, args.output_dir / "clip_durations.png")

    print("\n=== Duration bins ===")
    print(f"{'bin':<10}{'count':>10}{'pct':>8}{'paper':>8}{'delta':>8}")
    for i, label in enumerate(BIN_LABELS):
        print(
            f"{label:<10}{counts[i]:>10,}{pcts[i]:>7.1f}%"
            f"{PAPER_PCTS[i]:>7d}%{pcts[i] - PAPER_PCTS[i]:>+7.1f}"
        )
    print(f"\nMean duration: {mean_sec:.2f}s (paper: {PAPER_MEAN_SEC}s, "
          f"Δ {mean_sec - PAPER_MEAN_SEC:+.2f})")
    print(f"Median:        {summary['median_seconds']:.2f}s")
    print(f"Outputs → {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
