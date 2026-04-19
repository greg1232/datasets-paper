#!/usr/bin/env python3
"""Within-clip coherence judge (§7.1d).

For each sampled clip, extract the first and last frame, show both to a
Claude vision model, and ask it to rate whether the two frames plausibly
belong to the same semantic scene on a 3-point scale (3=same, 2=marginal,
1=different). Clips rated 1 span an unintended scene transition — a
segmentation failure mode for our PySceneDetect-based pipeline.

Reports the distribution of ratings and the "fully coherent" fraction
(rating==3). Can be extended to comparator datasets via --datasets.

Single entry point. Requires ANTHROPIC_API_KEY in the environment (or
--api-key).
"""

import argparse
import base64
import csv
import json
import os
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_DIR = REPO_ROOT / "data" / "segments"
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "labels"
DEFAULT_METADATA_CSV = REPO_ROOT / "results" / "clip_durations" / "video_metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "clip_coherence"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "visual_grounding" / "_comparator_cache"

VALID_DATASETS = {"ours", "openvid", "internvid"}

JUDGE_PROMPT = """You will see two frames from a single short video clip — the first frame (beginning of clip) and the last frame (end of clip). Your task is to judge whether both frames plausibly belong to the SAME semantic scene — i.e., whether the clip depicts continuous action in one location/context without an unintended scene cut between them.

Rate on this 3-point scale:
  3 = Same scene. The two frames clearly show the same location, subjects, camera context, or continuous activity. A plausible video segment.
  2 = Marginal / partially same. There is some continuity (same setting, same subjects, or natural camera cut within one scene) but also noticeable change. Ambiguous case.
  1 = Different scenes. The two frames clearly depict different locations, different subjects, or different contexts with no plausible continuity. This indicates that an unintended scene boundary fell inside the clip — a segmentation failure.

Return a single JSON object and nothing else, in this exact schema:
{"rating": 1 | 2 | 3, "reason": "<one short sentence explaining the rating>"}"""


def extract_endpoint_frames(mp4: Path, duration: float, tmp_dir: Path) -> list[Path]:
    """Extract the first frame (t=0) and last frame (t=duration-0.1s)."""
    first = tmp_dir / "first.jpg"
    last = tmp_dir / "last.jpg"
    t_last = max(duration - 0.1, 0.1)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "0", "-i", str(mp4),
         "-frames:v", "1", "-q:v", "3", str(first)],
        capture_output=True, timeout=30,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t_last:.3f}", "-i", str(mp4),
         "-frames:v", "1", "-q:v", "3", str(last)],
        capture_output=True, timeout=30,
    )
    if first.exists() and last.exists() and first.stat().st_size > 0 and last.stat().st_size > 0:
        return [first, last]
    return []


def judge_one(client, model: str, frames: list[Path], max_retries: int = 2) -> dict:
    content: list = []
    for fp in frames:
        try:
            b64 = base64.standard_b64encode(fp.read_bytes()).decode("utf-8")
        except Exception:
            continue
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            }
        )
    if len(content) != 2:
        return {"error": "fewer than 2 frames"}
    content.append({"type": "text", "text": JUDGE_PROMPT})

    for attempt in range(max_retries + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": content}],
            )
            text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end <= start:
                raise ValueError(f"no JSON in response: {text[:200]!r}")
            parsed = json.loads(text[start : end + 1])
            rating = int(parsed.get("rating", 0))
            if rating not in (1, 2, 3):
                raise ValueError(f"rating out of range: {rating}")
            return {"rating": rating, "reason": parsed.get("reason", "")}
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                return {"error": f"{type(exc).__name__}: {exc}"[:200]}
            time.sleep(1.5 * (attempt + 1))
    return {"error": "no verdict"}


def run_judge(
    pairs: list[dict],
    model: str,
    client,
    durations: dict,
    workers: int,
) -> list[dict]:
    results: list[dict] = [{} for _ in pairs]
    with tempfile.TemporaryDirectory(prefix="coh_") as tmp:
        tmp_root = Path(tmp)

        def work(i: int):
            pair = pairs[i]
            mp4 = pair["mp4_path"]
            dur = durations.get(mp4.name) or probe_duration(mp4)
            if not dur or dur <= 0:
                return i, {"error": "no duration"}
            pair_tmp = tmp_root / f"p{i:06d}"
            pair_tmp.mkdir()
            frames = extract_endpoint_frames(mp4, dur, pair_tmp)
            if not frames:
                return i, {"error": "no frames"}
            v = judge_one(client, model, frames)
            v["duration"] = dur
            return i, v

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i) for i in range(len(pairs))]
            done = 0
            for fut in as_completed(futures):
                i, v = fut.result()
                results[i] = v
                done += 1
                if done % 20 == 0 or done == len(pairs):
                    print(f"    judged {done}/{len(pairs)}", flush=True)
    return results


def summarize(name: str, pairs: list[dict], verdicts: list[dict]) -> dict:
    counts = {1: 0, 2: 0, 3: 0}
    n_valid = n_err = 0
    durations: list[float] = []
    for p, v in zip(pairs, verdicts):
        if "error" in v:
            n_err += 1
            continue
        r = v.get("rating")
        if r in counts:
            counts[r] += 1
            n_valid += 1
            if "duration" in v:
                durations.append(v["duration"])
    total = n_valid or 1
    return {
        "dataset": name,
        "n_submitted": len(pairs),
        "n_valid": n_valid,
        "n_errors": n_err,
        "rating_counts": counts,
        "rating_fractions": {k: counts[k] / total for k in counts},
        "fully_coherent_rate": counts[3] / total,
        "failure_rate_rating1": counts[1] / total,
        "mean_duration_judged": sum(durations) / len(durations) if durations else 0.0,
    }


def write_comparison(summaries: list[dict], output_dir: Path) -> None:
    with open(output_dir / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Dataset",
                "N Submitted",
                "N Valid",
                "Rating 3 (fully coherent)",
                "Rating 2 (marginal)",
                "Rating 1 (different scenes)",
                "Fully Coherent Rate",
                "Failure Rate",
                "Mean Duration (s)",
            ]
        )
        for s in summaries:
            c = s["rating_counts"]
            w.writerow(
                [
                    s["dataset"],
                    s["n_submitted"],
                    s["n_valid"],
                    c[3],
                    c[2],
                    c[1],
                    f"{s['fully_coherent_rate']:.3f}",
                    f"{s['failure_rate_rating1']:.3f}",
                    f"{s['mean_duration_judged']:.2f}",
                ]
            )
    lines = [
        "| Dataset | N | Valid | R=3 | R=2 | R=1 | Fully coherent | Failure rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        c = s["rating_counts"]
        lines.append(
            f"| {s['dataset']} | {s['n_submitted']} | {s['n_valid']} | "
            f"{c[3]} | {c[2]} | {c[1]} | "
            f"{s['fully_coherent_rate']:.2%} | "
            f"{s['failure_rate_rating1']:.2%} |"
        )
    with open(output_dir / "comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def write_per_clip(
    name: str, pairs: list[dict], verdicts: list[dict], output_dir: Path
) -> None:
    with open(output_dir / f"per_clip_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "duration", "rating", "reason", "error"])
        for p, v in zip(pairs, verdicts):
            w.writerow(
                [
                    p["video_id"],
                    f"{v.get('duration', 0):.2f}",
                    v.get("rating", ""),
                    (v.get("reason") or "").replace("\n", " ")[:300],
                    v.get("error", ""),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--datasets", default="ours")
    parser.add_argument("--sample-size-ours", type=int, default=100)
    parser.add_argument("--sample-size-openvid", type=int, default=30)
    parser.add_argument("--sample-size-internvid", type=int, default=20)
    parser.add_argument("--openvid-shard", default="OpenVidHD/OpenVidHD_part_1.zip")
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Claude model ID. Haiku 4.5 by default.",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY not set. Either export it or pass --api-key.",
            file=sys.stderr,
        )
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
            summaries.append(summarize(name, [], []))
            continue

        print(f"\n=== Judging {name} (Claude {args.model}) ===", flush=True)
        verdicts = run_judge(pairs, args.model, client, durations, args.workers)
        summaries.append(summarize(name, pairs, verdicts))
        write_per_clip(name, pairs, verdicts, args.output_dir)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "config": {
                    "model": args.model,
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
    print(f"Full summary → {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
