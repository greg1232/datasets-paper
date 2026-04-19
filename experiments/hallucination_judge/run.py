#!/usr/bin/env python3
"""Automatic hallucination judge (§7.1a) — Claude-as-judge.

For each sampled clip, extract three representative frames (begin/middle/end),
show them to a Claude vision model alongside the clip's caption, and ask the
judge to enumerate claims in the caption that are not supported by the visual
evidence. Report the fraction of captions containing at least one unsupported
claim, for ours vs InternVid vs OpenVid-1M with the same judge model.

Single entry point. Reuses dataset fetchers from ../visual_grounding/run.py.

Requires ANTHROPIC_API_KEY in the environment (or --api-key flag).
"""

import argparse
import base64
import csv
import json
import os
import random
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse fetchers + frame extractor from the visual_grounding experiment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visual_grounding"))
from run import (  # noqa: E402
    fetch_ours,
    fetch_openvid,
    fetch_internvid,
    extract_frames_for_pair,
    load_durations,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_DIR = REPO_ROOT / "data" / "segments"
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "labels"
DEFAULT_METADATA_CSV = REPO_ROOT / "results" / "clip_durations" / "video_metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "hallucination_judge"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "visual_grounding" / "_comparator_cache"

VALID_DATASETS = {"ours", "openvid", "internvid"}

JUDGE_PROMPT = """You are a factual-contradiction auditor for video captions. I will show you three frames from a short video clip (begin, middle, end) and a caption describing the clip. Your job is to identify ONLY factual CONTENT hallucinations — concrete assertions that directly contradict what is visible, or that name specific entities/objects that are demonstrably absent from all three frames.

ONLY flag a claim if it meets this high bar:
  1. A specific visual attribute that directly contradicts the frames (e.g., "blue shirt" when the shirt is clearly red; "7:30 on the clock" when the clock reads 9:00).
  2. A specific named person, object, text overlay, or concrete action that the caption says is in the video but that is absent from every one of the three frames (e.g., "a baseball card" when no card is visible anywhere).
  3. A structural claim about the scene that is directly contradicted by what is visible (e.g., "four people at a table" when only one person is visible).

DO NOT flag any of the following (these are NOT hallucinations for our purposes):
  - Inferential, epistemic, or hedged language: "suggesting", "suggests", "appears", "appears to be", "likely", "possibly", "seems", "indicating", "indicates", "probably", "implies", "apparent", "evident", "reflecting", "reflects". These are interpretation, not factual content claims.
  - Mood, atmosphere, or style descriptors: "casual", "professional", "relaxed", "elegant", "minimalist", "lively", "cozy", "formal".
  - Purpose or activity inferences drawn from visible props (e.g., "podcast recording" when microphones + headphones are visible; "tutorial" when slides and a presenter are visible; "live stream" when a chat overlay is visible).
  - Claims about audio, dialogue, speech content, or off-screen context — you cannot verify these from frames, so do not flag them.
  - Plausible activity or intent descriptions consistent with the visible setting.
  - Temporal or sequential narration (e.g., "the scene progresses from X to Y") when X and Y are both individually plausible from the sampled frames.
  - Subjective or continuous attributes roughly consistent with the frames ("well-lit", "crowded", "bright", "dimly lit").

When in doubt, DO NOT flag. False positives (flagging interpretation or plausible inference) are worse than false negatives (missing a real factual contradiction).

Respond with a single JSON object and nothing else, in this exact schema:
{"hallucinations": ["<concretely contradicted or absent factual claim>", ...], "has_hallucination": true | false}

If the caption contains no factual content hallucinations as defined above, return {"hallucinations": [], "has_hallucination": false}.

Caption:
\"\"\"
%CAPTION%
\"\"\"
"""


def judge_one(
    client,
    model: str,
    pair: dict,
    frames: list[Path],
    max_retries: int = 2,
) -> dict | None:
    """Submit one clip to Claude and return a parsed JSON verdict."""
    content: list = []
    for frame in frames:
        try:
            b64 = base64.standard_b64encode(frame.read_bytes()).decode("utf-8")
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
    if not content:
        return None
    content.append(
        {
            "type": "text",
            "text": JUDGE_PROMPT.replace("%CAPTION%", pair["caption"]),
        }
    )

    for attempt in range(max_retries + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=600,
                messages=[{"role": "user", "content": content}],
            )
            text = "".join(
                block.text for block in msg.content if hasattr(block, "text")
            ).strip()
            # Parse JSON — tolerate markdown fences / leading prose.
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"no JSON object in response: {text[:200]!r}")
            parsed = json.loads(text[start : end + 1])
            return {
                "has_hallucination": bool(parsed.get("has_hallucination", False)),
                "hallucinations": list(parsed.get("hallucinations") or []),
                "raw": text,
            }
        except Exception as exc:
            if attempt == max_retries:
                return {"error": f"{type(exc).__name__}: {exc}"}
            time.sleep(1.5 * (attempt + 1))
    return None


def run_judge(
    pairs: list[dict],
    model: str,
    client,
    frames_per_clip: int,
    durations: dict,
    workers: int,
) -> list[dict]:
    """Extract frames for every pair, then judge in parallel."""
    results: list[dict | None] = [None] * len(pairs)
    with tempfile.TemporaryDirectory(prefix="halluj_") as tmp:
        tmp_root = Path(tmp)

        # Stage 1: extract frames.
        print(f"    extracting {frames_per_clip} frames per clip ...", flush=True)
        frames_per_pair: list[list[Path]] = [[] for _ in pairs]
        with ThreadPoolExecutor(max_workers=min(8, workers * 2)) as pool:
            futures = {
                pool.submit(
                    extract_frames_for_pair, p, durations, frames_per_clip, tmp_root
                ): i
                for i, p in enumerate(pairs)
            }
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                frames_per_pair[i] = fut.result()
                done += 1
                if done % 50 == 0 or done == len(pairs):
                    print(f"      {done}/{len(pairs)}", flush=True)

        # Stage 2: parallel Claude calls.
        print(f"    dispatching {len(pairs)} Claude calls (workers={workers}) ...",
              flush=True)

        def work(i: int):
            p = pairs[i]
            fs = frames_per_pair[i]
            if not fs:
                return i, {"error": "no frames"}
            v = judge_one(client, model, p, fs)
            return i, v or {"error": "no verdict"}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i) for i in range(len(pairs))]
            done = 0
            for fut in as_completed(futures):
                i, verdict = fut.result()
                results[i] = verdict
                done += 1
                if done % 10 == 0 or done == len(pairs):
                    print(f"      judged {done}/{len(pairs)}", flush=True)
    return results


def summarize(dataset_name: str, pairs: list[dict], verdicts: list[dict]) -> dict:
    n_valid, n_error, n_hallucinated = 0, 0, 0
    total_flagged_claims = 0
    caption_lens: list[int] = []
    for p, v in zip(pairs, verdicts):
        if v is None or "error" in v:
            n_error += 1
            continue
        n_valid += 1
        caption_lens.append(len(p["caption"].split()))
        if v.get("has_hallucination"):
            n_hallucinated += 1
            total_flagged_claims += len(v.get("hallucinations", []))
    rate = (n_hallucinated / n_valid) if n_valid else 0.0
    total_words = sum(caption_lens)
    # Length-normalized hallucination metric — flagged claims per 100 caption
    # words. Much fairer across datasets with very different caption lengths.
    claims_per_100w = (
        100 * total_flagged_claims / total_words if total_words else 0.0
    )
    return {
        "dataset": dataset_name,
        "n_submitted": len(pairs),
        "n_valid": n_valid,
        "n_errors": n_error,
        "n_with_hallucination": n_hallucinated,
        "hallucination_rate": rate,
        "total_flagged_claims": total_flagged_claims,
        "total_caption_words": total_words,
        "flagged_claims_per_100_words": claims_per_100w,
        "avg_claims_per_hallucinated_caption": (
            total_flagged_claims / n_hallucinated if n_hallucinated else 0.0
        ),
        "caption_length_mean": (
            statistics.mean(caption_lens) if caption_lens else 0.0
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
                "N Hallucinated",
                "Hallucination Rate",
                "Avg Claims/Hallucinated",
                "Flagged Claims/100w",
                "Mean Caption Length",
            ]
        )
        for s in summaries:
            w.writerow(
                [
                    s["dataset"],
                    s["n_submitted"],
                    s["n_valid"],
                    s["n_with_hallucination"],
                    f"{s['hallucination_rate']:.3f}",
                    f"{s['avg_claims_per_hallucinated_caption']:.2f}",
                    f"{s['flagged_claims_per_100_words']:.2f}",
                    f"{s['caption_length_mean']:.1f}",
                ]
            )

    lines = [
        "| Dataset | N | Valid | Hallucinated | Caption rate | Claims/HC | "
        "Claims/100w | Mean len |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['dataset']} | {s['n_submitted']} | {s['n_valid']} | "
            f"{s['n_with_hallucination']} | {s['hallucination_rate']:.2%} | "
            f"{s['avg_claims_per_hallucinated_caption']:.2f} | "
            f"{s['flagged_claims_per_100_words']:.2f} | "
            f"{s['caption_length_mean']:.1f} |"
        )
    with open(output_dir / "comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def write_per_clip(name: str, pairs: list[dict], verdicts: list[dict],
                   output_dir: Path) -> None:
    with open(output_dir / f"per_clip_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["video_id", "caption_len_words", "has_hallucination",
             "n_claims", "claims", "error"]
        )
        for p, v in zip(pairs, verdicts):
            if v is None:
                w.writerow([p["video_id"], len(p["caption"].split()), "",
                            "", "", "no verdict"])
                continue
            if "error" in v:
                w.writerow([p["video_id"], len(p["caption"].split()), "",
                            "", "", v["error"]])
                continue
            claims = v.get("hallucinations", [])
            w.writerow(
                [
                    p["video_id"],
                    len(p["caption"].split()),
                    int(v.get("has_hallucination", False)),
                    len(claims),
                    " | ".join(claims),
                    "",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--datasets", default="ours,openvid,internvid")
    parser.add_argument("--sample-size-ours", type=int, default=100)
    parser.add_argument("--sample-size-openvid", type=int, default=30)
    parser.add_argument("--sample-size-internvid", type=int, default=20)
    parser.add_argument("--openvid-shard", default="OpenVidHD/OpenVidHD_part_1.zip")
    parser.add_argument("--frames-per-clip", type=int, default=3)
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Claude model ID. Haiku 4.5 by default for laptop-friendly latency.",
    )
    parser.add_argument("--workers", type=int, default=6,
                        help="Concurrent Claude API calls")
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
    all_pairs: dict[str, list[dict]] = {}
    all_verdicts: dict[str, list[dict]] = {}

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
            all_pairs[name] = []
            all_verdicts[name] = []
            continue

        print(f"\n=== Judging {name} (Claude {args.model}) ===", flush=True)
        verdicts = run_judge(
            pairs,
            args.model,
            client,
            args.frames_per_clip,
            durations,
            args.workers,
        )
        summaries.append(summarize(name, pairs, verdicts))
        all_pairs[name] = pairs
        all_verdicts[name] = verdicts
        write_per_clip(name, pairs, verdicts, args.output_dir)

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
    print(f"Full summary → {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
