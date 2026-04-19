#!/usr/bin/env python3
"""Zero-shot VideoQA caption-answerability (§7.2c).

A no-training reproducibility proxy for the paper's VideoQA fine-tuning
experiment. Instead of training a model, we test whether captions contain
information that is *verifiable* from the video frames alone:

  1. Generate: show Claude the dataset's caption (TEXT ONLY, no frames)
     and ask it to produce N verifiable factual QA pairs about the clip.
  2. Answer: show Claude the frames (IMAGES ONLY, no caption) and the
     questions, and ask it to answer each question or say "unknown".
  3. Score: accuracy = fraction of questions where the frame-based answer
     matches the caption-based answer (Claude as judge).

High accuracy = captions describe facts that are actually visible in the
video (not hallucinated, not inferred from off-screen context). This
captures the same intent as §7.2c VideoQA (caption-grounded QA) without
requiring fine-tuning.

Single entry point. Requires ANTHROPIC_API_KEY.
"""

import argparse
import base64
import csv
import json
import os
import re
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "videoqa_answerability"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "visual_grounding" / "_comparator_cache"

VALID_DATASETS = {"ours", "openvid", "internvid"}

QGEN_PROMPT = """You will be shown a written description of a short video clip (no frames). Generate exactly {n} concise factual question-answer pairs about the video, where each question asks about a concrete visible property (setting, attire, on-screen text, objects, action) that a viewer could verify by watching the clip.

Constraints:
- Each answer must be a short phrase (1-5 words).
- Each question must be answerable from the video content alone (not inferred from audio, context, or general world knowledge).
- Prefer questions whose answer is visually explicit (colour, text, posture) rather than interpretive.
- Do not ask yes/no questions.
- Avoid questions whose answer is a direct quote from the description unless the description's wording is clearly the video's on-screen text.

Return a single JSON object, and nothing else, in this exact schema:
{{"qa": [{{"q": "<question>", "a": "<short answer>"}}, {{"q": "...", "a": "..."}}, ...]}}

Description:
\"\"\"
%CAPTION%
\"\"\""""


ANSWER_PROMPT = """You will be shown a 2x2 grid of 4 frames sampled in chronological order from a short video clip, followed by a list of questions about that clip. For each question, produce a short-phrase answer (1-5 words) based ONLY on what is visible in the frames, or respond with "unknown" if the frames do not contain enough evidence to answer.

Return a single JSON object with the same schema, and nothing else:
{"answers": ["<answer 1>", "<answer 2>", ...]}

Questions:
%QUESTIONS%"""


JUDGE_PROMPT = """You are judging whether a model's video-derived answer matches a caption-derived reference answer. Both answers are short phrases.

For each question:
  - Rate 1 if the two answers clearly describe the same fact (allow synonyms, paraphrases, and minor specificity differences).
  - Rate 0 if they describe different facts OR if the video-derived answer is "unknown".
  - Rate 0 for an answer that is semantically off even if the phrasing overlaps.

Return a single JSON object:
{"matches": [1|0, 1|0, ...]}

Questions and answers:
%ITEMS%"""


def _parse_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def call_claude_text(client, model: str, prompt: str, max_tokens: int = 500,
                     max_retries: int = 2) -> str | None:
    for attempt in range(max_retries + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            )
            return "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        except Exception:
            if attempt == max_retries:
                return None
            time.sleep(1.5 * (attempt + 1))


def call_claude_with_image(
    client, model: str, image_b64: str, prompt: str, max_tokens: int = 500,
    max_retries: int = 2,
) -> str | None:
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
        {"type": "text", "text": prompt},
    ]
    for attempt in range(max_retries + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            return "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        except Exception:
            if attempt == max_retries:
                return None
            time.sleep(1.5 * (attempt + 1))


def extract_grid_b64(
    pair: dict, durations: dict, n_frames: int, tmp_dir: Path
) -> str | None:
    """Extract n_frames, stitch into a grid, return base64 JPEG."""
    mp4 = pair["mp4_path"]
    dur = durations.get(mp4.name) or probe_duration(mp4)
    if not dur or dur <= 0:
        return None
    frame_paths: list[Path] = []
    for i in range(n_frames):
        t = dur * (i + 0.5) / n_frames
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
    grid, _ = build_grid(frame_paths)
    grid_path = tmp_dir / "grid.jpg"
    grid.save(grid_path, quality=85)
    return base64.standard_b64encode(grid_path.read_bytes()).decode("utf-8")


def process_one(
    client,
    gen_model: str,
    ans_model: str,
    judge_model: str,
    pair: dict,
    grid_b64: str,
    n_questions: int,
) -> dict:
    """Full QA cycle for one clip. Returns dict with qa list + per-question match."""
    # 1. Generate Qs from caption only.
    qgen_text = call_claude_text(
        client, gen_model, QGEN_PROMPT.replace("%CAPTION%", pair["caption"]).replace("{n}", str(n_questions)),
        max_tokens=600,
    )
    if qgen_text is None:
        return {"error": "qgen failed"}
    try:
        qa_pairs = _parse_json(qgen_text).get("qa", [])
    except Exception as exc:
        return {"error": f"qgen parse: {exc}"}
    qa_pairs = [
        x for x in qa_pairs
        if isinstance(x, dict) and x.get("q") and x.get("a")
    ][:n_questions]
    if not qa_pairs:
        return {"error": "no QA pairs generated"}

    # 2. Answer Qs from frames only.
    questions_block = "\n".join(f"{i+1}. {x['q']}" for i, x in enumerate(qa_pairs))
    ans_text = call_claude_with_image(
        client, ans_model, grid_b64,
        ANSWER_PROMPT.replace("%QUESTIONS%", questions_block),
        max_tokens=400,
    )
    if ans_text is None:
        return {"error": "answering failed"}
    try:
        video_answers = _parse_json(ans_text).get("answers", [])
    except Exception as exc:
        return {"error": f"answer parse: {exc}"}
    # Pad / truncate to match Qs length
    while len(video_answers) < len(qa_pairs):
        video_answers.append("unknown")
    video_answers = video_answers[: len(qa_pairs)]

    # 3. Judge match.
    items = "\n".join(
        f"{i+1}. Q: {qa['q']}\n   Caption answer: {qa['a']}\n   Video answer: {video_answers[i]}"
        for i, qa in enumerate(qa_pairs)
    )
    judge_text = call_claude_text(
        client, judge_model, JUDGE_PROMPT.replace("%ITEMS%", items), max_tokens=200,
    )
    if judge_text is None:
        return {"error": "judging failed"}
    try:
        matches = _parse_json(judge_text).get("matches", [])
    except Exception as exc:
        return {"error": f"judge parse: {exc}"}
    while len(matches) < len(qa_pairs):
        matches.append(0)
    matches = [int(bool(m)) for m in matches[: len(qa_pairs)]]
    unknown_count = sum(1 for a in video_answers if a.strip().lower() == "unknown")

    return {
        "n_questions": len(qa_pairs),
        "n_matched": sum(matches),
        "n_unknown": unknown_count,
        "accuracy": sum(matches) / len(qa_pairs),
        "qa": [
            {
                "q": qa["q"],
                "caption_answer": qa["a"],
                "video_answer": video_answers[i],
                "match": matches[i],
            }
            for i, qa in enumerate(qa_pairs)
        ],
    }


def run_for_dataset(
    name: str,
    pairs: list[dict],
    client,
    gen_model: str,
    ans_model: str,
    judge_model: str,
    n_frames: int,
    n_questions: int,
    workers: int,
    durations: dict,
    output_dir: Path,
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"vqa_{name}_") as tmp:
        tmp_root = Path(tmp)

        print(f"    extracting {n_frames}-frame grids ...", flush=True)
        grids: list[str | None] = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=min(8, workers * 2)) as pool:
            futures = {}
            for i, p in enumerate(pairs):
                sub = tmp_root / f"p{i:05d}"
                sub.mkdir()
                futures[pool.submit(extract_grid_b64, p, durations, n_frames, sub)] = i
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                grids[i] = fut.result()
                done += 1
                if done % 25 == 0 or done == len(pairs):
                    print(f"      {done}/{len(pairs)}", flush=True)

        print(f"    QA generation + answering + judging via Claude (workers={workers}) ...", flush=True)
        per_clip: list[dict] = [{} for _ in pairs]

        def work(i: int):
            if grids[i] is None:
                return i, {"error": "no grid"}
            return i, process_one(client, gen_model, ans_model, judge_model,
                                  pairs[i], grids[i], n_questions)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i) for i in range(len(pairs))]
            done = 0
            for fut in as_completed(futures):
                i, r = fut.result()
                per_clip[i] = r
                done += 1
                if done % 10 == 0 or done == len(pairs):
                    print(f"      {done}/{len(pairs)}", flush=True)

    # Aggregate
    valid = [r for r in per_clip if "error" not in r]
    # Per-clip CSV.
    with open(output_dir / f"per_clip_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["video_id", "n_questions", "n_matched", "n_unknown", "accuracy", "error"]
        )
        for p, r in zip(pairs, per_clip):
            w.writerow(
                [
                    p["video_id"],
                    r.get("n_questions", ""),
                    r.get("n_matched", ""),
                    r.get("n_unknown", ""),
                    f"{r['accuracy']:.3f}" if "accuracy" in r else "",
                    r.get("error", ""),
                ]
            )
    # Detailed JSON dump of QA for audit.
    with open(output_dir / f"per_qa_{name}.json", "w") as f:
        json.dump(
            [
                {
                    "video_id": pairs[i]["video_id"],
                    "caption": pairs[i]["caption"],
                    "result": per_clip[i],
                }
                for i in range(len(pairs))
            ],
            f,
            indent=2,
        )

    total_q = sum(r["n_questions"] for r in valid)
    total_match = sum(r["n_matched"] for r in valid)
    total_unknown = sum(r["n_unknown"] for r in valid)
    return {
        "dataset": name,
        "n_submitted": len(pairs),
        "n_valid": len(valid),
        "n_errors": len(per_clip) - len(valid),
        "mean_accuracy_per_clip": (
            statistics.mean(r["accuracy"] for r in valid) if valid else 0.0
        ),
        "total_questions": total_q,
        "total_matched": total_match,
        "total_unknown": total_unknown,
        "overall_accuracy": (total_match / total_q) if total_q else 0.0,
        "unknown_rate": (total_unknown / total_q) if total_q else 0.0,
    }


def write_comparison(summaries: list[dict], output_dir: Path) -> None:
    with open(output_dir / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset", "N submitted", "N valid", "Total Qs", "Overall accuracy",
            "Unknown rate", "Mean per-clip accuracy",
        ])
        for s in summaries:
            w.writerow([
                s["dataset"], s["n_submitted"], s["n_valid"], s["total_questions"],
                f"{s['overall_accuracy']:.3f}", f"{s['unknown_rate']:.3f}",
                f"{s['mean_accuracy_per_clip']:.3f}",
            ])
    lines = [
        "| Dataset | N | Valid | Qs | Overall accuracy | Unknown rate | Per-clip accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['dataset']} | {s['n_submitted']} | {s['n_valid']} | "
            f"{s['total_questions']} | "
            f"{s['overall_accuracy']:.2%} | "
            f"{s['unknown_rate']:.2%} | "
            f"{s['mean_accuracy_per_clip']:.2%} |"
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
    parser.add_argument("--sample-size-ours", type=int, default=40)
    parser.add_argument("--sample-size-openvid", type=int, default=20)
    parser.add_argument("--sample-size-internvid", type=int, default=15)
    parser.add_argument("--openvid-shard", default="OpenVidHD/OpenVidHD_part_1.zip")
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--n-questions", type=int, default=3)
    parser.add_argument("--gen-model", default="claude-haiku-4-5-20251001",
                        help="Claude model for QA generation (text only).")
    parser.add_argument("--ans-model", default="claude-haiku-4-5-20251001",
                        help="Claude model for answering from frames.")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001",
                        help="Claude model for judging match.")
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
        print(f"\n=== VideoQA pipeline ({name}) ===", flush=True)
        s = run_for_dataset(
            name, pairs, client,
            args.gen_model, args.ans_model, args.judge_model,
            args.frames_per_clip, args.n_questions,
            args.workers, durations, args.output_dir,
        )
        summaries.append(s)
        print(
            f"  {name}: overall {s['overall_accuracy']:.2%}  unknown {s['unknown_rate']:.2%}",
            flush=True,
        )

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "config": {
                    "gen_model": args.gen_model,
                    "ans_model": args.ans_model,
                    "judge_model": args.judge_model,
                    "frames_per_clip": args.frames_per_clip,
                    "n_questions": args.n_questions,
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
