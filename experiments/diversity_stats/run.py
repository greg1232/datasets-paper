#!/usr/bin/env python3
"""Demographic + linguistic diversity (§7.3c).

Estimates two distributional properties across the three datasets using a
shared CLIP ViT-B/32 backbone and OpenAI Whisper-tiny:
  1. Apparent gender + age of on-screen subjects (zero-shot CLIP against
     a small bank of demographic prompts).
  2. Fraction of clips with non-English speech, from Whisper's
     language-ID head applied to the first 30s of audio.

Both measurements are reported as DISTRIBUTIONS across a sample, not as
individual-level labels. Geographic diversity (§7.3c third prong) is
deferred because YouTube channel-country metadata is not preserved in
the 8K validation subset.

Single entry point. Reuses fetchers + CLIP loader from
experiments/visual_grounding/run.py.
"""

import argparse
import csv
import json
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "diversity_stats"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "visual_grounding" / "_comparator_cache"

VALID_DATASETS = {"ours", "openvid", "internvid"}

GENDER_PROMPTS = [
    ("no_person", "a photo with no person visible"),
    ("man", "a photo of a man"),
    ("woman", "a photo of a woman"),
]

AGE_PROMPTS = [
    ("no_person", "a photo with no person visible"),
    ("child", "a photo of a child or teenager under 20"),
    ("young_adult", "a photo of a young adult in their 20s or 30s"),
    ("middle_aged", "a photo of a middle-aged adult in their 40s or 50s"),
    ("elderly", "a photo of an elderly person over 60"),
]


def extract_audio(mp4: Path, out_wav: Path, seconds: int = 30) -> bool:
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4),
            "-t",
            str(seconds),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_wav),
        ],
        capture_output=True,
        timeout=30,
    )
    return r.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 1000


def build_prompt_embeddings(prompts, model, tokenizer, device):
    import torch

    texts = [t for _, t in prompts]
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def classify_frames(
    pairs: list[dict],
    model,
    preprocess,
    gender_embeds,
    age_embeds,
    device: str,
    frames_per_clip: int,
    extraction_workers: int,
    batch_size: int,
    durations: dict,
) -> list[tuple[dict, int, int, float, float]]:
    """For each clip, return (pair, gender_idx, age_idx, gender_margin, age_margin).
    Margin = top1_score - top2_score (measure of classifier confidence)."""
    import torch
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="div_") as tmp:
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

        g_sims = image_embeds @ gender_embeds.T
        a_sims = image_embeds @ age_embeds.T
        # Top-1 + margin
        g_sorted, g_order = g_sims.sort(dim=-1, descending=True)
        a_sorted, a_order = a_sims.sort(dim=-1, descending=True)
        g_top = g_order[:, 0].cpu().tolist()
        a_top = a_order[:, 0].cpu().tolist()
        g_margin = (g_sorted[:, 0] - g_sorted[:, 1]).cpu().tolist()
        a_margin = (a_sorted[:, 0] - a_sorted[:, 1]).cpu().tolist()
        return [
            (kept[k][0], g_top[j], a_top[j], g_margin[j], a_margin[j])
            for j, k in enumerate(valid_idxs)
        ]


def detect_language_for_pair(
    whisper_model, mp4: Path, tmp_dir: Path
) -> tuple[str, float]:
    """Return (lang_code, probability). ('no_audio', 0.0) if audio missing."""
    import whisper

    audio_path = tmp_dir / f"{mp4.stem}.wav"
    if not extract_audio(mp4, audio_path, seconds=30):
        return ("no_audio", 0.0)
    try:
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels)
        mel = mel.to(whisper_model.device)
        _, probs = whisper_model.detect_language(mel)
        lang = max(probs, key=probs.get)
        return (lang, float(probs[lang]))
    except Exception:
        return ("error", 0.0)
    finally:
        try:
            audio_path.unlink()
        except Exception:
            pass


def run_language_id(
    pairs: list[dict], whisper_model, workers: int
) -> list[tuple[dict, str, float]]:
    """Language-ID every clip. Returns list of (pair, lang, prob). Single-
    threaded Whisper model, but parallelise audio extraction via workers."""
    results: list[tuple[dict, str, float]] = []
    with tempfile.TemporaryDirectory(prefix="lang_") as tmp:
        tmp_root = Path(tmp)
        done = 0
        for pair in pairs:
            lang, prob = detect_language_for_pair(whisper_model, pair["mp4_path"], tmp_root)
            results.append((pair, lang, prob))
            done += 1
            if done % 20 == 0 or done == len(pairs):
                print(f"    lang-ID {done}/{len(pairs)}", flush=True)
    return results


def summarize(
    name: str,
    classified: list[tuple[dict, int, int, float, float]],
    langs: list[tuple[dict, str, float]],
) -> dict:
    gender_counts = {label: 0 for label, _ in GENDER_PROMPTS}
    age_counts = {label: 0 for label, _ in AGE_PROMPTS}
    gender_margins: list[float] = []
    age_margins: list[float] = []
    for _, g, a, gm, am in classified:
        gender_counts[GENDER_PROMPTS[g][0]] += 1
        age_counts[AGE_PROMPTS[a][0]] += 1
        gender_margins.append(gm)
        age_margins.append(am)
    n_gender = sum(gender_counts.values()) or 1
    n_age = sum(age_counts.values()) or 1

    lang_counts: dict[str, int] = {}
    n_no_audio = 0
    english_count = 0
    for _, lang, _ in langs:
        if lang == "no_audio":
            n_no_audio += 1
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        if lang == "en":
            english_count += 1
    n_lang = sum(lang_counts.values()) or 1
    n_with_audio = n_lang - n_no_audio
    non_english_with_audio = n_with_audio - english_count
    return {
        "dataset": name,
        "n_clips": len(classified),
        "gender_counts": gender_counts,
        "gender_fractions": {k: v / n_gender for k, v in gender_counts.items()},
        "age_counts": age_counts,
        "age_fractions": {k: v / n_age for k, v in age_counts.items()},
        "gender_mean_margin": (
            sum(gender_margins) / len(gender_margins) if gender_margins else 0.0
        ),
        "age_mean_margin": (
            sum(age_margins) / len(age_margins) if age_margins else 0.0
        ),
        "language_counts": lang_counts,
        "n_with_audio": n_with_audio,
        "n_no_audio": n_no_audio,
        "english_fraction_of_with_audio": (
            english_count / n_with_audio if n_with_audio else 0.0
        ),
        "non_english_fraction_of_with_audio": (
            non_english_with_audio / n_with_audio if n_with_audio else 0.0
        ),
    }


def write_comparison(summaries: list[dict], output_dir: Path) -> None:
    # Comparison markdown
    gender_keys = [k for k, _ in GENDER_PROMPTS]
    age_keys = [k for k, _ in AGE_PROMPTS]
    with open(output_dir / "comparison.md", "w") as f:
        f.write("## Apparent gender (zero-shot CLIP)\n\n")
        header = "| Dataset | N | " + " | ".join(gender_keys) + " | Mean margin |\n"
        sep = "|---|---:|" + "".join("---:|" for _ in gender_keys) + "---:|\n"
        f.write(header + sep)
        for s in summaries:
            cells = [
                s["dataset"],
                str(s["n_clips"]),
                *[f"{s['gender_fractions'][k]:.1%}" for k in gender_keys],
                f"{s['gender_mean_margin']:.3f}",
            ]
            f.write("| " + " | ".join(cells) + " |\n")

        f.write("\n## Apparent age (zero-shot CLIP)\n\n")
        header = "| Dataset | N | " + " | ".join(age_keys) + " | Mean margin |\n"
        sep = "|---|---:|" + "".join("---:|" for _ in age_keys) + "---:|\n"
        f.write(header + sep)
        for s in summaries:
            cells = [
                s["dataset"],
                str(s["n_clips"]),
                *[f"{s['age_fractions'][k]:.1%}" for k in age_keys],
                f"{s['age_mean_margin']:.3f}",
            ]
            f.write("| " + " | ".join(cells) + " |\n")

        f.write("\n## Language distribution (Whisper-tiny language ID)\n\n")
        f.write("| Dataset | N with audio | N no-audio | % English | % non-English | Top other langs |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for s in summaries:
            lc = s["language_counts"]
            # top 3 non-en non-no_audio
            others = sorted(
                ((k, v) for k, v in lc.items() if k not in ("en", "no_audio")),
                key=lambda x: -x[1],
            )[:3]
            others_str = ", ".join(f"{k}({v})" for k, v in others) if others else "—"
            cells = [
                s["dataset"],
                str(s["n_with_audio"]),
                str(s["n_no_audio"]),
                f"{s['english_fraction_of_with_audio']:.1%}",
                f"{s['non_english_fraction_of_with_audio']:.1%}",
                others_str,
            ]
            f.write("| " + " | ".join(cells) + " |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--datasets", default="ours,openvid,internvid")
    parser.add_argument("--sample-size-ours", type=int, default=150)
    parser.add_argument("--sample-size-openvid", type=int, default=60)
    parser.add_argument("--sample-size-internvid", type=int, default=20)
    parser.add_argument("--openvid-shard", default="OpenVidHD/OpenVidHD_part_1.zip")
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--whisper-model", default="tiny")
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
    gender_embeds = build_prompt_embeddings(GENDER_PROMPTS, model, tokenizer, device)
    age_embeds = build_prompt_embeddings(AGE_PROMPTS, model, tokenizer, device)

    import whisper

    print(f"Loading whisper {args.whisper_model} ...", flush=True)
    # Whisper on MPS has issues with some ops; CPU is the safe default.
    whisper_device = "cpu" if device == "mps" else device
    whisper_model = whisper.load_model(args.whisper_model, device=whisper_device)

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
            summaries.append({"dataset": name, "n_clips": 0, "error": "no pairs"})
            continue

        print(f"\n=== CLIP demographic classification ({name}) ===", flush=True)
        classified = classify_frames(
            pairs, model, preprocess, gender_embeds, age_embeds, device,
            args.frames_per_clip, args.extraction_workers, args.batch_size, durations,
        )
        print(f"\n=== Whisper language-ID ({name}) ===", flush=True)
        langs = run_language_id(pairs, whisper_model, workers=args.extraction_workers)
        summaries.append(summarize(name, classified, langs))

        # Per-clip CSV
        with open(args.output_dir / f"per_clip_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "gender", "age", "gender_margin", "age_margin", "language", "lang_prob"])
            by_vid_lang: dict[str, tuple[str, float]] = {
                pair["video_id"]: (lang, prob) for pair, lang, prob in langs
            }
            for pair, g, a, gm, am in classified:
                lang, prob = by_vid_lang.get(pair["video_id"], ("", 0.0))
                w.writerow(
                    [
                        pair["video_id"],
                        GENDER_PROMPTS[g][0],
                        AGE_PROMPTS[a][0],
                        f"{gm:.4f}",
                        f"{am:.4f}",
                        lang,
                        f"{prob:.4f}",
                    ]
                )

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "config": {
                    "clip_model": args.model,
                    "clip_pretrained": args.pretrained,
                    "whisper_model": args.whisper_model,
                    "device": device,
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
