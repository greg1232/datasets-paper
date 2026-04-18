#!/usr/bin/env python3
"""Re-caption every mp4 under data/segments/ via an OpenAI-compatible VLM endpoint.

Rebuilds the broken caption↔video mapping in `data/labels/` (and the
partially-regenerated `data/fixed-labels/`) by running Qwen-VL locally-or-
remotely against each mp4 directly, with frames sampled across the *whole*
clip rather than only the first two seconds.

Differences from the original `data/labels/` pipeline:
  - **Frame sampling**: N frames evenly spaced over the full clip duration
    (default 4 frames). The original pipeline used only the first two frames
    (frame_number=0, interval=1s, batch_size=2), which made most captions
    describe only the opening 0-2 s of each clip — often an intro card,
    graphic, or B-roll unrelated to the clip's actual content.
  - **Deterministic pairing**: each caption is written to
    `<output_dir>/<md5(url)>/0.json` keyed by the very mp4 we fed to the VLM,
    so there is no opportunity for a batched-executor index shuffle.
  - **Resumable**: an existing `0.json` for a given video_id is skipped, so a
    killed run can be restarted without re-billing already-captioned clips.

Output schema matches `data/labels/0.json` with additional provenance fields:
    {
      "video_id": "<md5(url)>",
      "url": "./data/segments/<name>.mp4",
      "n_frames": <int>,
      "frame_timestamps": [<float>, ...],
      "model": "<model_id>",
      "prompt_version": "v2-relabel",
      "caption": "<paragraph>"
    }

Single entry point.
"""

import argparse
import base64
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_DIR = REPO_ROOT / "data" / "segments"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "labels-v2"

DEFAULT_ENDPOINT = "https://qwenbw.scalarllm.com/v1"
# The endpoint's hostname implies Qwen but it actually serves
# google/gemma-4-31B-it (as returned by its /v1/models listing).
DEFAULT_MODEL = "google/gemma-4-31B-it"
PROMPT_VERSION = "v2-relabel"

PROMPT_TEMPLATE = """You are looking at a single image composed of {n} frames sampled in chronological order from the same short video clip. The frames are arranged in a {grid_desc} grid (read left-to-right, top-to-bottom: first frame, then subsequent frames in time order).

Write one detailed paragraph (about 100-150 words) describing what the video shows across these frames.

Describe the setting and location, the people present (their clothing, actions, and interactions), any visible objects, equipment, on-screen text, or logos, and how the scene changes between frames. Quote any legible on-screen text exactly. Describe only what is visible in the frames; do not speculate about audio, off-screen context, or what happens before or after the clip. If you are uncertain about a detail, omit it rather than guess. Write in natural declarative prose — no preamble, no bullet points, just the paragraph, and do not refer to the grid layout or frame numbers in the description."""


GRID_LAYOUTS = {
    1: (1, 1),
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    6: (2, 3),
    8: (2, 4),
    9: (3, 3),
}


def describe_grid(rows: int, cols: int) -> str:
    return f"{rows}-row by {cols}-column"


def build_grid(frame_paths: list[Path], target_w: int = 640, target_h: int = 360):
    """Compose frames into a single image. Returns (PIL.Image, (rows, cols))."""
    from PIL import Image

    n = len(frame_paths)
    rows, cols = GRID_LAYOUTS.get(n, (1, n))
    grid = Image.new("RGB", (target_w * cols, target_h * rows), (0, 0, 0))
    for i, fp in enumerate(frame_paths):
        img = Image.open(fp).convert("RGB").resize((target_w, target_h))
        r, c = divmod(i, cols)
        grid.paste(img, (c * target_w, r * target_h))
    return grid, (rows, cols)


def md5_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def probe_duration(mp4: Path) -> float:
    result = subprocess.run(
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
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_frames(mp4: Path, n_frames: int, tmp_dir: Path) -> tuple[list[Path], list[float]]:
    duration = probe_duration(mp4)
    if duration <= 0:
        return [], []
    timestamps = [duration * (i + 0.5) / n_frames for i in range(n_frames)]
    frames: list[Path] = []
    used_ts: list[float] = []
    for i, t in enumerate(timestamps):
        out = tmp_dir / f"frame_{i}.jpg"
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{t:.3f}",
                "-i",
                str(mp4),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(out),
            ],
            capture_output=True,
            timeout=30,
        )
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            frames.append(out)
            used_ts.append(round(t, 3))
    return frames, used_ts


def caption_one(
    client,
    model: str,
    mp4: Path,
    n_frames: int,
    output_dir: Path,
    max_tokens: int,
    max_retries: int = 2,
    overwrite: bool = False,
) -> tuple[str, str]:
    """Caption a single mp4. Returns (status, detail).
    status ∈ {"ok", "cached", "no_frames", "api_error"}"""
    url = f"./data/segments/{mp4.name}"
    vid = md5_url(url)
    out_path = output_dir / vid / "0.json"
    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        return ("cached", "")

    with tempfile.TemporaryDirectory(prefix=f"relabel_{vid}_") as tmp:
        tmp_root = Path(tmp)
        frames, timestamps = extract_frames(mp4, n_frames, tmp_root)
        if not frames:
            return ("no_frames", "ffmpeg produced no frames")

        # Stitch frames into one grid image. The endpoint caps at ~2 images
        # per request; a grid keeps us to a single image payload while still
        # giving the VLM coverage across the whole clip.
        grid, (rows, cols) = build_grid(frames)
        grid_path = tmp_root / "grid.jpg"
        grid.save(grid_path, quality=85)
        b64 = base64.standard_b64encode(grid_path.read_bytes()).decode("utf-8")

        prompt_text = PROMPT_TEMPLATE.format(
            n=len(frames), grid_desc=describe_grid(rows, cols)
        )
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
            {"type": "text", "text": prompt_text},
        ]

        caption = None
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}],
                )
                caption = resp.choices[0].message.content.strip()
                break
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
        if caption is None:
            return ("api_error", last_err[:200])

    record = {
        "video_id": vid,
        "url": url,
        "n_frames": len(frames),
        "frame_timestamps": timestamps,
        "grid_layout": f"{rows}x{cols}",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "caption": caption,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(".json.tmp")
    tmp_out.write_text(json.dumps(record, indent=2))
    tmp_out.rename(out_path)
    return ("ok", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY")
        or os.environ.get("SCALARLM_API_KEY")
        or "EMPTY",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only caption a random sample of this many mp4s (0 = all). Useful for a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-caption even if an output 0.json already exists. Off by default to make runs resumable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(args.segments_dir.glob("*.mp4"))
    if not mp4s:
        print(f"No .mp4 found in {args.segments_dir}", file=sys.stderr)
        return 1
    if args.limit and args.limit < len(mp4s):
        random.Random(args.seed).shuffle(mp4s)
        mp4s = mp4s[: args.limit]

    print(f"Segments: {args.segments_dir}  ({len(mp4s)} mp4s to consider)")
    print(f"Output:   {args.output_dir}")
    print(f"Endpoint: {args.endpoint}  model: {args.model}  n_frames: {args.n_frames}")
    print(f"Workers:  {args.workers}   api_key: {'set' if args.api_key != 'EMPTY' else 'EMPTY/none'}")

    from openai import OpenAI

    client = OpenAI(base_url=args.endpoint, api_key=args.api_key, timeout=90)

    counts = {"ok": 0, "cached": 0, "no_frames": 0, "api_error": 0}
    errors: list[tuple[str, str]] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                caption_one,
                client,
                args.model,
                mp4,
                args.n_frames,
                args.output_dir,
                args.max_tokens,
                overwrite=args.overwrite,
            ): mp4
            for mp4 in mp4s
        }
        for fut in as_completed(futures):
            mp4 = futures[fut]
            status, detail = fut.result()
            counts[status] += 1
            if status == "api_error":
                errors.append((mp4.name, detail))
            done = sum(counts.values())
            if done % 10 == 0 or done == len(mp4s):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = len(mp4s) - done
                eta = remaining / rate if rate > 0 else 0.0
                print(
                    f"  {done}/{len(mp4s)}  "
                    f"ok={counts['ok']} cached={counts['cached']} "
                    f"no_frames={counts['no_frames']} api_error={counts['api_error']}  "
                    f"rate={rate:.2f}/s  eta={eta/60:.1f}min",
                    flush=True,
                )

    elapsed = (time.time() - start) / 60
    print(
        f"\n=== Done in {elapsed:.1f} min ===  "
        f"ok={counts['ok']} cached={counts['cached']} "
        f"no_frames={counts['no_frames']} api_error={counts['api_error']}"
    )
    if errors:
        print("\nFirst few API errors:")
        for name, msg in errors[:5]:
            print(f"  {name}: {msg}")
    return 0 if counts["api_error"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
