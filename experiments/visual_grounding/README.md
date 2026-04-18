# Visual Grounding (§7.1c)

CLIP cosine similarity between clip frames and their captions, computed for
three datasets with the **same** CLIP model for a fair comparison:

1. **Ours** — the 8K-clip subset (loaded from `data/labels/` + `data/segments/`).
2. **OpenVid-1M** — sampled from the OpenVidHD subset: metadata CSV +
   OpenVidHD shard fetched partially via `remotezip` (no full shard download).
3. **InternVid (FLT)** — sampled via HF streaming; each clip's YouTube segment
   fetched on demand via `yt-dlp --download-sections` (worst-quality, clip
   range only, ~500 KB per clip).

## Run

```
.venv/bin/python experiments/visual_grounding/run.py \
    --datasets ours,openvid,internvid
```

Flags:
- `--datasets LIST` (default `ours`) — any comma-separated subset
- `--sample-size-{ours,openvid,internvid} INT` — defaults 300 / 50 / 20 sized
  for a ~2-minute run on M2
- `--frames-per-clip INT` (default 1) — evenly-spaced frames per clip
- `--model NAME` (default `ViT-B-32`), `--pretrained NAME` (default `openai`)
- `--openvid-shard PATH` (default `OpenVidHD/OpenVidHD_part_1.zip`)
- `--device {auto,cpu,mps,cuda}` (default `auto`)
- `--batch-size INT`, `--extraction-workers INT`, `--seed INT`

## M2 laptop defaults

- ViT-B/32 instead of the paper's ViT-L/14 (~3× faster). Flip via `--model`.
- Small per-dataset samples to stay under a 5-minute wall time.
- Middle-frame only. Bump to `--frames-per-clip 3` for mean-pooled embeds.

Total wall time on M2 with defaults: ~1m40s after CLIP is downloaded and
OpenVidHD.csv is cached.

## Outputs

`results/visual_grounding/`:
- `summary.json` — config + per-dataset stats
- `comparison.csv` / `comparison.md` — side-by-side table
- `comparison_histogram.png` — overlaid score distributions
- `per_clip_scores_<dataset>.csv` — per-clip score with caption length

## OpenVid-1M comparator path

OpenVid-1M ships its videos in 18–46 GB `.zip` shards. Full-shard download is
infeasible on a laptop. This script:
1. Downloads `data/train/OpenVidHD.csv` (286 MB, cached by `huggingface_hub`).
2. Reads only the central directory of `OpenVidHD_part_1.zip` via `remotezip`
   (~1 s).
3. Intersects the CSV filenames with the shard file list (~26 k files).
4. Samples N from the intersection, fetches each via HTTP Range request
   (~500 KB per file).

Bandwidth: ~25 MB for 50 videos + 286 MB one-time for the CSV.

## InternVid (FLT) comparator path

Streams from `OpenGVLab/InternVid/InternVid-10M/FLT`, which is gated — login
via `hf auth login`. Each sample exposes YouTube ID + start/end timestamps
(no frames); `yt-dlp --download-sections` is used to fetch just the clip span
at worst quality. Expected yield ~30-40 % (some source videos are private or
deleted).

## Dependencies

- `ffprobe` / `ffmpeg`
- `torch`, `open_clip_torch`, `Pillow`
- `remotezip` — partial zip read over HTTP
- `yt-dlp` — InternVid clip fetch

All pinned in `requirements.txt`.
