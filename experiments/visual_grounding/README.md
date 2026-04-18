# Visual Grounding (§7.1c)

CLIP cosine similarity between clip frames and Qwen3VL captions, averaged
across a subsample of the 8K-clip set.

## Run

```
.venv/bin/python experiments/visual_grounding/run.py
```

Flags:
- `--sample-size INT` (default 1000) — pairs to score
- `--frames-per-clip INT` (default 1) — evenly-spaced frames per clip
- `--model NAME` (default `ViT-B-32`) — open_clip model
- `--pretrained NAME` (default `openai`)
- `--device {auto,cpu,mps,cuda}` (default `auto`; MPS picked on Apple Silicon)
- `--batch-size INT` (default 32)
- `--extraction-workers INT` (default 8) — parallel ffmpeg workers
- `--seed INT` (default 42)

## M2 laptop defaults

- ViT-B/32 instead of the paper's ViT-L/14 (~3× faster, still a solid visual
  grounding proxy). Swap back with `--model ViT-L-14 --pretrained openai` if
  you want paper-fidelity numbers and have the time.
- 1,000-clip subsample instead of the paper's 50K. Increase with `--sample-size`.
- 1 middle frame per clip. Bump to 3 with `--frames-per-clip 3` for a mean
  pooled image embedding closer to the paper's protocol.

## Outputs

`results/visual_grounding/`:
- `summary.json` — model + config, mean/median/std cosine similarity, quantiles
- `per_clip_scores.csv` — per-clip score with caption length for correlation analysis
- `alignment_histogram.png` — score distribution

## Paper targets

§7.1c: "mean alignment score and its variance across domain clusters" using
CLIP ViT-L/14. The paper's comparator columns (InternVid, Panda-70M) are out
of scope here — they require local frames from those datasets.

## Dependencies

- `ffprobe` / `ffmpeg`
- `torch`, `open_clip_torch`, `Pillow` — pinned in `requirements.txt`
