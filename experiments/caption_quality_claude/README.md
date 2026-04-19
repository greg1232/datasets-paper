# Caption Quality via Reference-Model Agreement (§7.2b)

A no-training reproducibility proxy for the paper's Video-LLaVA fine-tuning
captioning experiment. For each sampled clip:

1. Extract 4 frames evenly across the clip, stitch into a 2×2 grid
   (same as `experiments/relabel/run.py`).
2. Ask **Claude Sonnet** (a stronger independent vision model) to produce a
   reference caption from the same frames.
3. Compute **BLEU-4**, **METEOR**, and **ROUGE-L F** between the dataset's
   caption and Claude's reference.

High reference-model agreement means the dataset caption is describing
content a strong independent captioner would also describe from the same
frames. Lower agreement can indicate either caption hallucination or
stylistic divergence from the reference.

## Run

```
source config/env.sh
.venv/bin/python experiments/caption_quality_claude/run.py
```

Flags:
- `--datasets LIST` (default `ours,openvid,internvid`)
- `--sample-size-{ours,openvid,internvid} INT` — defaults 100 / 50 / 20
- `--frames-per-clip INT` (default 4 → 2×2 grid)
- `--model NAME` (default `claude-sonnet-4-6-20251001`)
- `--workers INT` (default 6) — concurrent Claude calls
- `--api-key STR` — override `ANTHROPIC_API_KEY` env
- `--seed INT` (default 42)

## M2 laptop defaults

170 Sonnet 4.6 calls, 4-frame grid each. Runs in ~3 min with 6 workers,
~$4–5 in API cost (Sonnet is ~10× Haiku pricing but much better at
descriptive captioning).

## Outputs

`results/caption_quality_claude/`:
- `summary.json` — config + per-dataset mean BLEU-4 / METEOR / ROUGE-L.
- `comparison.csv` / `comparison.md` — side-by-side table.
- `per_clip_<dataset>.csv` — per-clip scores + Claude's reference
  caption (truncated to 500 chars for the CSV).

## Paper targets

§7.2b: reference-caption agreement is a stand-in for the original
CIDEr/METEOR/ROUGE-L metrics against manual references on MSR-VTT /
YouCook2 / ActivityNet Captions. Using a strong VLM as the reference
avoids the need for manual annotation on the 8K subset, while preserving
the interpretability of the same three n-gram metrics.

## Known caveats

- Agreement with a reference model is **not** agreement with a human
  reference. Two systematic errors the reference model makes will not
  be caught; both the dataset caption and the reference might be wrong.
- BLEU-4 tends to be low (<0.3) for free-form descriptive captions even
  when content matches, because surface-form variability dominates.
  Report all three metrics together; METEOR and ROUGE-L are more
  robust to paraphrasing.
- Stylistic divergence (e.g. Claude tends to open with "The video
  shows..." whereas a dataset caption might open with a direct
  subject) depresses all three metrics without indicating factual
  disagreement.

## Dependencies

- `anthropic` (Claude SDK)
- `nltk` (BLEU-4 + METEOR; METEOR requires NLTK's WordNet data — the
  script downloads it on first run)
- `rouge-score` (ROUGE-L)
- `ffmpeg`/`ffprobe`
- Reuses `experiments/visual_grounding/run.py` fetchers and
  `experiments/relabel/run.py` grid-stitching.
