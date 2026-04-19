# VideoQA Answerability (§7.2c)

A no-training reproducibility proxy for the paper's VideoQA fine-tuning
experiment. For each sampled clip we:

1. **Generate** N questions + reference answers from the *caption* alone
   (Claude sees the text, not the video). Each QA pair targets a
   concrete visual property (setting, attire, on-screen text, objects,
   actions).
2. **Answer** each question from the *frames* alone (Claude sees a
   4-frame 2×2 grid, not the caption). The model answers concisely or
   says "unknown".
3. **Judge** match via a separate Claude call comparing the two
   answers per question. Accuracy = fraction matched.

High accuracy = the caption described information that is actually
visible in the video, not hallucinated or inferred from off-screen
context. Same intent as §7.2c VideoQA (caption-grounded QA) without
any model training.

## Run

```
source config/env.sh
.venv/bin/python experiments/videoqa_answerability/run.py
```

Flags:
- `--datasets LIST` (default `ours,openvid,internvid`)
- `--sample-size-{ours,openvid,internvid} INT` — defaults 40 / 20 / 15
- `--n-questions INT` (default 3) — QA pairs per clip
- `--frames-per-clip INT` (default 4 → 2×2 grid)
- `--gen-model` / `--ans-model` / `--judge-model` — independent Claude
  IDs for each stage (all default to Haiku 4.5 for cost)
- `--workers INT` (default 6)

## M2 laptop defaults

75 clips × 3 API calls = ~225 Haiku 4.5 calls. Runs in ~5 min with
6 workers, ~$0.50 API.

## Outputs

`results/videoqa_answerability/`:
- `summary.json` — config + per-dataset overall / per-clip accuracies.
- `comparison.csv` / `comparison.md` — side-by-side table.
- `per_clip_<dataset>.csv` — per-clip accuracy + unknown count.
- `per_qa_<dataset>.json` — full QA transcripts for audit
  (caption-derived Q, reference answer, video-derived answer, match).

## Paper targets

§7.2c: overall accuracy + unknown-rate, per dataset. Proxies the paper's
NExT-QA / MSVD-QA / ActivityNet-QA fine-tuning evaluation with a Claude-
as-judge zero-shot evaluation against self-generated questions.

## Known caveats

- **Self-play bias**: the same model family (Claude) generates, answers,
  and judges. Stylistic consistency across stages can inflate match rates
  above what an independent human rater would give.
- **Caption-derived Qs can be leaky**: if the question lifts distinctive
  phrasing from the caption, the frame-based answerer might never recover
  it from visual content alone. Mitigated but not eliminated by the
  prompt's ban on direct-quote questions.
- **3 questions per clip is a small sample**. Per-clip accuracy is noisy;
  overall accuracy across the dataset is the stable metric.

## Dependencies

- `anthropic` (Claude SDK)
- `ffmpeg` / `ffprobe`
- Reuses `experiments/visual_grounding/run.py` fetchers and
  `experiments/relabel/run.py` grid-stitching.
