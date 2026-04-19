# Zero-shot Retrieval + Caption Length Ablation (§7.2a + §7.4)

A training-free reproducibility proxy for the paper's downstream retrieval
and caption-length ablation experiments. Uses the same CLIP ViT-B/32
backbone as §7.1c; no fine-tuning.

## Intent vs. paper original

| Paper (Phase 4) | This experiment |
|---|---|
| §7.2a: fine-tune CLIP4Clip, evaluate R@k on MSR-VTT, DiDeMo, ActivityNet Captions | Zero-shot CLIP retrieval: captions rank clips from the same dataset (within-dataset test split). R@1/5/10, MdR, both directions. |
| §7.4: truncate captions to 154/32/17 words, train retrieval + captioning models | Zero-shot retrieval at several truncation caps (default 132/64/32/17 words). R@1 as a function of caption length. |

The zero-shot version **does not** prove fine-tuned models benefit from our
captions; it tests whether the captions already carry the discriminative
signal that a fine-tuned retrieval model would exploit. Same intent,
weaker evidence, dramatically cheaper to run.

## Run

```
.venv/bin/python experiments/zero_shot_retrieval/run.py
```

Flags:
- `--datasets LIST` (default `ours,openvid,internvid`)
- `--sample-size-{ours,openvid,internvid} INT` — defaults 1000 / 200 / 20
- `--length-caps STR` (default `132,64,32,17`) — comma-separated caps in words
- `--frames-per-clip INT` (default 1) — middle frame per clip
- `--model`, `--pretrained` — CLIP backbone (default `ViT-B-32 / openai`)
- `--device {auto,cpu,mps,cuda}`, `--batch-size INT`, `--extraction-workers INT`, `--seed INT`

## How it works

1. For each dataset, sample N pairs, extract the middle frame, compute
   CLIP image embeddings (mean-pool if frames_per_clip > 1), L2-normalise.
2. For each caption, compute CLIP text embeddings at each truncation cap,
   truncating at sentence boundaries when possible.
3. Build the N×N text×image cosine similarity matrix; the correct match
   for caption i is clip i. Sort each row (and each column for v2t).
4. R@k = fraction of queries whose correct match is in the top-k.
5. Random-baseline R@1 is 1/N, so R@k figures are comparable only between
   datasets at matched N.

## Outputs

`results/zero_shot_retrieval/`:
- `summary.json` — config + per-dataset × per-length metrics.
- `retrieval_results.csv` — long-form table: one row per (dataset, length cap).
- `comparison.md` — one markdown table per dataset, rows = length caps.
- `length_ablation.png` — R@1 vs caption length cap, one line per dataset.

## Paper targets

§7.2a (retrieval utility) and §7.4 (caption-length ablation).

## Known caveats

- Within-dataset retrieval is less rigorous than transfer retrieval on
  MSR-VTT / DiDeMo / ActivityNet Captions; a caption that uniquely
  identifies its clip within a same-dataset distractor pool may still
  fail on a different-distribution benchmark.
- CLIP ViT-B/32 has a 77-token text-encoder cap, so caption truncation
  above ~55 words may not add (or even subtract) signal depending on
  where the text encoder clips. The ablation makes this visible.
- N differs across datasets (random baseline R@1 = 1/N), so direct
  cross-dataset R@k comparisons should be interpreted with care; the
  paper reports N explicitly in every row.
