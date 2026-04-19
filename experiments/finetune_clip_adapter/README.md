# CPU-friendly CLIP fine-tuning via frozen-feature adapters (§7.2a upgrade)

Small-scale fine-tuning experiment that upgrades the zero-shot retrieval
proxy of `experiments/zero_shot_retrieval/` into an actual no-training-
vs-training comparison, runnable in under 10 minutes on an M2 MPS.

## How it stays fast

The whole "model" is **two rank-r linear adapters** (~16 K trainable
parameters total) bolted onto the projection outputs of a frozen CLIP
ViT-B/32. Backprop never touches the encoders; image and text features
are pre-computed once and reused for every training step.

```
frozen CLIP image encoder ──► img features ──► [img_adapter]  ─┐
                                                                ├─► cosine ── InfoNCE
frozen CLIP text encoder  ──► txt features ──► [txt_adapter]  ─┘
```

Each adapter: `dim (512) → rank (8) → dim (512)` with the up-projection
zero-initialised so the adapter is identity at step 0.

## Run

```
.venv/bin/python experiments/finetune_clip_adapter/run.py
```

Flags:
- `--n-train INT` (default 4000), `--n-test INT` (default 1000) —
  random-shuffle split of the local People's Video corpus
- `--rank INT` (default 8), `--epochs INT` (default 40),
  `--batch-size INT` (default 256), `--lr FLOAT` (default 1e-3),
  `--temperature FLOAT` (default 0.05) — InfoNCE softmax temperature
- `--model`, `--pretrained`, `--device` — CLIP backbone (default
  ViT-B-32 openai on MPS)
- `--frames-per-clip INT` (default 1, middle frame)
- `--seed INT` (default 42)

## Expected wall time on M2 MPS

| Stage | Cost |
|---|---|
| Load CLIP ViT-B/32 | ~5 s |
| Frame extraction on 5 K clips (parallel ffmpeg) | ~60 s |
| CLIP feature pre-compute (image + text) | ~2-3 min |
| Adapter training (40 epochs × ~16 batches × cached features) | ~5-10 s |
| Zero-shot + adapter evaluation on N=1000 test | ~1 s |
| **Total** | **~4-5 min** cold |

Cached-feature rerun (e.g., sweeping rank/epochs/lr): ~15 s.

## Outputs

`results/finetune_clip_adapter/`:
- `summary.json` — config, zero-shot metrics, adapter metrics, Δ, loss curve
- `comparison.md` — side-by-side retrieval table
- `adapters.pt` — trained adapter weights (img + txt)

## Paper targets

§7.2a: text-to-video retrieval — delivers a concrete no-training-vs-
training comparison with the same evaluation protocol as
`tab:retrieval` (zero-shot within-dataset R@k on held-out pairs). The
full Video-LLaVA-scale fine-tuning experiments remain future work but
the adapter result converts the §7.2a claim from "captions carry the
signal" (zero-shot proxy) to "training on our captions actually
improves retrieval over the frozen baseline" (small-scale fine-tune).

## Why not full LoRA?

Real LoRA injects low-rank updates into the internal attention
projections of the encoder; that requires backprop through every
encoder block. Our adapters operate only on the final pooled
embeddings, so we pre-compute those embeddings once and train in a
closed-form-ish regime where each step is two linear layers + a
contrastive loss. This is a weaker form of adaptation but about 20×
cheaper per step on CPU/MPS, and sufficient to surface a measurable
training signal.

## Dependencies

Already in `requirements.txt`:
- `torch`, `open_clip_torch`, `Pillow`
- `ffmpeg` / `ffprobe`
- reuses fetchers from `experiments/visual_grounding/run.py`
