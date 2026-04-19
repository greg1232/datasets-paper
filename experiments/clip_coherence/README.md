# Clip Coherence Judge (§7.1d)

Within-clip coherence evaluation: for each sampled clip, Claude is shown the
first and last frame and asked to rate on a 3-point scale whether they
plausibly belong to the same semantic scene. Clips rated 1 (different
scenes) indicate segmentation failures where PySceneDetect missed an
intended scene boundary.

## Run

```
source config/env.sh
.venv/bin/python experiments/clip_coherence/run.py
```

Flags:
- `--datasets LIST` (default `ours`; extend to `ours,openvid,internvid` to
  compare against comparator segmentations)
- `--sample-size-{ours,openvid,internvid} INT` — defaults 100 / 30 / 20
- `--model NAME` (default `claude-haiku-4-5-20251001`)
- `--workers INT` (default 6) — concurrent Claude API calls
- `--api-key STR` — override `ANTHROPIC_API_KEY` env
- `--seed INT` (default 42)

## M2 laptop defaults

100 ours-only calls complete in ~1 min on 6 workers at ~$0.003/call.

## Outputs

`results/clip_coherence/`:
- `summary.json` — per-dataset rating distribution + fully-coherent rate
- `comparison.csv` / `comparison.md` — side-by-side table
- `per_clip_<dataset>.csv` — per-clip rating + Claude's one-sentence reason

## Paper targets

§7.1d:
- Within-clip coherence: fraction rated fully coherent (rating=3), via VLM
  judge on first+last frame of each clip.
- Segmentation precision/recall: deferred (requires human-annotated ground-
  truth boundaries).
- Fixed-interval 10-second baseline: deferred (requires access to full
  source videos before scene segmentation; the 8K subset only ships the
  post-segmentation clips).

## Known caveats

- First/last frames are not always representative of the whole clip — a
  clip could have a momentary cutaway at the end yet still be coherent in
  its bulk. Judge ratings should be read as upper bounds on segmentation
  failure, not precise per-clip verdicts.
- The judge is an LLM; rates should be interpreted as relative ranking
  between datasets rather than absolute ground-truth rates.

## Dependencies

- `anthropic` (Claude Python SDK)
- `ffmpeg` / `ffprobe`
- Reuses fetchers from `experiments/visual_grounding/run.py`.
