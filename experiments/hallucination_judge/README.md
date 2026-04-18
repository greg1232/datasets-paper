# Hallucination Judge (§7.1a, automatic half)

Claude-as-judge evaluation of caption faithfulness: for each sampled clip,
three frames (begin/middle/end) are shown to Claude alongside the caption,
and the model is asked to enumerate any claim in the caption that is not
supported by what is visible in the frames.

Paper §7.1a originally specifies Gemini 1.5 Pro; we substitute **Claude
Haiku 4.5** for faster, cheaper judging at comparable fidelity.

## Run

```
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python experiments/hallucination_judge/run.py
```

Flags:
- `--datasets LIST` (default `ours,openvid,internvid`)
- `--sample-size-{ours,openvid,internvid} INT` — defaults 100 / 30 / 20
- `--frames-per-clip INT` (default 3) — begin/middle/end sampling
- `--model NAME` (default `claude-haiku-4-5-20251001`)
- `--workers INT` (default 6) — concurrent API calls
- `--api-key STR` — override the `ANTHROPIC_API_KEY` env var
- `--seed INT` (default 42)

## M2 laptop defaults

- Small samples (total ~150 API calls) so the run finishes in a few minutes.
- Haiku 4.5 chosen for latency + cost (~$0.003/call, ~1s/call).
- Comparator datasets reuse the video cache from the visual grounding
  experiment (`results/visual_grounding/_comparator_cache/`), so those
  clips aren't re-downloaded.

Expected wall time on M2 after caches are warm: ~1–2 minutes.

## Outputs

`results/hallucination_judge/`:
- `summary.json` — config + per-dataset stats
- `comparison.csv` / `comparison.md` — side-by-side table (N, valid, rate,
  avg claims per hallucinated caption, mean caption length)
- `per_clip_<dataset>.csv` — per-clip verdict with enumerated claims

## Paper targets

§7.1a automatic hallucination judge: fraction of captions containing at
least one unsupported claim, per dataset. The human-annotation half is
deferred (requires crowdsourcing infrastructure).

## Known caveats

- Three static frames cannot confirm or deny audio content, off-screen
  context, or activity that spans only un-sampled frames. The prompt
  explicitly instructs Claude not to flag such unverifiable claims as
  hallucinations, but some false-positive noise remains.
- The judge itself is an LLM; rates should be interpreted as a relative
  ranking between datasets rather than an absolute ground-truth rate.

## Dependencies

- `anthropic` (Claude Python SDK)
- ffmpeg, reused plumbing from `experiments/visual_grounding/`
