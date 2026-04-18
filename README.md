# The People's Video — Paper Experiments

Reproducibility code for the experiments in `docs/main.tex`. All experiments
are run on a representative **8K-clip subset** of the full 70.6M-clip corpus.

## Layout

```
data/
  labels/                          # 6,154 per-video JSON captions from Qwen3VL (tracked)
  segments/                        # 8,527 .mp4 clips (gitignored — ~7.3 GB)
  peoples-video-8k-segments/       # duplicate of segments/ (gitignored)
docs/                              # paper source (main.tex, references.bib)
experiments/                       # one subdirectory per experiment
  <experiment_name>/
    run.py                         # single entry point — reproduces the numbers
    README.md                      # what it computes, inputs, outputs
results/                           # committed outputs referenced by the paper
  <experiment_name>/
```

## Convention: one entry point per experiment

Every experiment has exactly one runnable entry point:

```
python experiments/<experiment_name>/run.py
```

It reads from `data/` and writes to `results/<experiment_name>/`. Results are
deterministic (fixed seeds) and committed so paper numbers are auditable.

## Experiments (paper §7)

| Section | Directory | Status |
|---|---|---|
| §5.1, §7.1b | `caption_stats/` | pending |
| §5.1 (clip durations) | `clip_durations/` | pending |
| §7.1c visual grounding | `visual_grounding/` | pending |
| §7.1d segmentation coherence | `clip_coherence/` | pending |
| §7.1a hallucination judge | `hallucination_judge/` | pending |
| §7.3b activity coverage | `activity_coverage/` | pending |
| §7.3c demographic/geographic | `diversity_stats/` | pending |
| §7.2 downstream tasks | `downstream_*/` | deferred (requires training) |
| §7.3a matched retrieval | `matched_retrieval/` | deferred (requires training) |
| §7.4 caption length ablation | `caption_length_ablation/` | deferred (requires training) |

Experiments marked *deferred* require fine-tuning on the dataset and will be
executed on a reduced scale against the 8K subset only.
