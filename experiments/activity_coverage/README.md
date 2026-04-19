# Activity Category Coverage (§7.3b)

Zero-shot map each clip to the 200-leaf ActivityNet 1.3 taxonomy via CLIP
ViT-B/32, then report per-dataset coverage of the taxonomy, Gini coefficient
of the class frequency distribution, and the top over-/under-represented
classes relative to each comparator.

## Run

```
.venv/bin/python experiments/activity_coverage/run.py
```

Flags:
- `--datasets LIST` (default `ours,openvid,internvid`)
- `--sample-size-{ours,openvid,internvid} INT` — defaults 300 / 100 / 20
- `--frames-per-clip INT` (default 3) — mean-pooled image embedding
- `--model NAME` (default `ViT-B-32`), `--pretrained NAME` (default `openai`)
- `--classes-file PATH` (default `activitynet_classes.txt` — 200 leaves)
- `--device {auto,cpu,mps,cuda}` (default `auto`)
- `--extraction-workers INT`, `--batch-size INT`, `--seed INT`

## How the classifier works

1. Embed each of the 200 class names as text prompt `"a video of {class}"`
   via the CLIP text encoder.
2. For each clip, extract `frames_per_clip` frames evenly spaced across the
   clip, embed with the CLIP image encoder, mean-pool and L2-normalise.
3. Compute cosine similarities between the pooled clip embedding and all
   200 class embeddings; assign top-1 class.

Same CLIP backbone as `experiments/visual_grounding/`, so the embeddings
are directly comparable across the two experiments.

## Class list

`activitynet_classes.txt` — 200 leaf nodes of the ActivityNet 1.3
taxonomy, extracted from `activity_net.v1-3.min.json` (the official
evaluation file). Examples: `Archery`, `Baking cookies`, `Playing guitarra`,
`Shoveling snow`, `Zumba`.

## Outputs

`results/activity_coverage/`:
- `summary.json` — per-dataset stats: n_classified, classes covered, Gini,
  mean top-1 similarity score
- `comparison.csv` / `comparison.md` — side-by-side table
- `class_fractions.csv` — wide matrix: 200 classes × datasets, value is
  per-dataset fraction of clips assigned to that class
- `coverage_gaps.md` — for each comparator, top-15 classes over- and
  under-represented in ours
- `per_clip_<dataset>.csv` — per-clip top-1 class with similarity score

## Paper targets

§7.3b: 203-class ActivityNet taxonomy (we use 200 leaves), frequency
distribution, Gini coefficient, coverage-gap ranking to identify
underserved activity types in CC-licensed content.

## Known caveats

- Zero-shot CLIP classification is noisy on activity labels; top-1 is an
  approximate mapping, not a curated label. Gini and coverage-fraction
  should be interpreted as coarse distributional comparisons rather than
  per-category ground truth.
- ActivityNet is biased toward athletic and lifestyle activities; many
  People's Video clips (podcasts, lectures, software demos) have no close
  match in the taxonomy and will be routed to the closest class available.
  This effect is shared by comparators scored with the same classifier.

## Dependencies

- `torch`, `open_clip_torch`, `Pillow`
- `ffmpeg` / `ffprobe`
- Reuses fetchers from `experiments/visual_grounding/run.py`.
