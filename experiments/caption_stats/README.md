# Caption Stats (§5.2, §7.1b)

Single-entry-point reproduction of the caption quality numbers in the paper.

## Run

```
python3 experiments/caption_stats/run.py
```

Optional flags:
- `--labels-dir PATH` (default `data/labels`)
- `--output-dir PATH` (default `results/caption_stats`)
- `--seed INT` (default 42)
- `--group-size INT` (default 100) — captions per pooled TTR group
- `--top-n-words INT` (default 500) — word-frequency rows emitted
- `--mtld-threshold FLOAT` (default 0.72) — standard McCarthy & Jarvis cutoff

## Inputs

`data/labels/<md5>/0.json` — one record per video with a `caption` field.

## Outputs

Under `results/caption_stats/`:
- `summary.json` — length stats, vocab size, global TTR, per-group TTR (matches
  `caption_diversity.py` output), MTLD
- `word_freq.csv` — top-N content words with counts and corpus fractions
- `length_histogram.png` — caption-length histogram with the paper's 154-word
  claim marked for comparison
- `word_cloud.png` — figure for §5.2 (only if `wordcloud` is installed)

## Paper targets

- §5.2: 154-word average caption length
- §7.1b: TTR and MTLD on a 100K-sample — we run over all ~6K captions in the 8K
  subset (smaller but representative)

## Dependencies

- Python 3.10+
- `matplotlib` (for histogram + word cloud rendering)
- `wordcloud` (optional — only needed for the `word_cloud.png` figure)
