# Caption Stats (§5.2, §7.1b)

Single-entry-point reproduction of the caption quality numbers in the paper,
with side-by-side comparison against InternVid and OpenVid-1M.

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
- `--hf-sample-size INT` (default 10_000) — captions streamed per comparator
- `--no-hf` — skip the InternVid / OpenVid-1M comparison arm

## Inputs

- `data/labels/<md5>/0.json` — one record per video with a `caption` field.
- HuggingFace streaming: `OpenGVLab/InternVid` (split `FLT`) and
  `nkp37/OpenVid-1M` (split `train`).

## Outputs

Under `results/caption_stats/`:
- `summary.json` — config + per-dataset stats (length, vocab, TTR, MTLD)
- `comparison.csv` / `comparison.md` — side-by-side summary table
- `word_freq.csv` — top-N content words for our corpus, with counts and fractions
- `length_histogram.png` — our caption-length histogram with the paper's 154-word
  claim marked
- `word_cloud.png` — figure for §5.2 (only if `wordcloud` is installed)

## InternVid access

`OpenGVLab/InternVid` is a **gated** HuggingFace dataset. To include it in the
comparison, request access at https://huggingface.co/datasets/OpenGVLab/InternVid
then authenticate:

```
huggingface-cli login
```

The script handles missing access gracefully — the InternVid row will show
zeros if you haven't been granted access.

## Paper targets

- §5.2: 154-word average caption length, content-word dominance
- §7.1b: TTR and MTLD on a 100K-caption sample (we sample 10k per HF comparator
  and use all ~6K captions in our 8K subset)

## Dependencies

Pinned in repo-root `requirements.txt`:
- `matplotlib` — histogram + word cloud rendering
- `wordcloud` — `word_cloud.png` figure
- `datasets` — HuggingFace streaming for the comparison arm
