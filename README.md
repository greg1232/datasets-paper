# The People's Video — Paper + Reproducibility Repo

Source and reproducibility code for *The People's Video: Exploring Commercially
Licensed Multimodal Data* (`docs/main.tex`). Every experiment referenced in
§7 of the paper has a single-entry-point script under `experiments/` that
reproduces the numbers on an 8,522-clip validation subset of the full 70.6M-clip
corpus.

- **Paper**: `docs/main.tex`, `docs/references.bib`
- **Raw scene segments**: 8,527 `.mp4` files under `data/segments/` (stored
  via Git LFS; 7.3 GB)
- **Captions**: 8,522 JSON records under `data/labels/` (one per clip;
  regenerated with our caption-pairing-safe pipeline)
- **Experiments**: 12 reproducibility scripts under `experiments/`
- **Committed outputs**: `results/<experiment>/` — tables, JSON, plots,
  per-clip CSVs

## Quick start

```bash
git clone git@github.com:greg1232/datasets-paper.git
cd datasets-paper
git lfs pull                            # 7.3 GB of scene-segmented mp4 files

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Experiments that use Claude need an Anthropic API key; put it in a
# file that is gitignored:
mkdir -p config && cat > config/env.sh <<'EOF'
export ANTHROPIC_API_KEY=sk-ant-...
EOF

# Example: reproduce §7.1b caption-diversity table
.venv/bin/python experiments/caption_stats/run.py

# Example: reproduce §7.2a retrieval + §7.4 length ablation
.venv/bin/python experiments/zero_shot_retrieval/run.py

# Example: reproduce §7.1a hallucination-judge table (needs Claude)
source config/env.sh
.venv/bin/python experiments/hallucination_judge/run.py
```

All scripts default to samples that finish in under 5 minutes on a MacBook
with M-series Apple Silicon. Each has a `--sample-size-*` flag if you
want more stable numbers.

## Experiment index

Every experiment is a single `run.py` entry point with a scoped
`README.md`. Tables below show each paper section, the script, and the
headline number it reproduces.

| Paper section | Script | Headline result (ours) |
|---|---|---|
| §5.1 clip durations | `clip_durations/` | 37/22/23/8/10% bins; 11.57s mean |
| §5.2, §7.1b caption stats | `caption_stats/` | MTLD 75.6; mean 132 words |
| §7.1a hallucination judge | `hallucination_judge/` | 0.35 flagged claims / 100 words |
| §7.1c CLIP visual grounding | `visual_grounding/` | Cosine 0.302 |
| §7.1d within-clip coherence | `clip_coherence/` | 95% fully coherent |
| §7.2a text-to-video retrieval | `zero_shot_retrieval/` | R@1 25.2% (N=1000) |
| §7.2b video captioning | `caption_quality_claude/` | BLEU-4 0.127, METEOR 0.390 |
| §7.2c VideoQA | `videoqa_answerability/` | 60.8% overall accuracy |
| §7.2d temporal grounding | `temporal_grounding/` | argmax 34.6% (chance 33.3%) |
| §7.3b activity coverage | `activity_coverage/` | 82/200 ActivityNet classes |
| §7.3c demographic + linguistic | `diversity_stats/` | CLIP demographics + Whisper lang-ID |
| §7.4 caption length ablation | (folded into `zero_shot_retrieval/`) | R@1 plateaus at 64 words |
| data regeneration pipeline | `relabel/` | Qwen-VL-style captions via ScalarLM |

## Full-scale claims vs measured results

The paper separates two layers of evidence cleanly in §7:

1. **Full-scale protocols** described per-task (CLIP4Clip fine-tuning for
   retrieval, Video-LLaVA fine-tuning for captioning, temporal-grounding
   model training, etc.) — these require GPU infrastructure and remain
   future work.
2. **Reproducibility proxies** reported in this repo — zero-shot CLIP or
   Claude-as-judge evaluations that test whether the captions and clips
   carry the signal a fine-tuned model would exploit. These run in under
   5 minutes each on a MacBook.

Cross-dataset `R@k` comparisons in `zero_shot_retrieval/` are valid only
at matched `N`; the paper reports `N` explicitly.

## Layout

```
data/
  labels/<md5(url)>/0.json   # regenerated captions (tracked)
  segments/*.mp4              # scene-segmented videos (LFS)
  fixed-labels/               # partial earlier fix attempt (tracked for history)
docs/
  main.tex, references.bib   # paper source
experiments/
  <name>/
    run.py                    # single reproducible entry point
    README.md                 # inputs, outputs, caveats, deps
results/
  <name>/                     # committed outputs (summary.json, CSVs, plots)
config/                        # gitignored (API keys etc.)
.venv/                         # gitignored
```

## Conventions

- Every experiment accepts `--datasets ours[,openvid,internvid]`,
  `--sample-size-*`, and `--seed` flags. Defaults are chosen for ≤5 min
  on M-series Apple Silicon.
- Datasets fetched for comparators:
  - `ours`: loaded directly from `data/labels/` + `data/segments/`.
  - `openvid`: OpenVid-1M HD subset via `huggingface_hub` caption CSV +
    `remotezip` partial HTTP reads against 45 GB shards. See
    `experiments/visual_grounding/README.md` for the partial-download
    protocol.
  - `internvid`: InternVid-10M FLT split via `yt-dlp` clip fetches
    (requires `hf auth login` — dataset is gated).
- Comparator videos are cached under
  `results/visual_grounding/_comparator_cache/` (gitignored) so reruns
  are fast.
- Claude model IDs:
  - `claude-haiku-4-5-20251001` — judge / generation in most experiments
  - `claude-sonnet-4-6` — reference captioner in
    `caption_quality_claude/`

## Caption regeneration pipeline

`experiments/relabel/run.py` rebuilds the caption set from
`data/segments/*.mp4` through an OpenAI-compatible vision endpoint,
stitching four evenly-spaced frames into a 2×2 grid per clip. This
pipeline was used to produce the captions shipped in `data/labels/`.
The endpoint and model are configurable; see
`experiments/relabel/README.md`.

## Reproducing the paper end-to-end

```bash
# Intrinsic quality (§7.1)
.venv/bin/python experiments/caption_stats/run.py
.venv/bin/python experiments/clip_durations/run.py
.venv/bin/python experiments/visual_grounding/run.py
source config/env.sh
.venv/bin/python experiments/hallucination_judge/run.py
.venv/bin/python experiments/clip_coherence/run.py

# Downstream (§7.2, §7.4)
.venv/bin/python experiments/zero_shot_retrieval/run.py
.venv/bin/python experiments/caption_quality_claude/run.py
.venv/bin/python experiments/videoqa_answerability/run.py
.venv/bin/python experiments/temporal_grounding/run.py

# Licensing effect (§7.3)
.venv/bin/python experiments/activity_coverage/run.py
.venv/bin/python experiments/diversity_stats/run.py
```

Total wall time with warm comparator caches: roughly 30 minutes on an
M2 MacBook. Total API cost (Haiku 4.5 + Sonnet 4.6): under $10.

## Dependencies

Pinned in `requirements.txt`:

- `torch`, `open_clip_torch`, `Pillow` — CLIP backbone
- `matplotlib`, `wordcloud` — figures
- `datasets`, `huggingface_hub`, `remotezip` — comparator fetching
- `yt-dlp` — InternVid clip fetch
- `anthropic` — Claude-as-judge experiments
- `openai-whisper` — audio language ID
- `nltk`, `rouge-score` — BLEU / METEOR / ROUGE-L for caption quality
- `openai` — OpenAI-compatible client (used by `relabel/`)

System: `ffmpeg` / `ffprobe` on `PATH`.

## License

Code and paper content: TBD. Dataset contents are redistributed under
the original per-video CC-BY 4.0 or CC0 1.0 licenses declared by
creators at the time of collection (see §2.1 of the paper for the
licensing scope and limitations).
