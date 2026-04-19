# Demographic + Linguistic Diversity (§7.3c)

Estimates apparent gender, apparent age, and spoken-language distributions
across the three datasets using a shared CLIP ViT-B/32 backbone and
OpenAI Whisper-tiny.

Geographic-diversity analysis (the third prong of §7.3c in the paper) is
deferred: it requires per-video channel-country metadata that is not
preserved in the 8K validation subset.

## Run

```
.venv/bin/python experiments/diversity_stats/run.py
```

Flags:
- `--datasets LIST` (default `ours,openvid,internvid`)
- `--sample-size-{ours,openvid,internvid} INT` — defaults 150 / 60 / 20
- `--frames-per-clip INT` (default 1, center frame) for CLIP classification
- `--model`, `--pretrained` — CLIP backbone (default `ViT-B-32 / openai`)
- `--whisper-model NAME` (default `tiny`; `tiny.en` variants skip language ID)
- `--device {auto,cpu,mps,cuda}` — CLIP runs on MPS, Whisper forced to CPU
  (MPS has missing ops for some Whisper kernels)
- `--batch-size INT`, `--extraction-workers INT`, `--seed INT`

## How each measurement works

**Apparent gender / age (zero-shot CLIP).** For each clip, extract one
center frame, compute its CLIP image embedding, and assign the top-1
label among a small bank of text prompts:

Gender prompts:
```
"a photo with no person visible"
"a photo of a man"
"a photo of a woman"
```

Age prompts:
```
"a photo with no person visible"
"a photo of a child or teenager under 20"
"a photo of a young adult in their 20s or 30s"
"a photo of a middle-aged adult in their 40s or 50s"
"a photo of an elderly person over 60"
```

We also report the mean top1−top2 similarity margin as a proxy for
classifier confidence.

**Spoken language (Whisper-tiny lang ID).** Extract the first 30s of audio
via ffmpeg, resample to 16 kHz mono, run `whisper.load_audio` →
`pad_or_trim` → `log_mel_spectrogram` → `detect_language`. Report the
top-1 language code per clip and the distribution.

## Outputs

`results/diversity_stats/`:
- `summary.json` — config + per-dataset counts, fractions, mean margins,
  language counts, English / non-English fractions.
- `comparison.md` — three side-by-side tables: gender, age, language.
- `per_clip_<dataset>.csv` — per-clip gender/age/language with margins.

## Paper targets

§7.3c: apparent gender + age distribution via face attribute classifier;
fraction of non-English speech via language ID on audio. Geographic
diversity is now explicitly deferred in the paper text.

## Known caveats

- Zero-shot CLIP is a coarse demographic classifier — it conflates
  visual cues (clothing, hairstyle, lighting) with gender/age and is
  known to encode social biases. Treat as distributional profiling of
  how a CLIP-space classifier would bucket the frames, not as
  individual-level attribute labels.
- The "no person visible" bucket reduces the forced-choice bias on
  non-person clips, but not eliminates it; some close-up object shots
  will still be routed to man/woman.
- Whisper-tiny is small and occasionally mis-IDs short or music-heavy
  clips; top-1 language is a point estimate, not a calibrated
  probability.

## Dependencies

- `torch`, `open_clip_torch`, `Pillow`
- `openai-whisper` (adds `ffmpeg` python binding for audio loading)
- `ffmpeg`/`ffprobe` on PATH
- Reuses fetchers from `experiments/visual_grounding/run.py`.
