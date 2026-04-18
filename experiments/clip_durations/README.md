# Clip Durations (§5.1)

Single-entry-point reproduction of the clip-duration distribution cited in
Section 5.1 (bin percentages) and Table 1 (11.2s average clip length).

## Run

```
.venv/bin/python experiments/clip_durations/run.py
```

Flags:
- `--segments-dir PATH` (default `data/segments`)
- `--output-dir PATH` (default `results/clip_durations`)
- `--workers INT` (default 8) — parallel ffprobe workers

## Inputs

`data/segments/*.mp4` — scene-segmented clips (8,527 on the 8K subset).

## Outputs

Under `results/clip_durations/`:
- `video_metadata.csv` — per-clip ffprobe metadata (duration, resolution,
  codec, bitrate). Same schema as the reference `video_metadata.csv` in
  llm-caption/infra/paper.
- `summary.json` — mean/median/std duration, per-bin percentages with paper
  deltas, resolution stats.
- `clip_duration_histogram.png` — linear histogram of clip duration up to 60s
  with measured mean and paper's 11.2s reference.
- `clip_durations.png` — pie chart matching the figure used in the paper.

## Paper targets

- §5.1: 0-2s 37%, 2-4s 22%, 4-10s 23%, 10-20s 8%, >20s 10%
- Table 1: Len_Clip (mean) = 11.2s; resolution = 720P

## Dependencies

- `ffprobe` (from `ffmpeg`) — `brew install ffmpeg`
- Python 3.10+
- `matplotlib` (optional — histogram and pie chart rendering)
