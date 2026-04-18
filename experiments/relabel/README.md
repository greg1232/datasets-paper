# Relabel (data regeneration pipeline)

Rebuilds the caption dataset for the 8K clips under `data/segments/` by
running Qwen-VL against each mp4 directly through an OpenAI-compatible
endpoint. Output goes to `data/labels-v2/<md5(url)>/0.json`, following the
same schema as `data/labels/` with additional provenance fields.

This exists because `data/labels/` (and the partially-regenerated
`data/fixed-labels/`) have a caption↔video mapping bug: 5 of 6 randomly
sampled clips had captions describing completely different videos. Root
cause was (a) the original pipeline showed Qwen3VL only the first two frames
(t=0 s, t=1 s) of each clip, and (b) some batched-executor index confusion.

## What this script does differently

| | Old `data/labels/` pipeline | This script |
|---|---|---|
| Frame sampling | `frame_number=0, interval=1 s, batch_size=2` → just t=0 and t=1 | N frames (default 4) evenly spaced across the clip's full duration |
| Pairing | Batched, with evidence of index shuffles | Each caption written under `md5(url)` of the exact mp4 whose frames were sent |
| Resume | — | Skips any `video_id` that already has `0.json` in the output dir |

## Run

```
.venv/bin/python experiments/relabel/run.py --limit 5
```

That captions 5 random mp4s as a smoke test. For the full 8K:

```
.venv/bin/python experiments/relabel/run.py
```

## Flags

- `--segments-dir PATH` (default `data/segments`)
- `--output-dir PATH` (default `data/labels-v2`)
- `--endpoint URL` (default `https://qwenbw.scalarllm.com/v1`)
- `--model NAME` (default `google/gemma-4-31B-it`, as reported by the endpoint's `/v1/models` listing; pass a different id to retarget)
- `--n-frames INT` (default 4) — frames per clip, evenly spaced
- `--max-tokens INT` (default 400)
- `--api-key STR` — falls back to `OPENAI_API_KEY` / `SCALARLM_API_KEY` env or
  `"EMPTY"` (some self-hosted endpoints don't require auth)
- `--workers INT` (default 4) — concurrent requests
- `--limit INT` (default 0 = all) — random sample of N mp4s for smoke testing
- `--seed INT` (default 42)

## Frame packaging

The endpoint at `qwenbw.scalarllm.com` caps each request at a single image
(multi-image requests return a server error that surfaces as
`AttributeError: 'str' object has no attribute 'choices'`). To preserve
temporal coverage on a single image, this script **stitches N frames into
one grid image** before sending:

| n_frames | grid |
|---:|---|
| 2 | 1×2 |
| 3 | 1×3 |
| 4 | 2×2 |
| 6 | 2×3 |
| 9 | 3×3 |

Each frame is resized to 640×360 before paste, so a 4-frame grid is 1280×720.

## Prompt

```
You are looking at a single image composed of {n} frames sampled in
chronological order from the same short video clip. The frames are arranged
in a {grid_desc} grid (read left-to-right, top-to-bottom: first frame, then
subsequent frames in time order).

Write one detailed paragraph (about 100-150 words) describing what the video
shows across these frames.

Describe the setting and location, the people present (their clothing,
actions, and interactions), any visible objects, equipment, on-screen text,
or logos, and how the scene changes between frames. Quote any legible
on-screen text exactly. Describe only what is visible in the frames; do not
speculate about audio, off-screen context, or what happens before or after
the clip. If you are uncertain about a detail, omit it rather than guess.
Write in natural declarative prose — no preamble, no bullet points, just the
paragraph, and do not refer to the grid layout or frame numbers in the
description.
```

Prompt version is recorded in each output record as `"prompt_version":
"v2-relabel"` so we can A/B this later.

## Output schema

```json
{
  "video_id": "<md5(url)>",
  "url": "./data/segments/<name>.mp4",
  "n_frames": 4,
  "frame_timestamps": [1.38, 4.15, 6.93, 9.70],
  "grid_layout": "2x2",
  "model": "google/gemma-4-31B-it",
  "prompt_version": "v2-relabel",
  "caption": "..."
}
```

## Dependencies

- `ffprobe` / `ffmpeg`
- `openai` (pinned in `requirements.txt`) — used as an OpenAI-compatible
  client against the ScalarLM endpoint.
