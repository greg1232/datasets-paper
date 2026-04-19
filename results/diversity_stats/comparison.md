## Apparent gender (zero-shot CLIP)

| Dataset | N | no_person | man | woman | Mean margin |
|---|---:|---:|---:|---:|---:|
| ours | 150 | 24.0% | 63.3% | 12.7% | 0.019 |
| openvid | 60 | 16.7% | 58.3% | 25.0% | 0.026 |
| internvid | 20 | 65.0% | 30.0% | 5.0% | 0.009 |

## Apparent age (zero-shot CLIP)

| Dataset | N | no_person | child | young_adult | middle_aged | elderly | Mean margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours | 150 | 14.7% | 6.0% | 17.3% | 56.7% | 5.3% | 0.012 |
| openvid | 60 | 3.3% | 13.3% | 28.3% | 53.3% | 1.7% | 0.015 |
| internvid | 20 | 35.0% | 15.0% | 15.0% | 35.0% | 0.0% | 0.007 |

## Language distribution (Whisper-tiny language ID)

| Dataset | N with audio | N no-audio | % English | % non-English | Top other langs |
|---|---:|---:|---:|---:|---|
| ours | 0 | 150 | 0.0% | 0.0% | — |
| openvid | 0 | 60 | 0.0% | 0.0% | — |
| internvid | 20 | 0 | 50.0% | 50.0% | ru(4), ko(3), fr(1) |
