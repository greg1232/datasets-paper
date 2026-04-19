## Retrieval on held-out test set (N=1000)

| Condition | t2v R@1 | t2v R@5 | t2v R@10 | t2v MdR | v2t R@1 | v2t R@5 | v2t R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot CLIP ViT-B-32 | 26.8% | 51.5% | 63.2% | 5 | 25.0% | 51.1% | 65.8% |
| + rank-8 adapter (trained on 4000) | **30.4%** | **62.2%** | **74.3%** | 3 | **27.7%** | **60.9%** | **73.2%** |

Δ t2v R@1 = +3.6%
