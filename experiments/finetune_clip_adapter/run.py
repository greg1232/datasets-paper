#!/usr/bin/env python3
"""Small-scale fine-tuning proxy for §7.2a retrieval via projection adapters.

Upgrades the zero-shot retrieval proxy of experiments/zero_shot_retrieval/
into a true fine-tuning experiment, but with the CLIP encoders fully
frozen and trainable weights limited to two tiny rank-r projection
adapters (~16 K params total). All backprop happens on pre-computed
features, so the experiment runs in minutes on an M2 MPS and gives a
direct, no-training-vs-training comparison.

Protocol:
  1. Sample N clip-caption pairs from ours, split into train and test.
  2. Pre-compute frozen CLIP image + text features once.
  3. Initialise two rank-r adapter heads (image-side and text-side),
     each: dim -> rank -> dim with upsample zero-initialised so the
     adapter is identity at step 0.
  4. Train with symmetric InfoNCE contrastive loss on pre-computed
     features.
  5. Evaluate within-test retrieval R@1/5/10 for both the zero-shot
     baseline (no adapters) and the adapter-trained model on the same
     held-out pairs.

Single entry point. Reuses fetchers from experiments/visual_grounding/.
"""

import argparse
import csv
import json
import random
import statistics
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visual_grounding"))
from run import (  # noqa: E402
    fetch_ours,
    load_durations,
    load_clip,
    select_device,
    extract_frames_for_pair,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_DIR = REPO_ROOT / "data" / "segments"
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "labels"
DEFAULT_METADATA_CSV = REPO_ROOT / "results" / "clip_durations" / "video_metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "finetune_clip_adapter"


def compute_features(
    pairs: list[dict],
    model,
    preprocess,
    tokenizer,
    device: str,
    frames_per_clip: int,
    extraction_workers: int,
    batch_size: int,
    durations: dict,
) -> tuple[list[dict], "torch.Tensor", "torch.Tensor"]:
    """Return (kept_pairs, image_embeds, text_embeds). Pre-normalised."""
    import torch
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="ft_") as tmp:
        tmp_root = Path(tmp)
        all_frames: list[tuple[dict, list[Path]]] = [(p, []) for p in pairs]
        with ThreadPoolExecutor(max_workers=extraction_workers) as pool:
            futures = {
                pool.submit(
                    extract_frames_for_pair, p, durations, frames_per_clip, tmp_root
                ): i
                for i, p in enumerate(pairs)
            }
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                all_frames[i] = (pairs[i], fut.result())
                done += 1
                if done % 500 == 0 or done == len(pairs):
                    print(f"    extracted {done}/{len(pairs)}", flush=True)
        kept = [(p, fs) for p, fs in all_frames if fs]
        if not kept:
            return [], None, None

        flat: list[tuple[int, Path]] = []
        for pair_idx, (_, frames) in enumerate(kept):
            for f in frames:
                flat.append((pair_idx, f))
        per_pair_sum = [
            torch.zeros(1, dtype=torch.float32, device=device) for _ in kept
        ]
        per_pair_count = [0] * len(kept)
        with torch.no_grad():
            for i in range(0, len(flat), batch_size):
                batch = flat[i : i + batch_size]
                imgs, idxs = [], []
                for pair_idx, fp in batch:
                    try:
                        img = Image.open(fp).convert("RGB")
                    except Exception:
                        continue
                    imgs.append(preprocess(img))
                    idxs.append(pair_idx)
                if not imgs:
                    continue
                inp = torch.stack(imgs).to(device)
                emb = model.encode_image(inp)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                for j, pair_idx in enumerate(idxs):
                    if per_pair_count[pair_idx] == 0:
                        per_pair_sum[pair_idx] = emb[j].detach().clone()
                    else:
                        per_pair_sum[pair_idx] = per_pair_sum[pair_idx] + emb[j].detach()
                    per_pair_count[pair_idx] += 1
        image_embeds, valid = [], []
        for k, c in enumerate(per_pair_count):
            if c == 0:
                continue
            v = per_pair_sum[k] / c
            image_embeds.append(v / v.norm())
            valid.append(kept[k][0])
        image_embeds = torch.stack(image_embeds)

        captions = [p["caption"] for p in valid]
        text_chunks = []
        with torch.no_grad():
            for i in range(0, len(captions), batch_size):
                tokens = tokenizer(captions[i : i + batch_size]).to(device)
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                text_chunks.append(emb)
        text_embeds = torch.cat(text_chunks, dim=0)

        return valid, image_embeds.detach(), text_embeds.detach()


def retrieval_metrics(sims) -> dict:
    """sims: N×N cosine matrix (rows=text, cols=image). Correct match is the diagonal."""
    sims_cpu = sims.detach().to("cpu").float()
    n = sims_cpu.shape[0]
    t2v = []
    for i in range(n):
        correct = sims_cpu[i, i].item()
        higher = (sims_cpu[i] > correct).sum().item()
        t2v.append(higher + 1)
    v2t = []
    for j in range(n):
        correct = sims_cpu[j, j].item()
        higher = (sims_cpu[:, j] > correct).sum().item()
        v2t.append(higher + 1)
    def rk(rs, k):
        return sum(1 for r in rs if r <= k) / len(rs)
    return {
        "n": n,
        "t2v_R@1": rk(t2v, 1),
        "t2v_R@5": rk(t2v, 5),
        "t2v_R@10": rk(t2v, 10),
        "t2v_MdR": statistics.median(t2v),
        "v2t_R@1": rk(v2t, 1),
        "v2t_R@5": rk(v2t, 5),
        "v2t_R@10": rk(v2t, 10),
        "v2t_MdR": statistics.median(v2t),
    }


def main() -> int:
    import torch
    import torch.nn as nn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Softmax temperature for InfoNCE (lower = sharper).")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--extraction-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    durations = load_durations(args.metadata_csv)

    print(f"Loading CLIP {args.model} / {args.pretrained} on {device} ...", flush=True)
    model, preprocess, tokenizer = load_clip(args.model, args.pretrained, device)

    total = args.n_train + args.n_test
    print(f"Fetching {total} clip-caption pairs from ours ...", flush=True)
    pairs = fetch_ours(total, args.seed, args.labels_dir, args.segments_dir)
    print(f"  got {len(pairs)} pairs")
    if len(pairs) < total:
        print(
            f"  WARNING: requested {total} but only {len(pairs)} available; "
            "reducing n_train proportionally.",
            flush=True,
        )
        keep = len(pairs)
        ratio = args.n_test / total
        args.n_test = max(200, int(keep * ratio))
        args.n_train = keep - args.n_test

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    train_pairs = pairs[: args.n_train]
    test_pairs = pairs[args.n_train : args.n_train + args.n_test]

    print(f"\nPre-computing frozen CLIP features on {len(train_pairs) + len(test_pairs)} clips ...", flush=True)
    train_kept, train_img, train_txt = compute_features(
        train_pairs, model, preprocess, tokenizer, device,
        args.frames_per_clip, args.extraction_workers, args.batch_size, durations,
    )
    test_kept, test_img, test_txt = compute_features(
        test_pairs, model, preprocess, tokenizer, device,
        args.frames_per_clip, args.extraction_workers, args.batch_size, durations,
    )
    n_train = len(train_kept)
    n_test = len(test_kept)
    dim = train_img.shape[-1]
    print(f"  train: {n_train}  test: {n_test}  feature dim: {dim}", flush=True)

    # Zero-shot baseline on test.
    print("\nZero-shot baseline on test ...", flush=True)
    zero_sims = test_txt @ test_img.T
    zero_metrics = retrieval_metrics(zero_sims)
    print(
        f"  zero-shot t2v R@1={zero_metrics['t2v_R@1']:.1%}  "
        f"R@5={zero_metrics['t2v_R@5']:.1%}  R@10={zero_metrics['t2v_R@10']:.1%}  "
        f"MdR={zero_metrics['t2v_MdR']}"
    )

    # Rank-r adapter: dim -> rank -> dim, identity at init.
    class Adapter(nn.Module):
        def __init__(self, dim, rank):
            super().__init__()
            self.down = nn.Linear(dim, rank, bias=False)
            self.up = nn.Linear(rank, dim, bias=False)
            nn.init.zeros_(self.up.weight)
            self.scale = nn.Parameter(torch.ones(1) * 0.5)

        def forward(self, x):
            return x + self.scale * self.up(self.down(x))

    # Train on MPS but keep features in float32.
    train_img_dev = train_img.to(device)
    train_txt_dev = train_txt.to(device)
    test_img_dev = test_img.to(device)
    test_txt_dev = test_txt.to(device)

    img_adapter = Adapter(dim, args.rank).to(device)
    txt_adapter = Adapter(dim, args.rank).to(device)
    n_trainable = sum(p.numel() for p in img_adapter.parameters() if p.requires_grad) + \
                  sum(p.numel() for p in txt_adapter.parameters() if p.requires_grad)
    print(f"\nAdapter params (trainable): {n_trainable}", flush=True)

    opt = torch.optim.AdamW(
        list(img_adapter.parameters()) + list(txt_adapter.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    print(f"\nTraining: {args.epochs} epochs x {n_train // args.batch_size} batches, batch={args.batch_size}", flush=True)
    temperature = args.temperature
    loss_history = []
    for epoch in range(args.epochs):
        # shuffle train indices
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, n_train - args.batch_size + 1, args.batch_size):
            idx = perm[start : start + args.batch_size]
            img = train_img_dev[idx]
            txt = train_txt_dev[idx]
            img_a = img_adapter(img)
            img_a = img_a / img_a.norm(dim=-1, keepdim=True)
            txt_a = txt_adapter(txt)
            txt_a = txt_a / txt_a.norm(dim=-1, keepdim=True)
            # Symmetric InfoNCE
            logits = txt_a @ img_a.T / temperature
            labels = torch.arange(args.batch_size, device=device)
            loss_t2i = nn.functional.cross_entropy(logits, labels)
            loss_i2t = nn.functional.cross_entropy(logits.T, labels)
            loss = (loss_t2i + loss_i2t) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1
        avg_loss = epoch_loss / max(n_steps, 1)
        loss_history.append(avg_loss)
        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
            # Quick eval on test
            with torch.no_grad():
                tia = img_adapter(test_img_dev)
                tia = tia / tia.norm(dim=-1, keepdim=True)
                tta = txt_adapter(test_txt_dev)
                tta = tta / tta.norm(dim=-1, keepdim=True)
                sims = tta @ tia.T
                m = retrieval_metrics(sims)
            print(
                f"  epoch {epoch+1:>3}/{args.epochs}  loss={avg_loss:.4f}  "
                f"test t2v R@1={m['t2v_R@1']:.1%}  R@5={m['t2v_R@5']:.1%}  "
                f"R@10={m['t2v_R@10']:.1%}",
                flush=True,
            )

    # Final adapter eval.
    with torch.no_grad():
        tia = img_adapter(test_img_dev)
        tia = tia / tia.norm(dim=-1, keepdim=True)
        tta = txt_adapter(test_txt_dev)
        tta = tta / tta.norm(dim=-1, keepdim=True)
        adapter_sims = tta @ tia.T
        adapter_metrics = retrieval_metrics(adapter_sims)

    # Save.
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "config": {
                    "model": args.model,
                    "pretrained": args.pretrained,
                    "device": device,
                    "n_train": n_train,
                    "n_test": n_test,
                    "rank": args.rank,
                    "n_trainable_params": n_trainable,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "temperature": args.temperature,
                    "seed": args.seed,
                },
                "zero_shot": zero_metrics,
                "adapter": adapter_metrics,
                "delta": {
                    "t2v_R@1": adapter_metrics["t2v_R@1"] - zero_metrics["t2v_R@1"],
                    "t2v_R@5": adapter_metrics["t2v_R@5"] - zero_metrics["t2v_R@5"],
                    "t2v_R@10": adapter_metrics["t2v_R@10"] - zero_metrics["t2v_R@10"],
                    "v2t_R@1": adapter_metrics["v2t_R@1"] - zero_metrics["v2t_R@1"],
                },
                "loss_history": loss_history,
            },
            f,
            indent=2,
        )
    # Comparison markdown.
    lines = [
        f"## Retrieval on held-out test set (N={n_test})",
        "",
        "| Condition | t2v R@1 | t2v R@5 | t2v R@10 | t2v MdR | v2t R@1 | v2t R@5 | v2t R@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Zero-shot CLIP {args.model} | "
        f"{zero_metrics['t2v_R@1']:.1%} | {zero_metrics['t2v_R@5']:.1%} | "
        f"{zero_metrics['t2v_R@10']:.1%} | {int(zero_metrics['t2v_MdR'])} | "
        f"{zero_metrics['v2t_R@1']:.1%} | {zero_metrics['v2t_R@5']:.1%} | "
        f"{zero_metrics['v2t_R@10']:.1%} |",
        f"| + rank-{args.rank} adapter (trained on {n_train}) | "
        f"**{adapter_metrics['t2v_R@1']:.1%}** | **{adapter_metrics['t2v_R@5']:.1%}** | "
        f"**{adapter_metrics['t2v_R@10']:.1%}** | {int(adapter_metrics['t2v_MdR'])} | "
        f"**{adapter_metrics['v2t_R@1']:.1%}** | **{adapter_metrics['v2t_R@5']:.1%}** | "
        f"**{adapter_metrics['v2t_R@10']:.1%}** |",
        "",
        f"Δ t2v R@1 = {(adapter_metrics['t2v_R@1'] - zero_metrics['t2v_R@1']):+.1%}",
    ]
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n")

    # Save adapter weights.
    torch.save(
        {"img_adapter": img_adapter.state_dict(), "txt_adapter": txt_adapter.state_dict()},
        args.output_dir / "adapters.pt",
    )

    print("\n=== Final ===")
    print("\n".join(lines))
    print(f"Outputs → {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
