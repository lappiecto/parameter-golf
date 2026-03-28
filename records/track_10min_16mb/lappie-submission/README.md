# Lappie AI — 11L Full Stack + Cautious Muon

**val_bpb: [PENDING H100 VALIDATION]** (local M4 Max: 1.3471 BPB at 20k iterations)

## Run Command

```bash
# Setup (once)
pip install zstandard

# Train + evaluate
RUN_ID=lappie_submission \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed.

## Summary

Full-stack submission combining established competition techniques with Cautious Muon (momentum masking for the Muon optimizer). All development and iteration conducted on an Apple M4 Max in Cape Town, South Africa — no cloud GPUs used during development.

23 experiments over 6 days, progressing from 2.41 BPB (naive baseline) to 1.3471 BPB (20k local iterations). Comprehensive experiment audit trail and documentation.

## Techniques Used

### Training — Architecture
| Technique | Source | Description |
|---|---|---|
| SmearGate | PR #198 (@unnir) | Per-dimension learned token blending with predecessor |
| BigramHash(10240) | PR #180 (@thwu1) | XOR-hashed token pair embeddings |
| TrigramHash(4096) | PR #553 (@Vighaneshs) | 3-token hash embeddings |
| QuadgramHash(2048) | PR #553 (@Vighaneshs) | 4-token hash embeddings |
| VRL | PR #562 (@bigbag), arXiv:2410.17897 | Layer-0 V mixed into all layers via learned gates |
| Gated Attention | PR #185 (@dttdrv), arXiv:2505.06708 | Per-head sigmoid gates on attention output |
| Partial RoPE (16/64) | PR #315 (@jfprincz) | RoPE on 25% of head dimensions |
| LN Scale | PR #315 (@jfprincz) | 1/sqrt(layer+1) scaling on residual contributions |
| LeakyReLU(0.9)^2 | PR #185 (@dttdrv) | Negative slope 0.9, squared activation |
| Orthogonal Init | PR #198 (@unnir) | QR decomposition for all linear weights |

### Training — Optimization
| Technique | Source | Description |
|---|---|---|
| **Cautious Muon** | **Novel in this competition** | Momentum updates masked when sign conflicts with gradient: `buf * (buf * g > 0)`. Based on Cautious Optimizers (Liang et al.) |
| Muon WD 0.04 | PR #180 (@thwu1) | Decoupled weight decay |
| Grad clip 0.3 | Standard | Global gradient norm clipping |
| Late QAT (STE) | PR #315 (@jfprincz) | Fake int6 quantisation during warmdown |
| SWA | PR #180 (@thwu1) | 23 averaged checkpoints, last 40% of warmdown |

### Compression
| Technique | Source | Description |
|---|---|---|
| GPTQ-lite clip search | PR #414 (@signalrush) | 5 clip percentile candidates per row |
| Int6 MLP + Int6 Attn | Standard | Mixed-precision quantisation |
| zstd-22 | Standard | Replaces zlib-9 |

### Evaluation
| Technique | Source | Description |
|---|---|---|
| Sliding window (stride=64) | PR #198 | 1984 tokens of context per scored position |

## Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), LeakyReLU(0.9)^2
- U-Net skip connections, tied embeddings
- Total parameters: 28,255,422
- Compressed artifact: ~11.6 MB (well under 16 MB limit)

## Local Results (Apple M4 Max, 128GB)

| Run | BPB | Steps | Key Change |
|---|---|---|---|
| Naive baseline | 2.41 | 200 | Starting point |
| +10L, 3x MLP, Muon 0.99 | 1.94 | 500 | Hyperparameter tuning |
| +SmearGate, BigramHash | 1.69 | 1000 | Architecture techniques |
| +SWA (fixed), 5k iters | 1.42 | 5000 | SWA bug fix |
| +VRL, Gated Attn, Partial RoPE, LN Scale, C-Muon | 1.36 | 10000 | Architecture stack |
| **+QuadgramHash, Late QAT, GPTQ clip, 80 shards** | **1.3471** | **20000** | **Full stack** |

## What's Unique About This Submission

1. **Cautious Muon** — the only technique we could not find in any prior submission. Masks momentum updates where the momentum buffer conflicts with the current gradient direction. Zero overhead, monotonic loss improvement.

2. **Development methodology** — entire research cycle conducted on consumer hardware (M4 Max) in Cape Town, South Africa. 23 experiments, full audit trail, PhD-level documentation. Demonstrates that competitive Parameter Golf research is accessible without cloud GPU access during development.

3. **Documentation depth** — see `docs/` directory for research paper, experiment log, technical roadmap, and deep dive.

## Documentation

| Document | Description |
|---|---|
| [docs/RESEARCH_PAPER.md](../../../docs/RESEARCH_PAPER.md) | Research paper with methodology and ablations |
| [docs/EXPERIMENT_LOG.md](../../../docs/EXPERIMENT_LOG.md) | Complete audit trail of all 23+ experiments |
| [docs/ROADMAP.md](../../../docs/ROADMAP.md) | 80+ techniques evaluated |
| [docs/DEEP_DIVE.md](../../../docs/DEEP_DIVE.md) | How every component works |

## About

**Lappie AI** — Cape Town, South Africa
Built by Tom Shields ([@lappiecto](https://github.com/lappiecto))
