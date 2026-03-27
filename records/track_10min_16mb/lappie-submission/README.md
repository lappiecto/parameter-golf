# Lappie AI — 11L VRL + Gated Attention + TrigramHash + Cautious Muon

**val_bpb: [PENDING H100 VALIDATION]** (local M4 Max: 1.36 BPB at 10k iterations)

## Run Command

```bash
# Setup (once)
pip install zstandard

# Train + evaluate
RUN_ID=lappie_submission \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed.

## Key Techniques

### Novel to this submission

| Technique | Description | Impact |
|---|---|---|
| **TrigramHash(4096)** | Hash consecutive token triples into 4096-bucket embedding table (dim=64), projected to model dim. Extends BigramHash to 3-token sequences. | Additive local context beyond pairs |
| **Value Residual Learning (VRL)** | Layer-0's V projection output cached and mixed into all subsequent layers via per-layer learned sigmoid gates. | Improved gradient flow through 11 layers |
| **Gated Attention** | Per-head sigmoid gates (init=0.5) controlling each attention head's output contribution. Enables head specialisation. | Heads learn when to contribute vs stay silent |
| **Cautious Muon** | Momentum updates masked when conflicting with current gradient direction: `update = momentum * (momentum * grad > 0)`. | Monotonic loss descent, zero overhead |
| **Partial RoPE (16/64)** | Rotary position encoding applied to only 16 of 64 head dimensions. Remaining 48 dimensions are position-invariant for content-only matching. | Better long-range content similarity |
| **LN Scale** | Each layer's attention and MLP outputs scaled by `1/sqrt(layer_idx + 1)`. Deeper layers contribute less, preventing residual stream magnitude explosion. | Deep network stability |

### From the competition stack

| Technique | Configuration |
|---|---|
| SmearGate | Per-dimension learned token blending with predecessor |
| BigramHash | 10240 buckets, 128-dim embeddings |
| Orthogonal init | QR decomposition for all linear weights |
| LeakyReLU(0.9)^2 | Negative slope 0.9, squared activation |
| Muon + Weight Decay | matrix_lr=0.02, momentum=0.99, WD=0.04 |
| SWA | Every 50 steps, last 40% of warmdown, 23 checkpoints |
| Sliding window eval | Stride=64, seq_len=2048 |
| Int6 quantisation | MLP and attention weights |
| zstd-22 | Compression (replacing zlib-9) |

## Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), LeakyReLU(0.9)^2
- U-Net skip connections, tied embeddings
- Total parameters: 28,173,501
- Compressed artifact: ~11.2 MB (4.8 MB under limit)

## Training

- Muon optimizer (Cautious variant) for matrices, Adam for embeddings/scalars
- Warmdown: 3000 iterations, dual-path (iteration + wallclock)
- SWA: collects checkpoints when lr_scale < 0.4
- Gradient clipping: 0.3

## Development

All research and iteration conducted on Apple M4 Max (128GB) in Cape Town, South Africa using the MLX framework. 23 experiments over 6 days, progressing from 2.41 BPB (baseline) to 1.36 BPB (full stack).

Comprehensive documentation available in the `docs/` directory:
- Research paper with full methodology and ablations
- Complete experiment audit trail (23 runs)
- Technical roadmap (80+ techniques evaluated)

## About

**Lappie AI** — Cape Town, South Africa
Built by Tom Shields ([@lappiecto](https://github.com/lappiecto))
