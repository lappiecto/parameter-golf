# Lappie AI — Parameter Golf Submission

**Team**: Lappie AI (Cape Town, South Africa)
**Author**: Tom Shields ([@tomasi001](https://github.com/tomasi001))
**Best Local BPB**: 1.36 (Apple M4 Max, 10,000 iterations)
**Projected H100 BPB**: 1.10-1.15 (8xH100 SXM, 20,000 iterations)

---

## Architecture

11-layer transformer with 6 techniques beyond the competition baseline:

| Technique | Description | Impact |
|---|---|---|
| **SmearGate** | Per-dimension learned blending of adjacent token embeddings | Bigram awareness before attention |
| **BigramHash (10240)** | XOR-hashed token pair embeddings, 128-dim, projected to model dim | Explicit pair features |
| **TrigramHash (4096)** | Three-token hash embeddings, 64-dim (novel — not in any other submission) | Three-token context features |
| **Value Residual Learning** | Layer-0 value vectors mixed into all subsequent layers via learned sigmoid gates | Better gradient flow through depth |
| **Gated Attention** | Per-head sigmoid gates controlling each head's output contribution | Head specialisation |
| **Partial RoPE (16/64)** | Rotary position encoding on 25% of head dimensions only | Position-invariant content matching |
| **LN Scale** | Layer outputs scaled by 1/sqrt(layer_idx+1) | Deep layer stabilisation |
| **Cautious Muon** | Momentum updates masked when conflicting with gradient direction | Monotonic loss descent |
| **LeakyReLU(0.9)^2** | 0.9 negative slope preserves gradient flow, squaring maintains sparsity | 0.013 BPB over relu^2 |
| **Orthogonal Init** | QR decomposition for all linear weights | Uniform gradient flow from step 1 |

## Training Configuration

| Parameter | Value |
|---|---|
| Layers | 11 |
| Model Dimension | 512 |
| Heads / KV Heads | 8 / 4 (GQA) |
| MLP Expansion | 3x (hidden=1536) |
| Vocabulary | 1024 (SentencePiece BPE) |
| Sequence Length | 2048 |
| Optimizer | Cautious Muon (matrices) + Adam (embeddings/scalars) |
| Matrix LR | 0.02 |
| Muon Momentum | 0.99 |
| Weight Decay | 0.04 (decoupled, Muon only) |
| Gradient Clip | 0.3 |
| Warmdown | 3000 iterations |
| SWA | Every 50 steps, last 40% of warmdown (23 checkpoints) |
| Quantisation | Int6 (MLP + Attention) + zstd-22 |
| Compressed Size | 11.2 MB (4.8 MB under 16 MB limit) |

## Results

### Local Development (Apple M4 Max, 128GB)

| Run | BPB | Steps | Key Change |
|---|---|---|---|
| Baseline | 2.41 | 200 | Starting point |
| Tuned | 1.94 | 500 | 10L, 3x MLP, Muon 0.99 |
| +SmearGate/BigramHash | 1.69 | 1000 | Architecture techniques |
| +SWA (fixed) | 1.42 | 5000 | SWA actually collecting checkpoints |
| **Full Stack** | **1.36** | **10000** | **+VRL, Gated Attn, Partial RoPE, LN Scale, C-Muon** |

23 total experiments documented with full audit trail. See [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

### Ablation Summary

See [docs/RESEARCH_PAPER.md](docs/RESEARCH_PAPER.md) Section 7 for complete ablation data.

## Documentation

| Document | Description |
|---|---|
| [docs/RESEARCH_PAPER.md](docs/RESEARCH_PAPER.md) | PhD-level research paper with full methodology, ablations, and analysis |
| [docs/RESEARCH_PAPER_ELI5.md](docs/RESEARCH_PAPER_ELI5.md) | Accessible version — same structure, explained for anyone |
| [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | Complete audit trail of all 23 experiments |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 80+ techniques mapped across 9 phases |
| [docs/DEEP_DIVE.md](docs/DEEP_DIVE.md) | How every model component works |

## Code

| File | Description |
|---|---|
| `train_gpt_mlx.py` | MLX training script for Apple Silicon (1784 lines) |
| `train_gpt.py` | PyTorch/CUDA training script (competition base) |
| `dashboard/` | Custom experiment dashboard (HTML + Python server) |

## Development Environment

All research and development was conducted on:
- **Hardware**: Apple M4 Max, 128GB unified memory
- **Framework**: MLX (Apple's ML framework for Apple Silicon)
- **Location**: Cape Town, South Africa
- **Throughput**: ~15-18k tokens/sec (vs ~500k+ on 8xH100)

## Running

### Local (Apple Silicon)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm zstandard
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 20

RUN_ID=local_run \
NUM_LAYERS=11 ITERATIONS=5000 MLP_MULT=3 TRAIN_SEQ_LEN=2048 \
USE_SMEARGATE=1 BIGRAM_VOCAB_SIZE=10240 TRIGRAM_VOCAB_SIZE=4096 \
MUON_WEIGHT_DECAY=0.04 MUON_MOMENTUM=0.99 MATRIX_LR=0.02 \
ROPE_DIMS=16 SWA_ENABLED=1 EVAL_STRIDE=64 \
QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6 \
python3 train_gpt_mlx.py
```

### Competition (8xH100 SXM)
```bash
cd /workspace && git clone https://github.com/tomasi001/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

RUN_ID=submission \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## About Lappie AI

Lappie AI is a Cape Town-based AI startup building intelligent orchestration systems. This competition entry demonstrates that frontier AI research — deep transformer architecture design, training optimisation, model compression — can originate from anywhere with the right tools, knowledge, and determination.

**Website**: [lappie.ai](https://lappie.ai)
**GitHub**: [@lappiecto](https://github.com/lappiecto)
