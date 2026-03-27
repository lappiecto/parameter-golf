#!/usr/bin/env python3
"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# SHARD FORMAT + COMPUTE DTYPE
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap
class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop. These defaults now mirror train_gpt.py on a single process.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    # Validation always uses the full fineweb_val split.
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    # Chunk each logical MLX microbatch into smaller sub-batches to reduce peak
    # memory pressure without changing the effective optimizer batch.
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    # Force MLX to materialize the graph after every sub-batch, preventing lazy
    # graph buildup across accumulation steps. Keeps peak memory low on 16GB machines.
    # Disable on 32GB+ unified memory for better throughput (MLX_EAGER_EVAL=0).
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model (defaults match the current baseline setup).
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims: int = int(os.environ.get("ROPE_DIMS", 16))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # SmearGate + BigramHash
    use_smeargate: bool = bool(int(os.environ.get("USE_SMEARGATE", "1")))
    bigram_vocab_size: int = int(os.environ.get("BIGRAM_VOCAB_SIZE", 4096))
    bigram_dim: int = int(os.environ.get("BIGRAM_DIM", 128))

    # Muon weight decay
    muon_weight_decay: float = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))

    # TrigramHash
    trigram_vocab_size: int = int(os.environ.get("TRIGRAM_VOCAB_SIZE", 0))
    trigram_dim: int = int(os.environ.get("TRIGRAM_DIM", 64))

    # EMA weight averaging
    ema_enabled: bool = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.997))

    # Quantisation precision
    quant_bits_mlp: int = int(os.environ.get("QUANT_BITS_MLP", 6))
    quant_bits_attn: int = int(os.environ.get("QUANT_BITS_ATTN", 6))

    # Test-Time Training
    ttt_enabled: bool = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr: float = float(os.environ.get("TTT_LR", 1.0))
    ttt_momentum: float = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_epochs: int = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_grad_clip: float = float(os.environ.get("TTT_GRAD_CLIP", 1.0))

    # Sliding window eval
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", 64))

    # Stochastic Weight Averaging
    swa_enabled: bool = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_start_frac: float = float(os.environ.get("SWA_START_FRAC", 0.4))
    swa_every: int = int(os.environ.get("SWA_EVERY", 50))

    # Optimizer. We keep the same per-group defaults as train_gpt.py.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        # Iteration-based warmdown: ramp down over the last warmdown_iters steps
        iter_scale = 1.0
        warmdown_start = max(self.iterations - self.warmdown_iters, 0)
        if warmdown_start <= step < self.iterations:
            iter_scale = max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0)
        elif step >= self.iterations:
            iter_scale = 0.0
        if self.max_wallclock_seconds <= 0:
            return iter_scale
        # Wallclock-based warmdown: ramp down over the last warmdown_iters-worth of time
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        wall_scale = remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
        # Use whichever warmdown is further along (lower value), so SWA triggers
        # regardless of whether the run is iteration-limited or wallclock-limited
        return min(iter_scale, wall_scale)


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    # Background on Muon: https://kellerjordan.github.io/posts/muon/
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    # Attention with: Partial RoPE, VRL (Value Residual Learning), Gated Attention
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 16,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rope_dims = min(rope_dims, self.head_dim)
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        # Partial RoPE: only rotate rope_dims of the head_dim dimensions.
        # The rest are position-invariant (content-only matching).
        self.rope = nn.RoPE(self.rope_dims, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5
        # Gated Attention: per-head sigmoid gate controls each head's contribution.
        # Init at 0 → sigmoid(0) = 0.5 — each head starts at half contribution.
        self.head_gate = mx.zeros((num_heads,), dtype=mx.float32)
        # VRL: learned blend gate for mixing layer-0's V into this layer's V.
        # Init at 0 → sigmoid(0) = 0.5 blend.
        self.vrl_alpha = mx.array(0.0, dtype=mx.float32)

    def __call__(self, x: mx.array, v0: mx.array | None = None) -> tuple[mx.array, mx.array]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # VRL: mix layer-0's V into this layer's V
        if v0 is not None:
            vrl_gate = mx.sigmoid(self.vrl_alpha.astype(v.dtype))
            v = (1.0 - vrl_gate) * v + vrl_gate * v0

        q = rms_norm(q).astype(COMPUTE_DTYPE)
        k = rms_norm(k).astype(COMPUTE_DTYPE)

        # Partial RoPE: apply to first rope_dims only
        rd = self.rope_dims
        q = mx.concatenate([self.rope(q[..., :rd]), q[..., rd:]], axis=-1)
        k = mx.concatenate([self.rope(k[..., :rd]), k[..., rd:]], axis=-1)

        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")

        # Gated Attention: per-head sigmoid gate
        gate = mx.sigmoid(self.head_gate.astype(y.dtype))[None, :, None, None]
        y = y * gate

        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y), v  # return v for VRL caching


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        # LeakyReLU(0.9)² instead of relu². Negative pre-activations pass through
        # at 90% strength then get squared (becoming positive). This preserves
        # gradient flow for all neurons, not just positive ones. The squaring
        # still creates sparsity-like behaviour (small values get crushed) while
        # the leaky negative path prevents dead neurons entirely.
        # The #1 submission gained 0.002 BPB from this single change.
        x = self.fc(x)
        x = mx.where(x >= 0, x, 0.9 * x)  # leaky_relu with slope 0.9 (discovered Mar 25 — 0.013 BPB over 0.5)
        return self.proj(x * x)


class SmearGate(nn.Module):
    """Blend each token's embedding with the previous token's embedding.

    The intuition: a token by itself is ambiguous ("bank" could be a river bank
    or a financial bank). By mixing in the previous token's embedding, the model
    gets bigram-level disambiguation *before* it even enters the transformer
    layers. The gate is learned per-dimension — so the model figures out which
    features benefit from looking backward and which don't.

    Initialised at sigmoid(0) = 0.5, meaning a 50/50 blend to start. During
    training, each dimension learns its own mixing ratio.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        g = mx.sigmoid(self.gate.astype(x.dtype))[None, None, :]
        # Shift x right by 1 position, padding with zeros for the first token
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table.

    Standard token embeddings treat each token independently. But in natural
    language, pairs of tokens carry meaning that neither token has alone
    ("New" + "York" vs "New" + "car"). This module hashes each (prev, current)
    token pair into one of `bigram_vocab_size` buckets and looks up a learned
    embedding. The hash is deterministic — the same pair always maps to the
    same bucket — but different pairs can collide (like a hash table).

    The embedding starts at zero and is scaled by a small learned factor (0.05),
    so the bigram signal ramps up gradually during training without disrupting
    the main token embeddings early on.
    """
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        # Zero-init: bigram signal starts silent and grows during training
        self.embed.weight = mx.zeros_like(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim) if bigram_dim != model_dim else None
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array(0.05, dtype=mx.float32)

    def bigram_hash(self, tokens: mx.array) -> mx.array:
        t = tokens.astype(mx.int32)
        mod = self.bigram_vocab_size - 1
        # XOR-based hash: maps each (prev_token, current_token) pair to a bucket.
        # The magic numbers 36313 and 27191 are coprime constants that spread
        # the hash values evenly across the bucket space.
        hashed = (36313 * t[..., 1:] ^ 27191 * t[..., :-1]) % mod
        # First position has no predecessor — use a sentinel bucket
        first = mx.full(t[..., :1].shape, mod, dtype=mx.int32)
        return mx.concatenate([first, hashed], axis=-1)

    def __call__(self, token_ids: mx.array) -> mx.array:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.astype(h.dtype)


class TrigramHashEmbedding(nn.Module):
    """Hash consecutive token TRIPLES into a learned embedding table.

    Extends BigramHash to 3-token sequences. Where BigramHash captures "New York",
    TrigramHash captures "New York City". The hash space is larger but the table
    is fixed size — we rely on the hash function to distribute triples evenly.

    Uses a different set of coprime constants to avoid correlation with BigramHash.
    Both modules can run simultaneously, giving the model both pair and triple
    level features before the first attention layer.
    """
    def __init__(self, trigram_vocab_size: int, trigram_dim: int, model_dim: int):
        super().__init__()
        self.trigram_vocab_size = trigram_vocab_size
        self.embed = nn.Embedding(trigram_vocab_size, trigram_dim)
        self.embed.weight = mx.zeros_like(self.embed.weight)
        self.proj = CastedLinear(trigram_dim, model_dim) if trigram_dim != model_dim else None
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array(0.03, dtype=mx.float32)

    def trigram_hash(self, tokens: mx.array) -> mx.array:
        t = tokens.astype(mx.int32)
        mod = self.trigram_vocab_size - 1
        # Hash of 3 consecutive tokens using different primes than BigramHash
        hashed = (48271 * t[..., 2:] ^ 31547 * t[..., 1:-1] ^ 17389 * t[..., :-2]) % mod
        # First two positions use sentinel
        sentinel = mx.full(t[..., :2].shape, mod, dtype=mx.int32)
        return mx.concatenate([sentinel, hashed], axis=-1)

    def __call__(self, token_ids: mx.array) -> mx.array:
        h = self.embed(self.trigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.astype(h.dtype)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
        rope_dims: int = 16,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))
        # LN Scale: deeper layers contribute less, stabilising gradient flow
        self._ln_scale = 1.0 / math.sqrt(layer_idx + 1)

    def __call__(self, x: mx.array, x0: mx.array, v0: mx.array | None = None) -> tuple[mx.array, mx.array | None]:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, v_out = self.attn(self.attn_norm(x), v0=v0)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out * self._ln_scale
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x)) * self._ln_scale
        return x, v_out


class GPT(nn.Module):
    # - token embedding + RMSNorm
    # - encoder half accumulates skip tensors
    # - decoder half consumes reversed skips with learned skip_weights
    # - tied embeddings for the LM head (the baseline default setup)
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, use_smeargate: bool = False, bigram_vocab_size: int = 0, bigram_dim: int = 128,
                 trigram_vocab_size: int = 0, trigram_dim: int = 64,
                 rope_dims: int = 16):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, dim) if bigram_vocab_size > 0 else None
        self.trigram = TrigramHashEmbedding(trigram_vocab_size, trigram_dim, dim) if trigram_vocab_size > 0 else None
        self.smear = SmearGate(dim) if use_smeargate else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, rope_dims=rope_dims)
            for i in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        # Orthogonal init: produces weight matrices whose singular values are all 1.
        # This gives gradients a uniform highway through the network from the very
        # first step — no vanishing/exploding signal. It matters especially when
        # you only get a few thousand training steps.
        for b in self.blocks:
            for linear in [b.attn.c_q, b.attn.c_k, b.attn.c_v, b.mlp.fc]:
                w_np = np.random.randn(*linear.weight.shape).astype(np.float32)
                q, _ = np.linalg.qr(w_np if w_np.shape[0] >= w_np.shape[1] else w_np.T)
                q = q if w_np.shape[0] >= w_np.shape[1] else q.T
                linear.weight = mx.array(q[:linear.weight.shape[0], :linear.weight.shape[1]], dtype=mx.float32)
            # Output projections start at zero — the residual stream is the identity at init
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        # BigramHash and TrigramHash add n-gram features to token embeddings
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        if self.trigram is not None:
            x = x + self.trigram(input_ids)
        x = rms_norm(x)
        # SmearGate blends each token with its predecessor
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        skips: list[mx.array] = []
        v0_cached: mx.array | None = None  # VRL: cache layer 0's V output

        for i in range(self.num_encoder_layers):
            x, v_out = self.blocks[i](x, x0, v0=v0_cached if i > 0 else None)
            if i == 0:
                v0_cached = v_out
            skips.append(x)
        for i in range(self.num_decoder_layers):
            # Odd layer counts have one more decoder block than encoder block. The baseline only
            # applies a skip connection when one exists, then runs the remaining decoder block(s)
            # without an added skip.
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x, _ = self.blocks[self.num_encoder_layers + i](x, x0, v0=v0_cached)
        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        # Cross-entropy over flattened tokens. We keep optional logit chunking because it is a useful
        # memory knob on Macs, but the common path is chunk_tokens=0 (single matmul + CE).
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)

# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        wd = self.args.muon_weight_decay
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            # Cautious Muon: mask out momentum updates that conflict with gradient direction.
            # If momentum points opposite to current gradient, zero it out.
            buf_masked = buf * (buf * g > 0).astype(buf.dtype)
            g_eff = g + momentum * buf_masked
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            # Decoupled weight decay: shrink weights toward zero before the
            # gradient update. This keeps the weight distribution tight and
            # well-centred, which improves both generalisation and post-training
            # quantisation (tighter distributions compress better).
            if wd > 0:
                p = p * (1.0 - wd * lr)
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    # - embeddings: Adam with the tied-embedding LR
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))

        # Embedding keys — tok_emb + bigram embedding (both get embed LR via Adam)
        self.embed_keys = ["tok_emb.weight"]
        if model.bigram is not None:
            self.embed_keys.append("bigram.embed.weight")
        if model.trigram is not None:
            self.embed_keys.append("trigram.embed.weight")

        # Matrix keys — all 2D block weights + bigram projection (Muon)
        self.matrix_keys = [
            k
            for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2 and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if model.bigram is not None and model.bigram.proj is not None:
            self.matrix_keys.append("bigram.proj.weight")
        if model.trigram is not None and model.trigram.proj is not None:
            self.matrix_keys.append("trigram.proj.weight")

        # Scalar keys — skip weights, block scalars, smear gate, bigram scale
        self.scalar_keys = [
            k
            for k, p in params.items()
            if k == "skip_weights" or (k.startswith("blocks.") and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
        ]
        if model.smear is not None:
            self.scalar_keys.append("smear.gate")
        if model.bigram is not None:
            self.scalar_keys.append("bigram.scale")
        if model.trigram is not None:
            self.scalar_keys.append("trigram.scale")

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        embed_grads = {k: grads[k] for k in self.embed_keys if k in grads}
        embed_params = {k: params[k] for k in self.embed_keys if k in params}
        if embed_grads:
            updated.update(self.adam_embed.apply_gradients(embed_grads, embed_params))

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys if k in grads}
        scalar_params = {k: params[k] for k in self.scalar_keys if k in params}
        if scalar_grads:
            updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB)
# ==============================================================================
# - per-row int8 for 2D float tensors
# - per-tensor int8 for other float tensors
# - fp16 passthrough for small float tensors
# - exact passthrough for non-floats

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr: mx.array, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Quantise a float array to `bits`-bit signed integer range.

    Int8 uses [-127, 127], int6 uses [-31, 31], int5 uses [-15, 15].
    Fewer bits = smaller compressed artifact (zlib loves smaller value ranges),
    but more quantisation error. MLP weights tolerate fewer bits well because
    relu² creates sparse activations that shape weight distributions tightly.
    Attention weights need more precision because they control the sharp
    attention patterns that matter for long-range dependencies.
    """
    max_val = (1 << (bits - 1)) - 1  # int8=127, int6=31, int5=15
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / max_val, 1.0 / max_val).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -max_val, max_val).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / max_val if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -max_val, max_val).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(flat_state: dict[str, mx.array], bits_mlp: int = 8, bits_attn: int = 8) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        # Mixed-precision quantisation: MLP weights get fewer bits (they compress
        # better because relu² creates sparse activations), attention weights get
        # more bits (precision-sensitive for sharp attention patterns).
        bits = 8  # default
        if ".mlp." in name and arr.ndim == 2:
            bits = bits_mlp
        elif ".attn." in name and arr.ndim == 2:
            bits = bits_attn
        elif "bigram.proj" in name and arr.ndim == 2:
            bits = bits_attn  # treat bigram proj like attention

        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr, bits=bits)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0, "bits": bits}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            # Broadcast the saved row scale back across trailing dimensions.
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    # The shard directory and tokenizer are coupled: val_bpb is only meaningful if we
    # decode bytes with the exact tokenizer that produced the shards. The manifest
    # lets the training script fail fast on accidental dataset/tokenizer mismatches.
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)  # materialize each chunk to cap peak memory
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(
    args: Hyperparameters,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb

def eval_val_sliding(
    args: Hyperparameters,
    model: GPT,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    stride: int,
    log_fn: Callable[[str], None] | None = None,
    # Legacy kwargs kept for call-site compatibility — ignored.
    use_ngram_cache: bool = False,
    ngram_max_order: int = 7,
    ngram_base_alpha: float = 0.3,
) -> tuple[float, float]:
    """Sliding window evaluation — each token scored with near-maximum context.

    Windows of `train_seq_len` advance by `stride`. Only the last `stride`
    tokens per window contribute to the score (the first window scores all
    its tokens). This matches the competition's standard sliding-window eval
    used by every record submission.

    The previous n-gram backoff cache has been removed. It was hurting BPB
    (1.6209 -> 1.6895) because of:
      - No document boundary resets (cross-document pollution)
      - Overly aggressive mixing alpha (0.3-0.8 vs the <0.05 used in LM lit)
      - Token-by-token Python loop (100x slower than vectorised)
      - Cache updates using only targets, not full context stream

    The correct eval-time BPB improvements in this competition are:
      - Sliding window (this function): ~0.03 BPB
      - TTT (eval_val_sliding_ttt): ~0.05-0.10 BPB
    """
    seq_len = args.train_seq_len
    total_tokens = val_tokens.size - 1

    # Build window start positions — include final partial window if >= 1 token
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]

    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0
    total_windows = len(window_starts)

    for wi, ws in enumerate(window_starts):
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws

        chunk = val_tokens[ws:end + 1]
        x_np = chunk[:-1].reshape(1, -1)
        y_np = chunk[1:].reshape(1, -1)

        # Pad to seq_len if needed
        if wlen < seq_len:
            x_padded = np.zeros((1, seq_len), dtype=np.int32)
            y_padded = np.zeros((1, seq_len), dtype=np.int32)
            x_padded[0, :wlen] = x_np[0]
            y_padded[0, :wlen] = y_np[0]
            x_np = x_padded
            y_np = y_padded

        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)

        # Forward pass — get logits via tied embedding projection + softcap
        hidden = model(x)  # (1, seq_len, dim)
        logits_proj = hidden.reshape(-1, model.tok_emb.weight.shape[1]) @ model.tok_emb.weight.astype(hidden.dtype).T
        logits_capped = model.softcap(logits_proj)

        # Vectorised cross-entropy: no per-token Python loop
        per_token_loss = nn.losses.cross_entropy(
            logits_capped.astype(mx.float32),
            y.reshape(-1),
            reduction="none",
        )
        mx.eval(per_token_loss)
        nll = np.array(per_token_loss, dtype=np.float64)

        # Score only the last `stride` tokens (or all for first window)
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_nll = nll[s:wlen]

        loss_sum += float(scored_nll.sum())
        token_count += float(wlen - s)

        # Byte counting for BPB
        y_flat = y_np.reshape(-1)
        x_flat = x_np.reshape(-1)
        tgt = y_flat[s:wlen]
        prev = x_flat[s:wlen]
        tb = base_bytes_lut[tgt].astype(np.float64)
        tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).astype(np.float64)
        byte_count += float(tb.sum())

        if log_fn is not None and (wi % 200 == 0 or wi == total_windows - 1):
            running_bpb = 0.0
            if token_count > 0:
                rl = loss_sum / token_count
                running_bpb = rl / math.log(2.0) * (token_count / byte_count)
            log_fn(f"val_progress:{wi + 1}/{total_windows} running_bpb:{running_bpb:.6f}")

    val_loss = loss_sum / token_count
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (token_count / byte_count)
    return val_loss, val_bpb


# -----------------------------
# TEST-TIME TRAINING (Score-First Backward-Looking TTT)
# -----------------------------
# At eval time we adapt ALL model parameters per-document on already-scored
# tokens.  The approach:
#   1.  Slide a window (stride-64) across each document in the validation set.
#   2.  For each window, score the last `stride` tokens under inference (no grad).
#   3.  After scoring, train ALL params on those same tokens with SGD (momentum=0.9,
#       LR=1.0, 3 epochs per chunk, cosine decay within each document, grad clip 1.0).
#   4.  Reset model to the quantised checkpoint between documents.

BOS_ID = 1  # SentencePiece BOS token id


def _find_doc_boundaries(tokens: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) for each document delimited by BOS tokens.

    `end` is exclusive — tokens[start:end] is the full document including its
    leading BOS.  The final document extends to the end of the array.
    """
    bos_positions = np.where(tokens == BOS_ID)[0]
    if len(bos_positions) == 0:
        return [(0, len(tokens))]
    docs: list[tuple[int, int]] = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else len(tokens)
        if end - start >= 2:
            docs.append((start, end))
    return docs


def _sgd_momentum_step(
    params_flat: dict[str, mx.array],
    grads_flat: dict[str, mx.array],
    velocity: dict[str, mx.array],
    lr: float,
    momentum: float = 0.9,
) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """One step of SGD with momentum.  Returns (updated_params, updated_velocity)."""
    new_params: dict[str, mx.array] = {}
    new_velocity: dict[str, mx.array] = {}
    for k in params_flat:
        g = grads_flat.get(k)
        if g is None:
            new_params[k] = params_flat[k]
            new_velocity[k] = velocity.get(k, mx.zeros_like(params_flat[k]))
            continue
        v = velocity.get(k, mx.zeros_like(g))
        v_new = momentum * v + g
        new_params[k] = params_flat[k] - lr * v_new
        new_velocity[k] = v_new
    return new_params, new_velocity


def _clip_grads_flat(grads_flat: dict[str, mx.array], max_norm: float) -> dict[str, mx.array]:
    """Clip gradient tree (already flattened) by global norm."""
    if max_norm <= 0:
        return grads_flat
    sq_sum = mx.array(0.0, dtype=mx.float32)
    for g in grads_flat.values():
        sq_sum = sq_sum + mx.sum(g.astype(mx.float32) ** 2)
    mx.eval(sq_sum)
    total_norm = float(sq_sum.item()) ** 0.5
    if total_norm <= max_norm:
        return grads_flat
    scale = max_norm / (total_norm + 1e-12)
    return {k: g * scale for k, g in grads_flat.items()}


def eval_val_sliding_ttt(
    args: Hyperparameters,
    model: GPT,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    stride: int,
    ttt_lr: float = 1.0,
    ttt_momentum: float = 0.9,
    ttt_epochs: int = 3,
    ttt_grad_clip: float = 1.0,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    """Sliding window eval with per-document backward-looking test-time training.

    For each document in the validation set:
      1. Save model weights (the quantised checkpoint).
      2. Slide a window across the document, scoring the last `stride` tokens
         (inference only — this is the BPB contribution).
      3. After scoring each chunk, train ALL model params on those scored tokens
         using SGD with momentum for `ttt_epochs` passes.
      4. Cosine LR decay within each document (chunk index / total chunks).
      5. After the document ends, restore model weights from the saved checkpoint.

    Returns (val_loss, val_bpb) — the same signature as eval_val_sliding.
    """
    seq_len = args.train_seq_len

    # -------------------------------------------------------------------------
    # 1. Identify document boundaries
    # -------------------------------------------------------------------------
    docs = _find_doc_boundaries(val_tokens)
    if log_fn is not None:
        log_fn(f"ttt:found {len(docs)} documents in validation set")

    # -------------------------------------------------------------------------
    # 2. Snapshot the base model weights (we restore after each document)
    # -------------------------------------------------------------------------
    base_state = [(k, mx.array(v)) for k, v in tree_flatten(model.state)]
    mx.eval(*[v for _, v in base_state])

    # -------------------------------------------------------------------------
    # 3. Build the loss+grad function for TTT training steps
    # -------------------------------------------------------------------------
    def _ttt_loss(model_ref: GPT, x: mx.array, y: mx.array) -> mx.array:
        logits = model_ref(x)
        logits_proj = (
            logits.reshape(-1, model_ref.tok_emb.weight.shape[1])
            @ model_ref.tok_emb.weight.astype(logits.dtype).T
        )
        logits_capped = model_ref.softcap(logits_proj)
        return nn.losses.cross_entropy(
            logits_capped.astype(mx.float32), y.reshape(-1), reduction="mean"
        )

    loss_and_grad_fn = nn.value_and_grad(model, lambda x, y: _ttt_loss(model, x, y))

    # -------------------------------------------------------------------------
    # 4. Main loop: iterate over documents
    # -------------------------------------------------------------------------
    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0

    for doc_idx, (doc_start, doc_end) in enumerate(docs):
        doc_tokens = val_tokens[doc_start:doc_end]
        doc_len = len(doc_tokens) - 1  # number of prediction positions

        if doc_len < 1:
            continue

        # Restore model to base checkpoint before each document
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in base_state]))
        mx.eval(*[v for _, v in tree_flatten(model.state)])

        # Initialise SGD momentum buffers (fresh per document)
        velocity: dict[str, mx.array] = {}

        # Build window start positions for this document
        window_starts: list[int] = []
        for ws in range(0, doc_len, stride):
            if min(ws + seq_len, doc_len) - ws >= 1:
                window_starts.append(ws)
        total_chunks = len(window_starts)

        for ci, ws in enumerate(window_starts):
            end = min(ws + seq_len, doc_len)
            wlen = end - ws

            # Slice input/target from the document
            chunk = doc_tokens[ws:end + 1]
            x_np = chunk[:-1].reshape(1, -1)
            y_np = chunk[1:].reshape(1, -1)

            # Pad to seq_len if needed
            if wlen < seq_len:
                x_padded = np.zeros((1, seq_len), dtype=np.int32)
                y_padded = np.zeros((1, seq_len), dtype=np.int32)
                x_padded[0, :wlen] = x_np[0]
                y_padded[0, :wlen] = y_np[0]
                x_np = x_padded
                y_np = y_padded

            x = mx.array(x_np, dtype=mx.int32)
            y = mx.array(y_np, dtype=mx.int32)

            # -----------------------------------------------------------------
            # SCORE: inference-only forward pass — accumulate BPB
            # -----------------------------------------------------------------
            logits = model(x)
            logits_proj = (
                logits.reshape(-1, model.tok_emb.weight.shape[1])
                @ model.tok_emb.weight.astype(logits.dtype).T
            )
            logits_capped = model.softcap(logits_proj)
            per_token_loss = nn.losses.cross_entropy(
                logits_capped.astype(mx.float32),
                y.reshape(-1),
                reduction="none",
            )
            mx.eval(per_token_loss)
            nll = np.array(per_token_loss, dtype=np.float64)

            # Score only the last `stride` tokens (or all for the first window)
            s = 0 if ws == 0 else max(wlen - stride, 0)
            scored_nll = nll[s:wlen]
            loss_sum += float(scored_nll.sum())
            token_count += float(wlen - s)

            tgt = y_np.reshape(-1)[s:wlen]
            prev = x_np.reshape(-1)[s:wlen]
            tb = base_bytes_lut[tgt].astype(np.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).astype(np.float64)
            byte_count += float(tb.sum())

            # -----------------------------------------------------------------
            # TRAIN: backward-looking SGD on the already-scored chunk
            # -----------------------------------------------------------------
            # Cosine LR decay within this document
            progress = ci / max(total_chunks - 1, 1)
            cos_lr = ttt_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

            for _epoch in range(ttt_epochs):
                loss_val, grads = loss_and_grad_fn(x, y)
                mx.eval(loss_val)

                # Flatten grads, clip, then SGD step
                grads_flat = dict(tree_flatten(grads))
                grads_flat = _clip_grads_flat(grads_flat, ttt_grad_clip)

                params_flat = dict(tree_flatten(model.parameters()))
                params_flat, velocity = _sgd_momentum_step(
                    params_flat, grads_flat, velocity, lr=cos_lr, momentum=ttt_momentum
                )
                model.update(tree_unflatten(list(params_flat.items())))
                mx.eval(*[v for _, v in tree_flatten(model.state)])

        if log_fn is not None and (doc_idx % 50 == 0 or doc_idx == len(docs) - 1):
            running_loss = loss_sum / max(token_count, 1)
            running_bpb = (running_loss / math.log(2.0)) * (token_count / max(byte_count, 1))
            log_fn(
                f"ttt_progress:doc {doc_idx + 1}/{len(docs)} "
                f"running_loss:{running_loss:.4f} running_bpb:{running_bpb:.4f}"
            )

    # -------------------------------------------------------------------------
    # 5. Restore model to base state (leave it unchanged from caller's POV)
    # -------------------------------------------------------------------------
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in base_state]))
    mx.eval(*[v for _, v in tree_flatten(model.state)])

    val_loss = loss_sum / max(token_count, 1)
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (token_count / max(byte_count, 1))
    return val_loss, val_bpb


# -----------------------------
# TRAINING
# -----------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(_np_float32(grad)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def main() -> None:
    # ==============================================================================
    # TOKENIZER + VALIDATION METRIC SETUP
    # ==============================================================================
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tie_embeddings:
        raise NotImplementedError("train_gpt_mlx.py only supports tied embeddings")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # ==============================================================================
    # TRAINING SETUP
    # ==============================================================================
    mx.random.seed(args.seed)

    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # ==============================================================================
    # MODEL + OPTIMIZER SETUP
    # ==============================================================================
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        use_smeargate=args.use_smeargate,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        trigram_vocab_size=args.trigram_vocab_size,
        trigram_dim=args.trigram_dim,
        rope_dims=args.rope_dims,
    )
    opt = SplitOptimizers(model, args)

    # ==============================================================================
    # COMPILED TRAIN / EVAL FUNCTIONS (MLX)
    # ==============================================================================
    # The crucial MLX detail is capture scope: this model contains non-trainable arrays too (for example
    # inside RoPE modules), so compiling only against trainable parameters throws "uncaptured inputs".
    # Compiling the model-bound functions and capturing the full model state fixes that while still
    # returning gradients only for trainable parameters via nn.value_and_grad(...).
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )

    # Print config once so logs are self-describing.
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings} "
        f"smeargate:{args.use_smeargate} bigram:{args.bigram_vocab_size}x{args.bigram_dim} "
        f"muon_wd:{args.muon_weight_decay} ortho_init:True"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_weights.dtype}"
    )

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    if args.warmup_steps > 0:
        # Warmup should only prime MLX compile/allocation paths. Updating parameters here forces us
        # to snapshot and restore model/optimizer state, which is expensive on unified-memory Macs.
        # Instead we run the real train shapes, force the loss/grads to materialize, and then reset
        # the loader so measured training still starts from the true init and token window.
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime the standalone eval graph once too. It is compiled separately from value_and_grad.
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < args.train_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        warm_val_loss = compiled_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None

    # SWA state: accumulate model snapshots during warmdown
    swa_state: dict[str, np.ndarray] | None = None
    swa_count = 0

    # EMA state: exponential moving average of parameters (continuous smoothing)
    ema_state: dict[str, np.ndarray] | None = None
    if args.ema_enabled:
        ema_state = {k: np.array(v.astype(mx.float32), dtype=np.float32, copy=True) for k, v in tree_flatten(model.parameters())}
        log(f"ema:enabled decay:{args.ema_decay}")

    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Validation always scans the same fixed full validation split.
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
            )
            if step % 25 == 0 or last_step:
                log(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)  # materialize each microbatch to cap peak memory

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        # EMA: update exponential moving average every 10 steps (not every step —
        # the numpy conversion is expensive on MLX and was eating 70% of our wallclock)
        if ema_state is not None and step % 10 == 0:
            for k, v in tree_flatten(model.parameters()):
                v_np = np.array(v.astype(mx.float32), dtype=np.float32, copy=True)
                ema_state[k] = args.ema_decay * ema_state[k] + (1.0 - args.ema_decay) * v_np

        # SWA: collect checkpoint during warmdown
        lr_scale = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        if args.swa_enabled and lr_scale < args.swa_start_frac and step % args.swa_every == 0:
            # Only snapshot trainable parameters (floats), skip internal buffers
            flat_np = {}
            for k, v in tree_flatten(model.parameters()):
                flat_np[k] = np.array(v.astype(mx.float32), dtype=np.float32, copy=True)
            if swa_state is None:
                swa_state = flat_np
                swa_count = 1
                log(f"swa:start step:{step}")
            else:
                for k in swa_state:
                    if k in flat_np:
                        swa_state[k] = swa_state[k] + flat_np[k]
                swa_count += 1

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # ==============================================================================
    # APPLY EMA (if enabled) — load the smoothed weights before SWA/quantisation
    # ==============================================================================
    if ema_state is not None and not (args.swa_enabled and swa_state is not None and swa_count > 1):
        # Only apply EMA if SWA isn't going to overwrite it anyway
        log(f"ema:applying smoothed weights")
        current = dict(tree_flatten(model.parameters()))
        ema_params = {}
        for k, v in ema_state.items():
            if k in current:
                ema_params[k] = mx.array(v, dtype=current[k].dtype)
        model.update(tree_unflatten(list(ema_params.items())))
    elif ema_state is not None:
        log(f"ema:skipped (SWA will apply {swa_count} averaged checkpoints instead)")

    # ==============================================================================
    # APPLY SWA (if collected)
    # ==============================================================================
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log(f"swa:applying averaged {swa_count} checkpoints")
        avg_state = {}
        current = dict(tree_flatten(model.parameters()))
        for k, v in swa_state.items():
            avg_np = v / swa_count
            if k in current:
                avg_state[k] = mx.array(avg_np, dtype=current[k].dtype)
            else:
                avg_state[k] = mx.array(avg_np)
        model.update(tree_unflatten(list(avg_state.items())))
    elif args.swa_enabled:
        log(f"swa:skipped (collected {swa_count} checkpoints, need >1)")

    # ==============================================================================
    # FINAL SERIALIZATION + QUANTIZED ROUNDTRIP EVAL
    # ==============================================================================
    # We always write a raw artifact and a quantized artifact, then validate the
    # quantized roundtrip directly by loading the dequantized tensors back into the
    # model and running one final validation pass.
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats = quantize_state_dict_int8(flat_state, bits_mlp=args.quant_bits_mlp, bits_attn=args.quant_bits_attn)
    log(f"quantisation:mixed mlp_bits:{args.quant_bits_mlp} attn_bits:{args.quant_bits_attn}")
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    if _COMPRESSOR == "zstd":
        quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
    else:
        quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(
        f"serialized_model_int8_zlib:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    if _COMPRESSOR == "zstd":
        quant_flat = dequantize_state_dict_int8(pickle.loads(zstandard.ZstdDecompressor().decompress(quant_blob_disk)))
    else:
        quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))

    # Use the best eval strategy available
    q_t0 = time.perf_counter()
    if args.ttt_enabled and args.eval_stride > 0:
        # TTT + sliding window + n-gram cache — the full arsenal
        log(f"final_eval_mode:TTT+sliding_window stride:{args.eval_stride} ttt_lr:{args.ttt_lr} ttt_epochs:{args.ttt_epochs}")
        q_val_loss, q_val_bpb = eval_val_sliding_ttt(
            args, model, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, log_fn=log,
            ttt_lr=args.ttt_lr, ttt_momentum=args.ttt_momentum,
            ttt_epochs=args.ttt_epochs, ttt_grad_clip=args.ttt_grad_clip,
        )
    elif args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        # Sliding window eval (no TTT)
        log(f"final_eval_mode:sliding_window stride:{args.eval_stride}")
        q_val_loss, q_val_bpb = eval_val_sliding(
            args, model, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, log_fn=log,
        )
    else:
        q_val_loss, q_val_bpb = eval_val(
            args, compiled_loss, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            log_fn=log,
        )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
