"""
N-gram hash embedding modules for Parameter Golf.

Ported from MLX (train_gpt_mlx.py) to PyTorch.
Drop-in modules: SmearGate, BigramHashEmbedding, TrigramHashEmbedding.

Integration: see INTEGRATION section at the bottom of this file.
"""

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Import CastedLinear from the main training script so projections stay
# consistent (fp32 weights, bf16 compute).  When running standalone tests
# you can fall back to nn.Linear.
# ---------------------------------------------------------------------------
try:
    from train_gpt import CastedLinear
except ImportError:
    # Fallback for standalone testing — mirrors the real CastedLinear.
    import torch.nn.functional as F

    class CastedLinear(nn.Linear):  # type: ignore[no-redef]
        def forward(self, x: Tensor) -> Tensor:
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, self.weight.to(x.dtype), bias)


# ---------------------------------------------------------------------------
# SmearGate
# ---------------------------------------------------------------------------

class SmearGate(nn.Module):
    """Blend each token's embedding with the previous token's embedding.

    A learned per-dimension gate (initialised at sigmoid(0) = 0.5) controls
    the mix ratio.  This gives the model bigram-level disambiguation before
    the first transformer layer.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Logit space — sigmoid(0) = 0.5, so equal blend at init.
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        # gate shape: (dim,) -> (1, 1, dim) for broadcasting over (B, T, D)
        g = torch.sigmoid(self.gate.to(x.dtype))[None, None, :]
        # Shift right by 1: first position gets a zero vector as "previous".
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


# ---------------------------------------------------------------------------
# BigramHashEmbedding
# ---------------------------------------------------------------------------

class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table.

    Each (prev_token, current_token) pair is mapped via a deterministic
    XOR-based hash to one of ``bigram_vocab_size`` buckets.  The embedding
    starts at zero and is scaled by a small learned factor (0.05) so the
    bigram signal ramps up gradually.
    """

    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        # Zero-init: bigram signal starts silent and grows during training.
        nn.init.zeros_(self.embed.weight)

        self.proj: nn.Module | None = (
            CastedLinear(bigram_dim, model_dim, bias=False)
            if bigram_dim != model_dim
            else None
        )
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)

        # Learned scale — small so the signal doesn't dominate early.
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        """Map each (prev, current) pair to a bucket index."""
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        # XOR hash with coprime constants 36313 and 27191.
        hashed = (36313 * t[..., 1:] ^ 27191 * t[..., :-1]) % mod
        # First position has no predecessor — use the last bucket as sentinel.
        first = torch.full(
            t[..., :1].shape, mod, dtype=torch.int32, device=tokens.device
        )
        return torch.cat([first, hashed], dim=-1)

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(h.dtype)


# ---------------------------------------------------------------------------
# TrigramHashEmbedding
# ---------------------------------------------------------------------------

class TrigramHashEmbedding(nn.Module):
    """Hash consecutive token TRIPLES into a learned embedding table.

    Extends BigramHash to 3-token windows using different coprime constants
    (48271, 31547, 17389) to avoid correlation.  Scale is 0.03 — smaller
    than bigram because trigram features are sparser.
    """

    def __init__(self, trigram_vocab_size: int, trigram_dim: int, model_dim: int):
        super().__init__()
        self.trigram_vocab_size = trigram_vocab_size
        self.embed = nn.Embedding(trigram_vocab_size, trigram_dim)
        nn.init.zeros_(self.embed.weight)

        self.proj: nn.Module | None = (
            CastedLinear(trigram_dim, model_dim, bias=False)
            if trigram_dim != model_dim
            else None
        )
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)

        self.scale = nn.Parameter(torch.tensor(0.03, dtype=torch.float32))

    def trigram_hash(self, tokens: Tensor) -> Tensor:
        """Map each (t-2, t-1, t) triple to a bucket index."""
        t = tokens.to(torch.int32)
        mod = self.trigram_vocab_size - 1
        hashed = (
            48271 * t[..., 2:] ^ 31547 * t[..., 1:-1] ^ 17389 * t[..., :-2]
        ) % mod
        # First two positions use sentinel bucket.
        sentinel = torch.full(
            t[..., :2].shape, mod, dtype=torch.int32, device=tokens.device
        )
        return torch.cat([sentinel, hashed], dim=-1)

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.trigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(h.dtype)


# ===================================================================
# INTEGRATION INSTRUCTIONS
# ===================================================================
#
# How to wire these into the GPT class in train_gpt.py:
#
# 1. IMPORTS — at the top of train_gpt.py add:
#
#        from modules_ngram import SmearGate, BigramHashEmbedding, TrigramHashEmbedding
#
# 2. GPT.__init__ — after self.tok_emb, add:
#
#        # --- n-gram modules ---
#        self.smear_gate = SmearGate(model_dim)
#        self.bigram_embed = BigramHashEmbedding(
#            bigram_vocab_size=8191,   # prime, tune to taste
#            bigram_dim=64,            # small intermediate dim
#            model_dim=model_dim,
#        )
#        self.trigram_embed = TrigramHashEmbedding(
#            trigram_vocab_size=16381,  # prime
#            trigram_dim=48,
#            model_dim=model_dim,
#        )
#
# 3. GPT.forward — replace the embedding section (lines ~700-703):
#
#    BEFORE:
#        x = self.tok_emb(input_ids)
#        x = F.rms_norm(x, (x.size(-1),))
#        x0 = x
#
#    AFTER:
#        x = self.tok_emb(input_ids)
#        x = x + self.bigram_embed(input_ids)
#        x = x + self.trigram_embed(input_ids)
#        x = self.smear_gate(x)
#        x = F.rms_norm(x, (x.size(-1),))
#        x0 = x
#
# 4. OPTIMIZER GROUPS — the scale and gate parameters are 1-D scalars /
#    vectors, so they'll automatically land in the "scalar_params" group
#    (ndim < 2) in the existing optimizer setup.  The embedding weights
#    (ndim == 2) go into the matrix group.  No changes needed unless you
#    want separate learning rates for n-gram params.
#
# 5. CONTROL_TENSOR_NAME_PATTERNS — if you want the gate and scale params
#    kept in fp32 by restore_low_dim_params_to_fp32(), add to the env var
#    or the default string:
#
#        "smear_gate.gate,bigram_embed.scale,trigram_embed.scale"
#
#    (They're already ndim < 2 so they'll be kept in fp32 by the existing
#    ndim check, but adding them explicitly is belt-and-braces.)
#
# 6. PARAMETER BUDGET — approximate additional params:
#        SmearGate:          model_dim                  (e.g. 768)
#        BigramHashEmbed:    8191*64 + 64*model_dim     (e.g. ~573K)
#        TrigramHashEmbed:   16381*48 + 48*model_dim    (e.g. ~823K)
#        Total:              ~1.4M extra params
#
#    Tune vocab sizes and dims to stay within your 16MB parameter budget.
#    Halving bigram_dim to 32 and trigram_dim to 24 roughly halves the cost.
# ===================================================================
