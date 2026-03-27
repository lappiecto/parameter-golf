"""
Modified attention, block, and GPT forward pass — ported from MLX to PyTorch.

Additions over the baseline train_gpt.py:
  - CausalSelfAttention: Partial RoPE, Value Residual Learning (VRL), Gated Attention
  - Block: LN Scale (1/sqrt(layer_idx+1)), VRL passthrough, layer_idx param
  - GPT.forward: VRL caching (layer 0's v passed to all subsequent layers)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Assumed imports from the main train_gpt.py — these classes must exist in the
# module that imports us, or be injected before instantiation.
#   - CastedLinear (nn.Linear subclass that casts weight to input dtype)
#   - Rotary (caches cos/sin tables)
#   - apply_rotary_emb (half-split rotary application)
#   - RMSNorm (parameterless wrapper around F.rms_norm)
#   - MLP (relu^2 feedforward)
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head attention with Partial RoPE, VRL, and Gated Attention.

    Partial RoPE: only the first `rope_dims` of each head_dim get rotary
    embeddings. The remaining dimensions are position-invariant (content-only).

    Value Residual Learning (VRL): layer 0 caches its V tensor. All subsequent
    layers blend their own V with layer 0's V via a learned sigmoid gate.

    Gated Attention: a per-head learned sigmoid gate scales each head's output.
    Initialised at 0 → sigmoid(0) = 0.5, so every head starts at half strength.
    """

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
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))

        # Partial RoPE: only rotate first rope_dims of head_dim
        self.rotary = Rotary(self.rope_dims, base=rope_base)

        # Gated Attention: per-head sigmoid gate.
        # Init at 0 → sigmoid(0) = 0.5, each head starts at half contribution.
        self.head_gate = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))

        # VRL: learned blend gate for mixing layer-0's V into this layer's V.
        # Init at 0 → sigmoid(0) = 0.5 blend.
        self.vrl_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: Tensor, v0: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # VRL: mix layer-0's V into this layer's V
        if v0 is not None:
            vrl_gate = torch.sigmoid(self.vrl_alpha.to(dtype=v.dtype))
            v = (1.0 - vrl_gate) * v + vrl_gate * v0

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Partial RoPE: apply to first rope_dims only, leave the rest untouched
        rd = self.rope_dims
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q_rope = apply_rotary_emb(q[..., :rd], cos, sin)
        k_rope = apply_rotary_emb(k[..., :rd], cos, sin)
        q = torch.cat([q_rope, q[..., rd:]], dim=-1)
        k = torch.cat([k_rope, k[..., rd:]], dim=-1)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )

        # Gated Attention: per-head sigmoid gate
        gate = torch.sigmoid(self.head_gate.to(dtype=y.dtype))[None, :, None, None]
        y = y * gate

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y), v  # return v for VRL caching


class Block(nn.Module):
    """Transformer block with LN Scale and VRL passthrough.

    LN Scale: attention and MLP outputs are scaled by 1/sqrt(layer_idx + 1).
    Deeper layers contribute less to the residual, stabilising gradient flow.
    """

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
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        # LN Scale: deeper layers contribute less, stabilising gradient flow
        self._ln_scale = 1.0 / math.sqrt(layer_idx + 1)

    def forward(
        self, x: Tensor, x0: Tensor, v0: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, v_out = self.attn(self.attn_norm(x), v0=v0)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out * self._ln_scale
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x)) * self._ln_scale
        return x, v_out


def gpt_forward_with_vrl(
    model: nn.Module,
    input_ids: Tensor,
    target_ids: Tensor,
) -> Tensor:
    """GPT forward pass with VRL caching.

    Drop-in replacement for GPT.forward that:
      1. Caches layer 0's V output (v0_cached)
      2. Passes v0_cached to all subsequent layers
      3. Handles the updated Block signature (returns tuple)

    Expects `model` to have: tok_emb, blocks, num_encoder_layers,
    num_decoder_layers, skip_weights, final_norm, tie_embeddings,
    lm_head, logit_softcap.
    """
    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[Tensor] = []
    v0_cached: Optional[Tensor] = None  # VRL: cache layer 0's V output

    # Encoder half — accumulate skip tensors
    for i in range(model.num_encoder_layers):
        x, v_out = model.blocks[i](x, x0, v0=v0_cached if i > 0 else None)
        if i == 0:
            v0_cached = v_out
        skips.append(x)

    # Decoder half — consume reversed skips
    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x, _ = model.blocks[model.num_encoder_layers + i](x, x0, v0=v0_cached)

    x = model.final_norm(x).reshape(-1, x.size(-1))
    targets = target_ids.reshape(-1)

    if model.tie_embeddings:
        logits_proj = F.linear(x, model.tok_emb.weight)
    else:
        if model.lm_head is None:
            raise RuntimeError("lm_head is required when tie_embeddings=False")
        logits_proj = model.lm_head(x)

    logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)
    return F.cross_entropy(logits.float(), targets, reduction="mean")
