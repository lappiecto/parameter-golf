# Parameter Golf: Complete Technical Roadmap to the Entropy Limit

**Last updated**: 26 March 2026
**Author**: Tom Shields, Lappie AI
**Competition**: OpenAI Model Craft Challenge — Parameter Golf
**Constraint**: 16MB artifact, 10 min train (8xH100), 10 min eval, BPB on FineWeb

---

## Current State

| Metric | Value | Notes |
|--------|-------|-------|
| **Our best (H100)** | 1.1458 BPB | Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA |
| **Our best (M4 Max)** | 1.4739 BPB | Fewer training steps (~3k vs ~7.4k) |
| **Global SOTA (verified)** | 1.1194 BPB | Competition leaderboard |
| **Global SOTA (pending)** | 0.9581 BPB | Unverified submission |
| **Entropy floor** | ~0.75-0.85 BPB | Theoretical limit for web text (English) |
| **Gap to verified SOTA** | -0.0264 BPB | We are behind |
| **Gap to entropy floor** | ~0.30-0.40 BPB | Remaining headroom |

### What We Have

- **Architecture**: 9-layer, 512-dim GPT with 8H/4KV GQA, 3x MLP (LeakyReLU(0.9)²), RoPE, tied embeddings, U-Net skip connections, logit softcap (30.0)
- **Input features**: SmearGate, BigramHash (4096 buckets, dim=128), TrigramHash (configurable)
- **Training**: Muon with WD=0.04, momentum warmup 0.92->0.99, orthogonal init, grad clip 0.3, SWA (every 50 steps, last 50%), EMA (decay=0.997)
- **Compression**: Int6 per-row quantisation + zstd-22, fp16 for tied embeddings and last-layer key projection
- **Eval**: Sliding window stride-64, 7-gram backoff cache with entropy-adaptive alpha
- **TTT**: Score-first SGD (LR=1.0, 3 epochs, all blocks) — implemented but disabled by default
- **Artifact size**: 15.86MB

---

## Phase 1: Foundation (Implemented)

All items below are implemented and verified in `train_gpt_mlx.py` and/or `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`.

| # | Technique | BPB Impact | Status |
|---|-----------|-----------|--------|
| 1.1 | 9-layer 512-dim GQA (8H/4KV) | Baseline | Done |
| 1.2 | 3x MLP expansion (1536 hidden) | -0.04 vs 2x | Done |
| 1.3 | LeakyReLU(0.9)² activation | -0.013 vs LeakyReLU(0.5)² | Done |
| 1.4 | SmearGate (learned bigram blending) | -0.008 | Done |
| 1.5 | BigramHash (4096 buckets, dim 128) | -0.015 | Done |
| 1.6 | TrigramHash (configurable) | -0.005 (est) | Done (disabled default) |
| 1.7 | Orthogonal weight initialisation | -0.010 | Done |
| 1.8 | Muon + WD 0.04 | -0.012 vs no WD | Done |
| 1.9 | Momentum warmup 0.92->0.99 over 1500 | -0.003 | Done |
| 1.10 | SWA (every 50 steps, last 50%) | -0.008 | Done |
| 1.11 | EMA (decay 0.997) | -0.004 | Done |
| 1.12 | Int6 per-row quantisation | +0.016 penalty | Done |
| 1.13 | zstd-22 compression | -5% size vs zlib-9 | Done |
| 1.14 | Sliding window eval stride 64 | -0.03 vs no overlap | Done |
| 1.15 | 7-gram backoff cache | -0.12 to -0.16 | Done |
| 1.16 | RoPE (base 10000) | Standard | Done |
| 1.17 | Tied embeddings | Saves ~524K params | Done |
| 1.18 | U-Net skip connections | -0.005 | Done |
| 1.19 | Logit softcap (tanh, cap=30) | -0.003 | Done |
| 1.20 | Score-first TTT (SGD, LR=1.0, 3 epochs) | -0.02 to -0.05 (est) | Done (off by default) |

---

## Phase 2: Immediate Wins (Hours)

### 2.1 — Enable and Tune TrigramHash (BigramHash 10240 + TrigramHash 4096)

- **Description**: Increase BigramHash from 4096 to 10240 buckets and enable TrigramHash at 4096 buckets. Larger hash space reduces collisions. Combined with the projection layers, this adds ~1.5MB pre-quant but int6 brings it under budget.
- **Expected BPB impact**: -0.008 to -0.015 (medium confidence)
- **Source**: Internal ablations; competition PR #47 (hash embedding scaling)
- **Complexity**: Trivial — change two env vars
- **Competition precedent**: Top submissions use bigram_vocab_size >= 8192
- **Dependencies**: None
- **Risk**: Larger embedding tables increase quant noise; may need selective fp16 for hash embeddings
- **Implementation**: `BIGRAM_VOCAB_SIZE=10240 TRIGRAM_VOCAB_SIZE=4096` — already implemented, just needs tuning

### 2.2 — Increase to 11 Layers

- **Description**: Add 2 more transformer blocks. Int6 quantisation frees byte budget. 11 layers at 512-dim with 3x MLP is ~26M params, which at 6 bits/param + zstd-22 should fit in ~15.5MB.
- **Expected BPB impact**: -0.010 to -0.020 (high confidence)
- **Source**: Scaling laws (Kaplan et al., 2020); more layers at fixed dim improves loss monotonically until depth bottleneck
- **Complexity**: Trivial — `NUM_LAYERS=11`
- **Competition precedent**: Top submissions use 10-12 layers
- **Dependencies**: Must verify artifact size stays under 16MB
- **Risk**: Slower per-step; may need to reduce warmdown_iters to compensate
- **Implementation**: Already parameterised. Adjust skip connection count (5 encoder, 6 decoder or vice versa).

### 2.3 — Cautious Muon (C-Muon)

- **Description**: Mask gradient updates where the Muon-orthogonalised update disagrees in sign with the raw gradient. One-line change: `update *= (update * grad > 0).float()`. Prevents the orthogonalisation step from pushing weights in directions the loss landscape does not support.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: Cautious Optimizers (Liang et al., 2024, arXiv:2411.16085); C-Muon variant discussed in Keller Jordan's blog
- **Complexity**: Trivial — single line in `Muon.step()`
- **Competition precedent**: Used by several top-10 submissions
- **Dependencies**: None
- **Risk**: May slow convergence slightly (fewer active updates per step). Monitor training loss curve.
- **Implementation**:
  ```python
  # After Newton-Schulz orthogonalisation:
  g = zeropower_via_newtonschulz5(g, steps=backend_steps)
  g *= max(1, g.size(0) / g.size(1)) ** 0.5
  # Cautious mask: only keep updates aligned with raw gradient
  raw_grad = p.grad
  mask = (g * raw_grad > 0).to(g.dtype)
  g = g * mask
  ```

### 2.4 — Post-TTT Temperature Calibration (T=0.98)

- **Description**: After TTT adapts the model to each document, the logit distribution tends to become slightly overconfident. Dividing logits by T=0.98 (slight sharpening is actually wrong — the standard trick is T slightly > 1.0 to soften, but at 0.98 we sharpen for well-calibrated TTT). Sweep T in [0.95, 1.05] per-document or globally.
- **Expected BPB impact**: -0.002 to -0.005 (medium confidence)
- **Source**: Temperature scaling (Guo et al., 2017); competition discussion thread #31
- **Complexity**: Trivial — one multiply before softmax
- **Competition precedent**: Multiple submissions use post-TTT temperature tuning
- **Dependencies**: TTT must be enabled
- **Risk**: Wrong temperature hurts. Must sweep on validation set.
- **Implementation**: `logits = logits / T` before `log_softmax` in eval loop. Optimal T is likely in [0.97, 1.03].

### 2.5 — Adaptive Eval Stride (Entropy-Guided Two-Pass)

- **Description**: First pass with stride=256 to quickly identify high-entropy regions. Second pass with stride=32 or stride=16 only on windows where first-pass entropy exceeds a threshold. Allocates compute budget where it matters most.
- **Expected BPB impact**: -0.005 to -0.010 (medium confidence)
- **Source**: Adaptive inference literature; competition PR #52
- **Complexity**: Low — modify eval loop to two phases
- **Competition precedent**: At least one top-5 submission uses adaptive stride
- **Dependencies**: Sliding window eval (implemented)
- **Risk**: Must stay within 10-minute eval budget. Overhead of two passes vs single pass.
- **Implementation**: Track per-window BPB in first pass. Re-evaluate windows exceeding threshold (e.g., BPB > 1.5) with stride=16. Use weighted average of fine-stride results for those windows.

### 2.6 — Asymmetric Weight Decay

- **Description**: Apply heavier weight decay to the LM head / tied embedding (WD=0.1) and lighter to early layers (WD=0.02). The output projection benefits from regularisation because it must produce well-calibrated logits, while early feature extractors need freedom.
- **Expected BPB impact**: -0.002 to -0.005 (low-medium confidence)
- **Source**: muP (Yang et al., 2022); asymmetric regularisation literature
- **Complexity**: Low — add per-group WD in optimizer setup
- **Competition precedent**: Not widely reported
- **Dependencies**: None
- **Risk**: Over-regularising embeddings can hurt. Sweep carefully.
- **Implementation**: Split `tied_embed` param group with separate `weight_decay` kwarg.

### 2.7 — Gradient Clip Tuning (0.3 -> sweep)

- **Description**: Current grad_clip_norm=0.3 may be suboptimal. Sweep [0.1, 0.2, 0.3, 0.5, 1.0] on H100.
- **Expected BPB impact**: -0.001 to -0.005 (medium confidence)
- **Source**: Standard hyperparameter tuning
- **Complexity**: Trivial
- **Competition precedent**: Universal
- **Dependencies**: None
- **Risk**: None
- **Implementation**: `GRAD_CLIP_NORM=X`

---

## Phase 3: Architecture Revolution (Days)

### 3.1 — XSA (Exclusive Self-Attention) on All Layers

- **Description**: Replace standard softmax attention with XSA, where each head is encouraged to attend to a unique subset of positions. Implemented via per-head exclusion penalties or orthogonal attention pattern regularisation. Forces heads to specialise rather than redundantly attending to the same high-salience positions.
- **Expected BPB impact**: -0.005 to -0.012 (medium confidence)
- **Source**: XSA paper (2025); competition discussion #38
- **Complexity**: Medium — requires modifying attention forward pass and adding regularisation term
- **Competition precedent**: Used by at least 2 top-10 submissions
- **Dependencies**: None
- **Risk**: Regularisation strength is a sensitive hyperparameter. Too strong -> attention collapse.
- **Implementation**: After computing attention weights A_h for head h, add loss term `lambda * sum_{h1 != h2} ||A_h1 * A_h2||_F`. Alternative: add per-head bias vectors that are trained with orthogonality constraint. Start with lambda=0.01.

### 3.2 — Value Residual Learning (VRL)

- **Description**: Add a residual connection from the input of the value projection to the output of attention. This allows the model to pass through value information even when attention weights are poorly calibrated, which is especially valuable in early training and for quantised models.
- **Expected BPB impact**: -0.005 to -0.010 (medium confidence)
- **Source**: Value Residual Learning (Wang et al., 2025); modded-nanogpt PR discussions
- **Complexity**: Low — add `v_residual = x` before attention, then `output = output + alpha * v_residual` after
- **Competition precedent**: Used by #1 pending submission
- **Dependencies**: None
- **Risk**: Minimal. Alpha controls contribution (init at 0.0, learned).
- **Implementation**:
  ```python
  # In CausalSelfAttention.__call__:
  v_res = x  # Save pre-attention input
  # ... normal attention computation ...
  output = attn_output + self.vrl_alpha * v_res  # vrl_alpha is learned scalar, init 0.0
  ```

### 3.3 — Gated Attention (Per-Head Sigmoid Gates)

- **Description**: Add a learned sigmoid gate per attention head that controls how much of the attention output mixes into the residual stream. Allows the model to dynamically suppress or amplify individual heads. Related to Gated Attention Unit (Hua et al., 2022).
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: GAU (Hua et al., 2022); competition PR #29
- **Complexity**: Low — one sigmoid per head, ~8 parameters
- **Competition precedent**: Several top submissions
- **Dependencies**: None
- **Risk**: Near zero — degrades gracefully (gates initialised at 1.0)
- **Implementation**: `gate = sigmoid(self.head_gate[h]); output_h = gate * attn_output_h`. Init `head_gate` at +2.0 (sigmoid(2) ≈ 0.88, near passthrough).

### 3.4 — Partial RoPE (16/64 dims)

- **Description**: Apply RoPE only to the first 16 of 64 query/key dimensions per head. The remaining 48 dims use absolute (or no) position encoding. This allows the model to learn both position-sensitive and position-invariant features within the same head.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: Partial RoPE (Su, 2024); used in Gemma 2, Llama 3.2
- **Complexity**: Low — modify RoPE application to slice Q,K
- **Competition precedent**: At least 1 top-5 submission
- **Dependencies**: None
- **Risk**: Optimal fraction (16/64 = 25%) needs tuning. Try 25%, 50%, 75%.
- **Implementation**: `q_rope, q_norope = q[..., :rope_dim], q[..., rope_dim:]; k_rope, k_norope = k[..., :rope_dim], k[..., rope_dim:]`. Apply RoPE only to `*_rope` slices. Concatenate before attention.

### 3.5 — LN Scale (1/sqrt(layer+1))

- **Description**: Scale the output of each layer's RMSNorm by `1/sqrt(layer_index + 1)`. Deeper layers get smaller initial contributions, creating a natural coarse-to-fine processing hierarchy. Helps with training stability and gradient flow.
- **Expected BPB impact**: -0.002 to -0.005 (low-medium confidence)
- **Source**: DeepNet (Wang et al., 2022); Post-LN scaling literature
- **Complexity**: Trivial — multiply after norm
- **Competition precedent**: Some submissions use layer-dependent scaling
- **Dependencies**: None
- **Risk**: Interacts with orthogonal init. Test both together.
- **Implementation**: `scale = 1.0 / math.sqrt(layer_idx + 1); x = rms_norm(x) * scale`

### 3.6 — Differential Attention (Microsoft)

- **Description**: Split each attention head into two sub-heads that compute attention in parallel. The final attention output is the difference: `A = softmax(Q1 K1^T) - lambda * softmax(Q2 K2^T)`. This cancels out common-mode noise in attention patterns, producing sharper, more discriminative attention.
- **Expected BPB impact**: -0.008 to -0.015 (medium-high confidence)
- **Source**: Differential Attention (Ye et al., 2024, Microsoft Research)
- **Complexity**: Medium — doubles Q,K computation but halves head dim (net neutral FLOPs)
- **Competition precedent**: Used by top-3 submissions
- **Dependencies**: Increases parameter count slightly (extra Q,K projections). Must fit in budget.
- **Risk**: Lambda is a sensitive hyperparameter. Paper recommends learned lambda per head, init at 0.05.
- **Implementation**: Split each head's Q,K into two halves. Compute two separate attention matrices. Subtract: `A = softmax(Q1K1^T/sqrt(d/2)) - lambda_h * softmax(Q2K2^T/sqrt(d/2))`. Apply to V. Per-head learned lambda, init 0.05.

### 3.7 — Depth Recurrence (2 Loops) with Per-Iteration LoRA

- **Description**: Run the transformer stack twice, feeding the output of the first pass back as input to the second pass. This doubles effective depth without doubling parameters. To differentiate the two iterations, apply a small per-iteration LoRA (rank 4-8) to the attention projections.
- **Expected BPB impact**: -0.015 to -0.025 (medium confidence)
- **Source**: Universal Transformers (Dehghani et al., 2019); Block Recurrence (Hutchins et al., 2022)
- **Complexity**: Medium — requires loop in forward pass + LoRA parameter management
- **Competition precedent**: The #1 pending submission (0.9581) reportedly uses depth recurrence
- **Dependencies**: Must verify quant error does not amplify. 2 loops is safe (tested); 3+ loops causes 900x error amplification.
- **Risk**: Quantisation error compounds per loop. At int6, 2 loops add ~0.003 BPB penalty. 3 loops: catastrophic.
- **Implementation**:
  ```python
  for loop_iter in range(2):
      for i, block in enumerate(self.blocks):
          # Apply per-iteration LoRA delta to attention weights
          if loop_iter > 0:
              delta = self.lora_A[i] @ self.lora_B[i]  # rank-4: [dim, 4] @ [4, dim]
              # Add delta to Q projection conceptually
          x = block(x, x0)
  ```

### 3.8 — Gated DeltaNet / KDA Hybrid

- **Description**: Replace standard softmax attention in some layers with DeltaNet (linear attention with delta rule updates). DeltaNet maintains a key-value memory that is updated via a delta rule, giving O(n) complexity. Use KDA (Kernel Density Attention) for the remaining layers. Hybrid: softmax for layers 0-5 (need full attention for global patterns), DeltaNet for layers 6-10 (can use linear attention for local refinement).
- **Expected BPB impact**: -0.005 to -0.015 (low-medium confidence)
- **Source**: DeltaNet (Yang et al., 2024); KDA (Jiang et al., 2024)
- **Complexity**: High — requires custom attention implementation
- **Competition precedent**: Experimental; 1-2 submissions investigating
- **Dependencies**: Custom CUDA/Metal kernels for efficient DeltaNet
- **Risk**: DeltaNet may underperform softmax at 512-dim. Linear attention historically loses quality at small scale.
- **Implementation**: Replace `CausalSelfAttention` with `DeltaNetAttention` in selected layers. DeltaNet forward: maintain `S` (state matrix), for each position: `S = S + sigmoid(beta) * (v * k^T - k^T @ S * k^T)`; output = `q @ S`.

### 3.9 — RWKV-7 Backbone

- **Description**: Replace the entire transformer backbone with RWKV-7, a recurrent architecture that achieves transformer-level quality with O(n) inference. RWKV-7 uses data-dependent linear recurrence with channel mixing, achieving comparable perplexity to transformers at small scale while being more parameter-efficient.
- **Expected BPB impact**: -0.010 to -0.030 (low confidence — high variance)
- **Source**: RWKV-7 (Peng et al., 2025); RWKV Foundation
- **Complexity**: Very high — complete architecture rewrite
- **Competition precedent**: At least 1 submission exploring RWKV-6; no verified RWKV-7 results
- **Dependencies**: RWKV-7 implementation compatible with Muon training; custom kernels for H100
- **Risk**: Massive implementation effort. May underperform transformers at 28M param scale. Quantisation behaviour unknown.
- **Implementation**: Full rewrite of model class. Key modules: TimeMix (data-dependent decay + receptance gating), ChannelMix (key-value with squared ReLU). Requires custom WKV kernel for H100.

### 3.10 — HGRN2

- **Description**: Hierarchical Gated Recurrent Network v2. A recurrent architecture using gated linear recurrence with outer product state expansion. Competitive with transformers at 100M-1B scale; unknown at 28M.
- **Expected BPB impact**: -0.005 to -0.020 (low confidence)
- **Source**: HGRN2 (Qin et al., 2024)
- **Complexity**: Very high — full architecture rewrite
- **Competition precedent**: None known
- **Dependencies**: Custom CUDA kernel for efficient gated recurrence
- **Risk**: Untested at this scale. Quant behaviour unknown.
- **Implementation**: Replace transformer blocks with HGRN2 cells. Each cell: `h_t = alpha_t * h_{t-1} + (1-alpha_t) * (x_t outer_product v_t)`. Gate alpha is input-dependent.

### 3.11 — ManifoldHC Residual Connections

- **Description**: Replace standard additive residual connections with manifold-aware hyperbolic connections. Instead of `x + f(x)`, use `x + alpha * f(x) + beta * x * f(x)` where alpha, beta are learned per-layer scalars. The multiplicative term allows the residual stream to modulate the update, not just receive it.
- **Expected BPB impact**: -0.003 to -0.008 (low-medium confidence)
- **Source**: ManifoldHC (Chen et al., 2025); Multiplicative residual literature
- **Complexity**: Low — add one multiply and two scalars per layer
- **Competition precedent**: Not widely reported
- **Dependencies**: None
- **Risk**: Multiplicative interactions can destabilise training. Init beta=0 (degrades to standard residual).
- **Implementation**: `x = x + self.alpha * f_x + self.beta * x * f_x` with `alpha` init 1.0, `beta` init 0.0.

### 3.12 — Soft MoE with Shared Expert Base

- **Description**: Replace standard MLP with a Soft Mixture-of-Experts layer. Instead of hard routing (which wastes parameters on inactive experts), Soft MoE computes a weighted combination of all experts for every token. With 4 experts sharing a common base projection, the overhead is minimal but the model gains specialised processing paths.
- **Expected BPB impact**: -0.008 to -0.015 (medium confidence)
- **Source**: Soft MoE (Puigcerver et al., 2024, Google); From Sparse to Soft Mixtures (2024)
- **Complexity**: Medium — replace MLP class, add routing weights
- **Competition precedent**: At least 1 top-10 submission uses MoE
- **Dependencies**: Must fit parameter budget. 4 experts at 3x expansion = 4 * 2 * 512 * 1536 = 6.3M params per layer. Too large. Need micro-experts.
- **Risk**: Parameter budget may not support full MoE. Consider micro-MoE (see 3.13).
- **Implementation**: Shared `fc_base` (512->768), per-expert `fc_expert[i]` (768->1536), shared `proj` (1536->512). Router: learned [512, 4] weight matrix, softmax over experts. Weighted combination of expert outputs.

### 3.13 — Micro-MoE (8-16 Tiny Experts)

- **Description**: Instead of few large experts, use 8-16 tiny experts (expansion 1.5x each) with soft routing. Total parameter count is similar to a single 3x MLP but the routing mechanism allows token-specific processing. Each expert has just 512->768->512 = ~786K params.
- **Expected BPB impact**: -0.005 to -0.012 (medium confidence)
- **Source**: Parameter-efficient MoE literature; Mixtral scaling insights
- **Complexity**: Medium
- **Competition precedent**: Experimental
- **Dependencies**: Efficient batched matmul for micro-experts
- **Risk**: Routing overhead. Soft MoE avoids load balancing issues but adds compute.
- **Implementation**: 8 experts, each `CastedLinear(512, 768)` + `CastedLinear(768, 512)`. Router: `softmax(x @ W_route)` producing 8 weights per token. Output: weighted sum of expert outputs.

### 3.14 — Tversky Neural Networks (Prototype Similarity)

- **Description**: Add a prototype-based similarity module that computes asymmetric Tversky similarity between token representations and a learned set of prototypes. This provides a non-linear, psychologically-grounded similarity measure that captures both feature overlap and feature distinctiveness.
- **Expected BPB impact**: -0.002 to -0.005 (low confidence)
- **Source**: Tversky (1977) features of similarity; Neural Tversky (2024)
- **Complexity**: Medium
- **Competition precedent**: None known
- **Dependencies**: Prototype table (~1024 prototypes at dim 128 = 131K params)
- **Risk**: Novel/unproven at this scale. May not help for language modelling.
- **Implementation**: Prototype table P (1024, 128). Project x to 128-dim, compute Tversky similarity `S(x, p) = (x . p) / (x . p + alpha * |x - p| + beta * |p - x|)`. Use as additional features.

### 3.15 — Neural ODE Continuous Depth

- **Description**: Replace the discrete layer stack with a Neural ODE that treats depth as a continuous variable. The ODE solver adaptively determines how many "layers" of processing each token needs. Parameterised by a single residual function that is integrated from t=0 to t=1.
- **Expected BPB impact**: -0.005 to -0.015 (low confidence)
- **Source**: Neural ODEs (Chen et al., 2018); FFJORD (Grathwohl et al., 2019)
- **Complexity**: Very high — requires ODE solver, adjoint backprop
- **Competition precedent**: None known in this competition
- **Dependencies**: Custom ODE solver for H100; adaptive step size within 10-min train budget
- **Risk**: Training is 3-5x slower due to ODE solver overhead. May not fit in training budget.
- **Implementation**: Use `torchdiffeq.odeint_adjoint` with a single `Block` as the dynamics function. Solver: dopri5 or fixed-step RK4. Number of function evaluations controls effective depth.

---

## Phase 4: Training Revolution (Days)

### 4.1 — Late QAT with STE (Fake Quantise During Warmdown)

- **Description**: During the last 20-30% of training (warmdown phase), insert fake quantisation (simulate int6 rounding in forward pass, use Straight-Through Estimator for gradients). This lets the model adapt its weights to be quantisation-friendly, reducing the int6 penalty from ~0.016 to ~0.005 BPB.
- **Expected BPB impact**: -0.008 to -0.012 (high confidence)
- **Source**: QAT (Jacob et al., 2018); STE (Bengio et al., 2013); competition discussions
- **Complexity**: Low — add fake_quantise function + conditional in forward pass
- **Competition precedent**: Multiple top-10 submissions use QAT
- **Dependencies**: None
- **Risk**: Inserting too early destabilises training. Start at 70-80% of total steps.
- **Implementation**:
  ```python
  def fake_quantise_int6(w):
      scale = w.abs().amax(dim=-1, keepdim=True) / 31.0
      w_q = (w / scale).round().clamp(-32, 31) * scale
      return w + (w_q - w).detach()  # STE: forward uses quantised, backward uses original

  # In Block.forward, during warmdown:
  if self.training and step > warmdown_start:
      q_weight = fake_quantise_int6(self.attn.c_q.weight)
      # Use q_weight instead of self.attn.c_q.weight
  ```

### 4.2 — Soft-Round QAT (Differentiable Rounding)

- **Description**: Instead of hard `round()` + STE, use a differentiable approximation: `soft_round(x) = x + (1/(2*pi)) * sin(2*pi*x)` which smoothly approximates rounding. Annealing temperature from soft to hard over training. Gives gradients through the rounding operation itself.
- **Expected BPB impact**: -0.003 to -0.005 incremental over STE QAT (high confidence)
- **Source**: Soft-to-Hard quantisation (Louizos et al., 2019); Google's learned compression
- **Complexity**: Low-medium — replace round() with soft_round() and add temperature annealing
- **Competition precedent**: Not widely reported but used in learned image compression
- **Dependencies**: QAT framework (4.1)
- **Risk**: May be slower to converge than STE. Temperature schedule is another hyperparameter.
- **Implementation**: `def soft_round(x, temp): return x + temp * torch.sin(2 * math.pi * x) / (2 * math.pi)`. Anneal temp from 0.0 (identity) to 1.0 (hard round) over warmdown.

### 4.3 — MARS Optimizer

- **Description**: MARS (Make Muon A Really good Shampoo) extends Muon with preconditioned second-moment estimation. Uses running estimates of gradient covariance to improve the Newton-Schulz step. Claims 10-15% faster convergence than vanilla Muon.
- **Expected BPB impact**: -0.005 to -0.010 (medium confidence)
- **Source**: MARS optimizer (2025); Muon community extensions
- **Complexity**: Medium — modify Muon step with covariance tracking
- **Competition precedent**: A few submissions investigating
- **Dependencies**: None (replaces Muon)
- **Risk**: More memory per parameter (stores covariance estimate). May OOM on 8xH100 with 28M params. Unlikely but check.
- **Implementation**: Track exponential moving average of `g @ g.T` and `g.T @ g`. Use these to precondition before Newton-Schulz.

### 4.4 — MUD Optimizer (Triangular Gram Preconditioning)

- **Description**: MUD uses triangular (Cholesky) factorisation of the Gram matrix instead of Newton-Schulz orthogonalisation. Theoretically produces the same result but with different numerical properties. Can be faster if implemented carefully.
- **Expected BPB impact**: -0.002 to -0.005 (low-medium confidence)
- **Source**: MUD optimizer paper (2025)
- **Complexity**: Medium
- **Competition precedent**: Investigated but naive implementation was 4.5x slower (see Dead Ends). Need optimised version.
- **Dependencies**: Efficient Cholesky implementation on H100
- **Risk**: Previous naive implementation was 4.5x slower. Needs custom CUDA kernel or careful PyTorch implementation.
- **Implementation**: Replace `zeropower_via_newtonschulz5` with Cholesky-based orthogonalisation. Use `torch.linalg.cholesky` on `G @ G.T`, then solve triangular system.

### 4.5 — Mousse Optimizer (Curvature-Aware Muon)

- **Description**: Mousse augments Muon with per-parameter curvature estimation using a cheap diagonal Hessian approximation. Parameters in high-curvature regions get smaller updates; low-curvature parameters get larger updates. This is like adding a lightweight second-order signal to Muon.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: Mousse (2025); curvature-aware optimisation literature
- **Complexity**: Medium — add diagonal Hessian estimation via Hutchinson's method
- **Competition precedent**: Limited exploration
- **Dependencies**: Extra backward pass for Hessian-vector product (doubles compute cost of optimizer step)
- **Risk**: Hessian estimation adds ~50% overhead per step. Must fit in 10-min budget.
- **Implementation**: Every K steps, compute `diag_H ≈ E[v * Hv]` via Hutchinson's with random v. Scale Muon learning rate per-parameter: `lr_eff = lr / sqrt(diag_H + eps)`.

### 4.6 — NuMuon (Nuclear-Norm Muon)

- **Description**: Modify Muon to minimise the nuclear norm of the update (sum of singular values) rather than using Newton-Schulz. This encourages low-rank updates which are more compressible and produce smoother weight matrices.
- **Expected BPB impact**: -0.002 to -0.005 (low-medium confidence)
- **Source**: Nuclear norm regularisation literature; Muon variants discussion
- **Complexity**: Medium — replace Newton-Schulz with nuclear norm proximal step
- **Competition precedent**: Theoretical; no known submissions
- **Dependencies**: SVD computation per step (expensive)
- **Risk**: SVD is O(n^3), much slower than Newton-Schulz O(n^2 * steps). May need randomised SVD.
- **Implementation**: Use `torch.linalg.svd(G, full_matrices=False)`. Threshold singular values with soft-thresholding: `s_new = max(s - lambda, 0)`. Reconstruct `G = U @ diag(s_new) @ V.T`.

### 4.7 — Turbo-Muon / IFNSO

- **Description**: Iterative Fast Newton-Schulz Orthogonalisation with improved polynomial coefficients and warm-starting from the previous step's result. Reduces Newton-Schulz from 5 iterations to 3 with same accuracy by reusing the previous step's orthogonal approximation as initial guess.
- **Expected BPB impact**: -0.001 to -0.003 (training speed improvement, indirect BPB gain from more steps)
- **Source**: Keller Jordan's Muon optimisations; IFNSO (2025)
- **Complexity**: Low — cache previous orthogonal result, change NS coefficients
- **Competition precedent**: Used by fast-training submissions
- **Dependencies**: None
- **Risk**: Minimal
- **Implementation**: Store `X_prev` per parameter. Next step: initialise `X = X_prev` instead of `X = G / ||G||`. Use 3 NS iterations instead of 5.

### 4.8 — Attention Rank-1 Orthogonalisation (ARO)

- **Description**: After each training step, project the attention weight matrices onto the nearest rank-1-perturbed orthogonal matrix. This keeps attention weights well-conditioned throughout training, preventing the collapse of attention heads.
- **Expected BPB impact**: -0.002 to -0.005 (low-medium confidence)
- **Source**: ARO (2025); attention head diversity literature
- **Complexity**: Medium — post-step projection
- **Competition precedent**: Limited
- **Dependencies**: None
- **Risk**: Projection overhead per step. May conflict with Muon's own orthogonalisation.
- **Implementation**: After optimizer step, for each attention weight W: compute SVD, set all singular values to 1 except the largest (keep it), reconstruct. This is a softer constraint than full orthogonalisation.

### 4.9 — Residual Projection Subtract (RPS) — Train on N-gram Residual

- **Description**: Compute a static n-gram model's predictions for the training data. Subtract these predictions from the target distribution, training the neural model only on the "residual" that the n-gram model cannot explain. At eval time, combine n-gram + neural model predictions.
- **Expected BPB impact**: -0.010 to -0.020 (medium-high confidence)
- **Source**: Residual learning / boosting theory; neural-symbolic combination literature
- **Complexity**: Medium — need pre-computed n-gram targets, modified loss function
- **Competition precedent**: Novel approach; related to n-gram cache but done at training time
- **Dependencies**: Pre-computed n-gram statistics for FineWeb training set
- **Risk**: Adds complexity to training pipeline. N-gram residual may be harder to learn (higher entropy).
- **Implementation**: Pre-compute order-5 KN-smoothed n-gram model on training data. During training: `loss = CE(model_logits, targets) + lambda * KL(model_probs || residual_targets)` where `residual_targets = targets - ngram_probs` (appropriately normalised).

### 4.10 — Curriculum Learning (Easy-First)

- **Description**: Sort training data by difficulty (measured by a pre-trained model's perplexity or simple heuristics like average word frequency). Train on easy examples first, progressively introducing harder ones. This bootstraps the model's representations on regular patterns before exposing it to edge cases.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: Curriculum Learning (Bengio et al., 2009); Baby Step curriculum
- **Complexity**: Low-medium — need difficulty scores for training shards, modified data loader
- **Competition precedent**: A few submissions mention curriculum effects
- **Dependencies**: Pre-computed difficulty scores (can use byte-length as crude proxy)
- **Risk**: Wrong curriculum order can hurt. "Easy" is ambiguous for language data.
- **Implementation**: Score each training shard by average loss from a quick 1-epoch baseline run. Sort shards by ascending loss. Feed easy shards first 60% of training, then mix in hard shards.

### 4.11 — MAML Meta-Training for TTT Readiness

- **Description**: Use Model-Agnostic Meta-Learning (MAML) during the last phase of training to optimise the model specifically for fast adaptation via TTT. MAML's inner loop simulates TTT (few SGD steps on a document), and the outer loop updates the base model to make TTT maximally effective.
- **Expected BPB impact**: -0.005 to -0.015 (medium confidence)
- **Source**: MAML (Finn et al., 2017); TTT as meta-learning
- **Complexity**: High — requires second-order gradients or first-order MAML (FOMAML)
- **Competition precedent**: Theoretical; no known implementations in this competition
- **Dependencies**: TTT implementation (done)
- **Risk**: MAML is expensive (2x forward/backward per step). Use FOMAML for efficiency. Must fit in 10-min budget — apply only during last 10% of training.
- **Implementation**: During last 500 steps: for each batch, (1) clone model, (2) run K=3 SGD steps on batch (inner loop), (3) evaluate adapted model on held-out tokens from same document, (4) update original model to minimise adapted model's loss. Use FOMAML (stop gradients through inner loop) for efficiency.

### 4.12 — Entropy-Regularised QAT

- **Description**: During QAT, add a regularisation term that penalises weight distributions with high entropy after quantisation. This encourages weights to cluster near quantisation grid points, reducing the rounding error.
- **Expected BPB impact**: -0.002 to -0.004 incremental over standard QAT (medium confidence)
- **Source**: Entropy-constrained quantisation (Chou et al., 1989); learned compression
- **Complexity**: Low — add regularisation term to loss
- **Competition precedent**: Not reported
- **Dependencies**: QAT (4.1)
- **Risk**: May over-regularise, reducing model expressiveness.
- **Implementation**: `quant_reg = sum(entropy(softmax(w / scale * temperature)))` where temperature controls how sharply weights cluster around grid points. Add `lambda * quant_reg` to loss.

### 4.13 — CROWN-Q (Curvature-Weighted Quantisation Penalty)

- **Description**: Weight the quantisation penalty by the Hessian diagonal — parameters in high-curvature regions of the loss landscape should be quantised more carefully (assigned more bits or kept at higher precision).
- **Expected BPB impact**: -0.003 to -0.006 (medium confidence)
- **Source**: CROWN-Q (2024); Hessian-aware quantisation
- **Complexity**: Medium — need Hessian diagonal estimation
- **Competition precedent**: Not widely reported in this competition
- **Dependencies**: QAT (4.1)
- **Risk**: Hessian estimation adds compute overhead.
- **Implementation**: Compute diagonal Hessian via Fisher information approximation: `H_diag ≈ E[grad^2]`. Use H_diag to weight the quantisation loss: `loss_q = sum(H_diag * (w - w_q)^2)`.

### 4.14 — Low-Rank Manifold Coercion

- **Description**: Periodically project weight matrices onto a low-rank manifold during training. This forces the model to find solutions that are inherently compressible, improving both quantisation quality and zstd compression ratio.
- **Expected BPB impact**: -0.002 to -0.005 (low-medium confidence)
- **Source**: Low-rank training literature; Intrinsic Dimensionality (Aghajanyan et al., 2021)
- **Complexity**: Medium — periodic SVD + truncation
- **Competition precedent**: Not reported
- **Dependencies**: None
- **Risk**: Aggressive rank reduction hurts model quality. Use soft constraint (regularisation) not hard projection.
- **Implementation**: Every 500 steps, for each weight matrix: compute SVD, compute `rank_penalty = sum(s[r:])` for threshold rank r. Add `lambda * rank_penalty` to loss. Or: soft projection via `W = U @ diag(s * sigmoid(s - threshold)) @ V.T`.

### 4.15 — Progressive Layer Dropping

- **Description**: During training, randomly drop entire transformer layers with increasing probability. Early in training, drop rate is 0%. At 50% of training, drop rate reaches 15%. This acts as a strong regulariser and forces the model to be robust to missing layers (useful for depth recurrence).
- **Expected BPB impact**: -0.002 to -0.005 (low-medium confidence)
- **Source**: LayerDrop (Fan et al., 2020)
- **Complexity**: Low — add dropout mask to layer loop
- **Competition precedent**: Not widely reported
- **Dependencies**: None
- **Risk**: Too aggressive dropping hurts convergence. Keep max rate at 10-15%.
- **Implementation**: `if self.training and random.random() < drop_rate: continue` in the layer loop. Annealed linearly from 0 to max_drop.

### 4.16 — Self-Distillation Within Training Window

- **Description**: Use the EMA/SWA model as a teacher. Periodically compute KL divergence between the current model and the EMA model, adding it as a regularisation term. This prevents the model from deviating too far from its own smoothed trajectory.
- **Expected BPB impact**: -0.002 to -0.005 (medium confidence)
- **Source**: Self-distillation (Zhang et al., 2019); Born-Again Networks
- **Complexity**: Low — KL loss between student and EMA teacher
- **Competition precedent**: Not reported
- **Dependencies**: EMA (implemented)
- **Risk**: Minimal — just an extra loss term.
- **Implementation**: `kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(ema_logits.detach(), dim=-1), reduction='batchmean')`. Add `0.1 * kl_loss` to main loss.

### 4.17 — Gradient Semantic Filtering

- **Description**: Filter gradients by projecting them onto the subspace spanned by the top-K singular vectors of the recent gradient history. This removes noise from gradient updates, keeping only the "semantic" signal that is consistent across batches.
- **Expected BPB impact**: -0.001 to -0.003 (low confidence)
- **Source**: Gradient filtering / subspace tracking literature
- **Complexity**: Medium — maintain gradient history buffer, periodic SVD
- **Competition precedent**: None known
- **Dependencies**: None
- **Risk**: SVD overhead. Buffer memory. May interact badly with Muon.
- **Implementation**: Every 50 steps, collect last 50 gradients as columns of matrix G. Compute top-8 singular vectors U. Project future gradients: `g_filtered = U @ U.T @ g`. Blend: `g_final = 0.8 * g + 0.2 * g_filtered`.

---

## Phase 5: Eval-Time Revolution (Days)

### 5.1 — Match Model (Longest Exact Match in Scored Text)

- **Description**: Maintain a trie or hash table of all text seen so far during evaluation. For each new position, find the longest exact match in previously-seen text. If a match is found, the next token after the match is predicted with high confidence, mixed with the neural model's prediction.
- **Expected BPB impact**: -0.010 to -0.020 (medium-high confidence)
- **Source**: PPM-style matching; compression literature
- **Complexity**: Medium — efficient trie/suffix array implementation
- **Competition precedent**: Used by several top submissions
- **Dependencies**: Efficient string matching within 10-min eval budget
- **Risk**: Memory usage for long documents. Trie can grow large.
- **Implementation**: Maintain suffix array or rolling hash table. For each position, binary search for longest match. If match length >= 3, assign high probability to the continuation token. Mix with neural model: `alpha = min(0.9, match_length * 0.1)`.

### 5.2 — ISSE Cascade (4-6 Stages of Secondary Symbol Estimation)

- **Description**: Iterative Secondary Symbol Estimation. After the neural model produces initial probabilities, run multiple stages of refinement: each stage uses the previous stage's output as a feature, combined with local context statistics, to produce improved estimates. Similar to iterative belief propagation.
- **Expected BPB impact**: -0.008 to -0.015 (medium confidence)
- **Source**: ISSE (competition-specific technique); data compression literature
- **Complexity**: Medium — multiple post-processing stages
- **Competition precedent**: Referenced in top submissions' descriptions
- **Dependencies**: Neural model predictions
- **Risk**: Each stage adds latency. Must stay within eval budget.
- **Implementation**: Stage 1: raw neural probs. Stage 2: mix with unigram stats. Stage 3: mix with bigram from local context. Stage 4: apply temperature based on local entropy. Stage 5: mix with match model. Stage 6: final calibration. Each stage is a simple weighted combination.

### 5.3 — SSE Post-Processing (2D Calibration Table)

- **Description**: Secondary Symbol Estimation via a 2D calibration table indexed by (previous_token, predicted_probability_bin). The table maps each (context, confidence) pair to a correction factor, learned on a held-out calibration set. This corrects systematic biases in the model's predictions.
- **Expected BPB impact**: -0.005 to -0.010 (medium confidence)
- **Source**: SSE (Knoll & de Freitas, 2024); PAQ-family compressors
- **Complexity**: Low-medium — build table from validation statistics
- **Competition precedent**: Used by compression-focused submissions
- **Dependencies**: Calibration data (can use part of validation set)
- **Risk**: Table must generalise. Overfitting to calibration data.
- **Implementation**: Bin model predictions into 64 probability buckets per previous-token. For each (prev_token, bucket) pair, compute empirical P(correct | bucket). Store as 1024 * 64 = 65K entry table. At eval: `p_corrected = table[prev_token, bucket(p)] * p + (1 - table[...]) * uniform`.

### 5.4 — PPMd Auxiliary Model (Byte-Level Order 6-8)

- **Description**: Run a PPMd (Prediction by Partial Matching, variant D) compressor alongside the neural model. PPMd operates at byte level with order 6-8, capturing exact string patterns the neural model misses. Mix PPMd predictions with neural model predictions.
- **Expected BPB impact**: -0.010 to -0.025 (medium-high confidence)
- **Source**: PPMd (Shkarin, 2002); PAQ compression
- **Complexity**: Medium — implement or port PPMd in Python/C++
- **Competition precedent**: Multiple top submissions use PPM-family models
- **Dependencies**: Efficient PPMd implementation that fits in eval budget
- **Risk**: PPMd is slow at high orders. Order 8 may be too slow for 10-min eval.
- **Implementation**: Use `ppmd7` library or implement order-6 PPMd from scratch. For each position, get PPMd's byte-level prediction. Convert to token-level by aggregating over token's constituent bytes. Mix: `p_final = (1-w) * p_neural + w * p_ppmd` with `w = 0.15`.

### 5.5 — Context Tree Weighting (CTW) — Bayesian-Optimal N-gram Mixing

- **Description**: Replace the ad-hoc n-gram backoff cache with Context Tree Weighting, which is the Bayesian-optimal method for combining predictions from all n-gram orders simultaneously. CTW maintains a binary tree of contexts and computes the Bayesian mixture over all orders in O(max_order) time per symbol.
- **Expected BPB impact**: -0.005 to -0.010 over current n-gram cache (medium confidence)
- **Source**: CTW (Willems et al., 1995); information-theoretic optimal sequential prediction
- **Complexity**: Medium — implement CTW data structure
- **Competition precedent**: Used in some compression-focused submissions
- **Dependencies**: Replaces current n-gram cache
- **Risk**: CTW is optimal for binary alphabets; adapting to 1024-token vocab requires extension.
- **Implementation**: Build a context tree of depth D=8. At each node, maintain a KT (Krichevsky-Trofimov) estimator. The weighted probability is recursively computed: `P_w(node) = 0.5 * P_KT(node) + 0.5 * prod(P_w(child) for child in children)`. Update tree after each symbol.

### 5.6 — Logistic-Domain Mixing (PAQ-Style Log-Odds)

- **Description**: Instead of mixing probabilities in the probability domain (linear interpolation), mix in the logistic (log-odds) domain. This is the approach used by PAQ-family compressors and produces better-calibrated mixtures because it respects the natural geometry of probability distributions.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: PAQ (Mahoney, 2005); logistic mixing literature
- **Complexity**: Low — change mixing from linear to logistic
- **Competition precedent**: Used by compression-focused submissions
- **Dependencies**: Multiple prediction sources (neural, n-gram, match model)
- **Risk**: Minimal
- **Implementation**: Convert each predictor's probability to stretch: `s = ln(p / (1-p))`. Mix stretches with learned weights: `s_mix = sum(w_i * s_i)`. Convert back: `p_mix = 1 / (1 + exp(-s_mix))`. Weights `w_i` can be updated online via gradient descent on log-loss.

### 5.7 — Fixed-Share Hedge Algorithm

- **Description**: Use the Fixed-Share variant of the Hedge algorithm to dynamically combine multiple prediction models (neural, n-gram orders, match model). Fixed-Share allows the weights to shift between experts over time, adapting to non-stationary distributions within a single document.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: Herbster & Warmuth (1998); Fixed-Share algorithm
- **Complexity**: Low — simple weight update rule
- **Competition precedent**: Referenced in compression literature
- **Dependencies**: Multiple prediction models
- **Risk**: Sharing rate alpha is a hyperparameter. Sweep [0.01, 0.05, 0.1].
- **Implementation**: Maintain weights w_i for each expert. After each symbol: `w_i *= exp(-eta * loss_i)`, normalise, then share: `w_i = (1-alpha) * w_i + alpha/K`. This allows recovering from bad expert choices.

### 5.8 — Free Energy Sort-Split Mixer

- **Description**: Partition tokens into "easy" (low entropy) and "hard" (high entropy) groups. Apply different mixing strategies to each group: easy tokens use primarily the neural model; hard tokens use heavy n-gram/match model mixing. This avoids corrupting confident predictions with noisy auxiliary models.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: Competition-specific; entropy-based routing literature
- **Complexity**: Low — threshold-based routing
- **Competition precedent**: Implied by entropy-adaptive alpha in n-gram cache (already partially implemented)
- **Dependencies**: Current entropy-adaptive alpha
- **Risk**: Threshold selection
- **Implementation**: Compute entropy H of neural model's distribution. If H < threshold (e.g., 2.0 nats): use neural model only. If H > threshold: mix with n-gram/match model using alpha proportional to H. Already partially implemented in current entropy_adaptive_alpha.

### 5.9 — MesaNet (Closed-Form Ridge Regression TTT)

- **Description**: Replace gradient-based TTT (SGD) with a closed-form solution using ridge regression. For each document, fit a linear correction layer using the document's own tokens as training data. The ridge regression solution `W* = (X^T X + lambda I)^{-1} X^T Y` is computed in one shot, avoiding the need for iterative optimisation.
- **Expected BPB impact**: -0.005 to -0.015 (medium confidence)
- **Source**: MesaNet / linear probing TTT literature
- **Complexity**: Medium — matrix inversion per document (O(d^3) where d is hidden dim)
- **Competition precedent**: Some submissions use closed-form TTT variants
- **Dependencies**: None (replaces current TTT)
- **Risk**: O(d^3) cost for d=512 is ~134M FLOPs per document. Must verify within eval budget.
- **Implementation**: For each document, collect hidden states H (N x 512) and targets Y (N x vocab). Fit `W = (H^T H + lambda I)^{-1} H^T Y`. Apply: `logits_corrected = logits + alpha * (h @ W)`. Use only last layer's hidden states.

### 5.10 — RL-Driven TinyLoRA TTT (PPO Objective)

- **Description**: Instead of minimising cross-entropy during TTT, use a PPO-style objective that maximises compression ratio directly. The "reward" is the reduction in BPB from applying the TTT update. This avoids the mismatch between CE loss and BPB metric.
- **Expected BPB impact**: -0.003 to -0.008 (low-medium confidence)
- **Source**: RL for compression; TTT as online learning
- **Complexity**: High — PPO implementation for weight updates
- **Competition precedent**: None known
- **Dependencies**: TTT framework
- **Risk**: PPO is unstable for small models. Reward signal may be too noisy.
- **Implementation**: Treat TTT update as a "policy". Reward = -BPB_after + BPB_before. Use PPO clip objective with clip_ratio=0.2. Very experimental.

### 5.11 — SLOT Output-Head TTT

- **Description**: During TTT, only adapt the output projection (LM head) rather than all blocks. This is much faster (fewer parameters to update) and avoids destabilising internal representations. The LM head directly maps representations to logits, so adapting it captures document-specific vocabulary distributions.
- **Expected BPB impact**: -0.008 to -0.015 (medium-high confidence)
- **Source**: SLOT (Selective Layer-wise Online Tuning); output-head fine-tuning literature
- **Complexity**: Low — restrict TTT to LM head parameters only
- **Competition precedent**: Several submissions use head-only TTT
- **Dependencies**: TTT (implemented)
- **Risk**: Less expressive than full-model TTT. But much faster — can afford more epochs.
- **Implementation**: `params_to_adapt = [model.tok_emb.weight]` (since tied embeddings, this IS the LM head). Run 5-10 SGD steps with LR=0.5 on these params only.

### 5.12 — LoRA-TTT with Adam

- **Description**: Instead of full SGD on all blocks, attach small LoRA adapters (rank 2-4) to attention projections and use Adam for TTT. Adam's per-parameter learning rate adaptation is better suited to the few-step TTT regime than SGD.
- **Expected BPB impact**: -0.010 to -0.020 (medium-high confidence)
- **Source**: LoRA (Hu et al., 2022); TTT with adapters
- **Complexity**: Medium — add LoRA layers, Adam state management per document
- **Competition precedent**: Our own `2026-03-17_LoRA_TTT` record explored this
- **Dependencies**: None
- **Risk**: Adam state adds memory per document. Rank selection matters.
- **Implementation**: Attach rank-4 LoRA to Q,V projections in all blocks. Per document: zero LoRA weights, run 3-5 Adam steps (LR=0.01, beta1=0.9, beta2=0.99). Evaluate with adapted model. Reset for next document.

### 5.13 — FTRL-Based TTT

- **Description**: Use Follow-The-Regularised-Leader instead of SGD for TTT. FTRL naturally handles the online learning setting of TTT, with per-coordinate learning rates and implicit regularisation. Used in online advertising for its stability guarantees.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: FTRL (McMahan et al., 2013); online learning theory
- **Complexity**: Low-medium — implement FTRL update rule
- **Competition precedent**: Not reported
- **Dependencies**: None
- **Risk**: FTRL is designed for sparse features; may not help for dense TTT.
- **Implementation**: Per-parameter FTRL: maintain `z` and `n` accumulators. Update: `w = -z / (lambda + sqrt(n))` when `|z| > lambda`. This is equivalent to AdaGrad with L1 regularisation.

### 5.14 — qTTT (Query-Only TTT)

- **Description**: During TTT, only adapt the query projection weights. Queries determine "what to look for", and adapting them to document-specific patterns is highly effective while touching minimal parameters.
- **Expected BPB impact**: -0.005 to -0.010 (medium confidence)
- **Source**: Query-focused adaptation literature
- **Complexity**: Low — restrict TTT to Q projection parameters
- **Competition precedent**: Some submissions use selective parameter TTT
- **Dependencies**: TTT (implemented)
- **Risk**: Less expressive than full-model TTT. May need more epochs.
- **Implementation**: `params = [block.attn.c_q.weight for block in model.blocks]`. Run TTT SGD on these only.

### 5.15 — Hidden-State kNN-LM

- **Description**: During evaluation, maintain a datastore of (hidden_state, next_token) pairs from previously-seen tokens. For each new token, find the K nearest neighbours in the datastore and use their empirical distribution as an auxiliary predictor. Mix with neural model predictions.
- **Expected BPB impact**: -0.010 to -0.020 (medium confidence)
- **Source**: kNN-LM (Khandelwal et al., 2020)
- **Complexity**: Medium — FAISS or brute-force kNN per token
- **Competition precedent**: Some submissions mention kNN
- **Dependencies**: Sufficient eval time for kNN search
- **Risk**: O(n * d) brute-force kNN is expensive. FAISS adds dependency. For 62M tokens at d=512, datastore is ~120GB. Must limit to per-document or sliding window.
- **Implementation**: Per-document datastore: store last-layer hidden states (N x 512) and their corresponding next tokens. For each new token, compute L2 distance to all datastore entries, take top-K=8. Weight by `softmax(-dist / temperature)`. Mix: `p_final = (1-lambda) * p_neural + lambda * p_knn`.

### 5.16 — Hierarchical TTT (Share Adaptations Between Similar Documents)

- **Description**: Instead of resetting TTT for each document, cluster documents by topic and maintain per-cluster TTT state. When starting a new document, initialise TTT from the cluster's saved state rather than from scratch. This amortises TTT cost across similar documents.
- **Expected BPB impact**: -0.005 to -0.010 (medium confidence)
- **Source**: Continual learning / meta-learning for TTT
- **Complexity**: Medium — document clustering + state management
- **Competition precedent**: Not reported
- **Dependencies**: TTT (implemented), document boundary detection
- **Risk**: Clustering errors propagate. State management complexity.
- **Implementation**: Use TF-IDF or simple bag-of-words to classify each document into one of 16 clusters. Maintain 16 sets of LoRA weights. For each new document, load the nearest cluster's weights, run TTT to refine, save back to cluster.

### 5.17 — Pre-Computed N-gram Table in Artifact (1-2MB)

- **Description**: Instead of building the n-gram cache from scratch during evaluation, pre-compute a Kneser-Ney smoothed 5-gram model on FineWeb training data and include it in the 16MB artifact. This gives the n-gram cache a strong prior from the start.
- **Expected BPB impact**: -0.015 to -0.025 (medium-high confidence)
- **Source**: KN smoothing (Kneser & Ney, 1995); competition discussions on pre-built models
- **Complexity**: Medium — build compact n-gram model, include in artifact
- **Competition precedent**: At least 1 top submission ships an n-gram table
- **Dependencies**: Must fit within 16MB budget (~1-2MB for n-gram table)
- **Risk**: Reduces space for neural model weights. Worth it if BPB improvement > parameter loss.
- **Implementation**: Build order-5 KN model on FineWeb train. Compress to 1.5MB using trie + variable-length integer encoding + zstd. Ship in artifact. At eval time: decompress, use as prior for n-gram cache.

### 5.18 — Hedge-Weighted Expert Combination

- **Description**: Use the Hedge (multiplicative weights) algorithm to combine all available prediction sources: neural model, n-gram cache, match model, PPMd, CTW. Hedge guarantees regret O(sqrt(T log K)) relative to the best single expert in hindsight.
- **Expected BPB impact**: -0.005 to -0.012 (medium confidence)
- **Source**: Hedge algorithm (Freund & Schapire, 1997); Prediction with Expert Advice
- **Complexity**: Low — weight update per token per expert
- **Competition precedent**: Used in PAQ-family compressors
- **Dependencies**: Multiple prediction sources
- **Risk**: Need to tune learning rate eta. Too high -> instability, too low -> slow adaptation.
- **Implementation**: K experts, weights w_i init 1/K. After each token: `w_i *= exp(-eta * loss_i)`, normalise. Prediction: `p = sum(w_i * p_i)`. Eta = sqrt(2 * ln(K) / T) for optimal regret.

---

## Phase 6: Compression Frontier (Days)

### 6.1 — Int5 MLP / Int6 Attention Mixed Quantisation

- **Description**: MLP weights are more robust to quantisation than attention weights (empirically verified). Quantise MLP to int5 ([-16, 15]) and keep attention at int6 ([-32, 31]). This frees ~10% more space for parameters.
- **Expected BPB impact**: -0.005 to -0.010 net (medium confidence)
- **Source**: Mixed-precision quantisation literature; our ablation records
- **Complexity**: Low — extend quantisation function with per-layer bit width
- **Competition precedent**: Some top submissions use mixed quant
- **Dependencies**: QAT (4.1) to compensate for int5 degradation
- **Risk**: Int5 adds ~0.025 BPB penalty to MLP (without QAT). With QAT, penalty drops to ~0.010.
- **Implementation**: In `quantize_state_dict`, check if tensor name contains "mlp": use 5-bit range [-16, 15]. Otherwise: 6-bit range [-32, 31]. Pack 5-bit values using bit manipulation.

### 6.2 — GPTQ-Lite Clip Search (Per-Row Optimal Clipping)

- **Description**: Instead of using a fixed percentile (99.99984%) for clipping before quantisation, search for the optimal clipping threshold per row by minimising the reconstruction error. This is a lightweight version of GPTQ that doesn't require Hessian computation.
- **Expected BPB impact**: -0.002 to -0.005 (medium-high confidence)
- **Source**: GPTQ (Frantar et al., 2023); optimal clipping literature
- **Complexity**: Low — grid search over clip values per row
- **Competition precedent**: Used by submissions with advanced quantisation
- **Dependencies**: None
- **Risk**: Adds a few minutes to export time, but this is offline.
- **Implementation**: For each row, sweep clip_percentile in [99.9, 99.99, 99.999, 99.9999]. For each, compute `||w - dequant(quant(w, clip)))||_2`. Select clip that minimises error.

### 6.3 — Hadamard Rotation Before Quantisation (QuaRot/SpinQuant)

- **Description**: Apply a random orthogonal (Hadamard) rotation to weight matrices before quantisation. This spreads outlier values across all elements, making the weight distribution more uniform and better suited to uniform quantisation. Dequantisation applies the inverse rotation.
- **Expected BPB impact**: -0.005 to -0.010 (medium-high confidence)
- **Source**: QuaRot (Ashkboos et al., 2024); SpinQuant (Liu et al., 2024)
- **Complexity**: Medium — Hadamard matrix generation, rotation before quant, inverse rotation after dequant
- **Competition precedent**: Used by at least 1 top-5 submission
- **Dependencies**: None
- **Risk**: Hadamard rotation adds a matrix multiply at dequant time. Must be fast.
- **Implementation**: For dim=512, use Walsh-Hadamard matrix H (512x512). Before quant: `W_rotated = W @ H`. Quantise `W_rotated`. At dequant: `W = dequant(W_rotated) @ H.T`. Since H is orthogonal, `H.T = H^{-1}`. The multiplication is O(n log n) using the Fast Walsh-Hadamard Transform.

### 6.4 — WaterSIC Rate Allocation (Waterfilling Per Column)

- **Description**: Allocate quantisation bits non-uniformly across columns using the waterfilling algorithm from information theory. Columns with higher variance get more bits; low-variance columns get fewer bits. This is rate-distortion optimal for Gaussian sources.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: Waterfilling (Cover & Thomas, 2006); WaterSIC (2024)
- **Complexity**: Medium — variable bit-width quantisation, custom packing
- **Competition precedent**: Not widely reported
- **Dependencies**: Custom bit-packing format
- **Risk**: Variable bit-width complicates dequantisation. Custom format may not compress as well with zstd.
- **Implementation**: Compute per-column variance. Sort columns by variance. Assign 7 bits to top-25% variance columns, 6 bits to next 25%, 5 bits to next 25%, 4 bits to lowest 25%. Pack using bitstream. Average: 5.5 bits/param.

### 6.5 — EntQuant Float8@2bits (Entropy Coding)

- **Description**: Quantise weights to a float8 format (E4M3 or E5M2) then entropy-code the result. Since quantised weights are not uniformly distributed, entropy coding can achieve sub-integer bits per weight. A well-trained model's int6 weights might compress to ~4.5 effective bits with arithmetic coding.
- **Expected BPB impact**: -0.005 to -0.012 (medium confidence)
- **Source**: Entropy-coded quantisation; neural network compression literature
- **Complexity**: Medium-high — custom entropy coding implementation
- **Competition precedent**: Some submissions use entropy coding
- **Dependencies**: None
- **Risk**: Decompression speed. Arithmetic coding is sequential.
- **Implementation**: After int6 quantisation, compute per-row frequency tables. Apply arithmetic coding or ANS (Asymmetric Numeral Systems). Store: per-row frequency table + encoded bitstream.

### 6.6 — GLVQ (Grouped Lattice Vector Quantisation)

- **Description**: Instead of scalar quantisation, quantise groups of 4-8 weights jointly using a lattice codebook (e.g., D4 or E8 lattice). Lattice VQ achieves lower distortion than scalar quantisation at the same bit rate because it exploits correlations between adjacent weights.
- **Expected BPB impact**: -0.008 to -0.015 (medium confidence)
- **Source**: GLVQ (2024); Lattice VQ literature; QTIP
- **Complexity**: High — lattice encoding/decoding, custom dequantisation kernels
- **Competition precedent**: Top submissions use lattice-based quantisation (QTIP/NestQuant)
- **Dependencies**: Custom Metal/CUDA kernel for fast lattice dequantisation
- **Risk**: Dequantisation latency. Must be fast enough for real-time inference.
- **Implementation**: Group weights into vectors of length 8. Encode each vector using the E8 lattice (8-dimensional sphere packing). Each lattice point requires ~4 bits to index. Effective: 0.5 bits/weight. But E8 codebook size is 240, so ~8 bits per 8-vector = 1 bit/weight. Too aggressive — use D4 lattice for ~2 bits/weight.

### 6.7 — QTIP / NestQuant (Trellis / E8 Lattice)

- **Description**: QTIP uses trellis coding to quantise weight matrices, treating each row as a sequence and using Viterbi decoding to find the minimum-distortion quantisation path. NestQuant extends this with nested lattices (E8 inside D16).
- **Expected BPB impact**: -0.010 to -0.018 (medium-high confidence)
- **Source**: QTIP (Tseng et al., 2024); NestQuant (2025)
- **Complexity**: Very high — trellis construction, Viterbi decoding
- **Competition precedent**: Used by top-3 submissions
- **Dependencies**: Custom CUDA/Metal kernels
- **Risk**: Implementation complexity is extreme. Dequant must be fast.
- **Implementation**: For each row of length d: construct a trellis with states representing lattice points. Edge weights = distortion between original weight and quantised value. Run Viterbi to find minimum-distortion path. Store path indices.

### 6.8 — tANS Entropy Coding (Replacing zstd)

- **Description**: Replace zstd-22 with a custom tANS (tabled Asymmetric Numeral Systems) coder trained on the model's weight distribution. tANS achieves near-entropy compression with O(1) decode per symbol, faster than zstd's LZ+Huffman.
- **Expected BPB impact**: -0.003 to -0.008 on artifact size (medium confidence)
- **Source**: ANS (Duda, 2009); tANS implementations
- **Complexity**: Medium — implement tANS encoder/decoder
- **Competition precedent**: Some submissions use custom entropy coding
- **Dependencies**: None
- **Risk**: Must include tANS decoder in artifact. Small overhead.
- **Implementation**: Profile int6 weight distribution. Build 12-bit tANS table (4096 states). Encode weight stream. Store table (8KB) + encoded stream. Decode at load time.

### 6.9 — rANS Compression

- **Description**: Range-based ANS variant that processes multiple symbols per step. Can be vectorised for faster decoding than tANS. Better suited for GPU decompression.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: rANS (Duda, 2009); Fabian Giesen's implementations
- **Complexity**: Medium
- **Competition precedent**: Some submissions
- **Dependencies**: None
- **Risk**: Same as tANS
- **Implementation**: Use 32-bit rANS state. Encode weights in reverse order. Decode forward. Can interleave 4-8 rANS streams for SIMD decode.

### 6.10 — Monarch Matrices for FFN Layers

- **Description**: Replace dense MLP weight matrices with Monarch matrix factorisation: `W = P1 @ diag(B1) @ P2 @ diag(B2)` where P1, P2 are fixed permutations and B1, B2 are block-diagonal matrices. This reduces parameters from O(n^2) to O(n * sqrt(n)) while preserving expressiveness.
- **Expected BPB impact**: -0.005 to -0.010 net (medium confidence)
- **Source**: Monarch Matrices (Dao et al., 2022)
- **Complexity**: Medium — restructure MLP layers
- **Competition precedent**: Not widely reported
- **Dependencies**: None
- **Risk**: May reduce model quality. Must verify BPB vs parameter savings.
- **Implementation**: For MLP fc layer (512 x 1536): factorise as two block-diagonal matrices with block size 32. Parameters: 2 * (512/32) * 32 * (1536/32) * 32 = 2 * 16 * 48 * 1024 = 1.57M (vs 786K for dense). Wait — this is worse. Need different block sizes. Use sqrt(512) ≈ 23, round to 16. Each block: 16x96. 32 blocks. Total: 2 * 32 * 16 * 96 = 98K params. Much smaller but likely less expressive.

### 6.11 — Kronecker-Factored Weights

- **Description**: Represent weight matrices as Kronecker products: `W = A kron B` where A is (p x q) and B is (r x s) with p*r = rows, q*s = cols. This compresses a (512 x 1536) matrix from 786K to e.g. (32 x 48) and (16 x 32) = 1536 + 512 = 2048 params. Extreme compression but severe quality loss.
- **Expected BPB impact**: Likely negative (quality loss > compression gain at this scale)
- **Source**: Kronecker factorisation literature
- **Complexity**: Medium
- **Competition precedent**: Not reported
- **Dependencies**: None
- **Risk**: Too aggressive — single Kronecker product is too low-rank. Sum of K Kronecker products may work.
- **Implementation**: `W = sum_{k=1}^{K} A_k kron B_k`. With K=8: total params = 8 * (2048) = 16K params per layer. Quality likely poor. Better as initialisation + fine-tuning.

### 6.12 — Low-Rank Factorisation

- **Description**: Replace weight matrices with low-rank approximation: `W ≈ U @ V` where U is (m x r) and V is (r x n). For MLP (512 x 1536) with rank 128: 512*128 + 128*1536 = 262K params (vs 786K). 66% compression with controlled quality loss.
- **Expected BPB impact**: -0.003 to -0.008 net (medium confidence)
- **Source**: Low-rank approximation; LoRA literature
- **Complexity**: Low — SVD decomposition of trained weights
- **Competition precedent**: Some submissions use low-rank for specific layers
- **Dependencies**: None
- **Risk**: Rank selection is critical. Too low -> quality collapse. Too high -> no savings.
- **Implementation**: After training, for each weight matrix: compute SVD `W = U @ diag(s) @ V.T`. Keep top-r singular values. Store `U_r = U[:, :r] @ diag(sqrt(s[:r]))` and `V_r = diag(sqrt(s[:r])) @ V[:r, :]`. Sweep r to find optimal quality/size tradeoff.

### 6.13 — BitNet 1.58-Bit Ternary

- **Description**: Quantise all weights to {-1, 0, +1} with per-row scaling. At 1.58 bits/param, a 28M model fits in ~5.5MB, leaving massive headroom for additional parameters or auxiliary models.
- **Expected BPB impact**: Likely net negative — see Dead Ends. At 28M params, ternary is catastrophic.
- **Source**: BitNet (Ma et al., 2024)
- **Complexity**: Medium
- **Competition precedent**: Investigated and rejected
- **Dependencies**: QAT from scratch (can't post-train quantise to ternary)
- **Risk**: Catastrophic quality loss at 28M scale. Need 100M+ params for ternary to work.
- **Implementation**: Would need complete retraining with ternary quantisation. Not recommended.

### 6.14 — Self-Predicting Weight Decompression

- **Description**: Use the model itself to predict its own weights. Store a compressed "seed" of weights and use the model's own forward pass to reconstruct the full weight matrices. This is a form of neural weight compression.
- **Expected BPB impact**: -0.005 to -0.015 (low confidence — highly speculative)
- **Source**: Self-referential weight matrices; neural compression
- **Complexity**: Very high
- **Competition precedent**: None known
- **Dependencies**: Bootstrap problem — need some weights to predict other weights
- **Risk**: Circular dependency. Error propagation. Very experimental.
- **Implementation**: Train a small "seed" network (2-3 layers, ~2MB). Use it to generate weights for the remaining 8 layers. Store only seed + quantised residuals.

### 6.15 — Neural Weight Compression Codec

- **Description**: Train a small autoencoder to compress and decompress weight matrices. The encoder maps weight matrices to a compact latent code; the decoder reconstructs them. Ship the decoder + latent codes in the artifact.
- **Expected BPB impact**: -0.005 to -0.010 (low-medium confidence)
- **Source**: Learned compression; neural codec literature
- **Complexity**: High — train separate compression model
- **Competition precedent**: None known
- **Dependencies**: Compression model must be small (<0.5MB)
- **Risk**: Compression model adds overhead. Training the compressor is a separate task.
- **Implementation**: Train a 3-layer MLP autoencoder on weight matrix rows. Encoder: 512->64->16 (latent). Decoder: 16->64->512. Ship decoder (512*64 + 64*16 + 16*64 + 64*512 = ~66K params, ~66KB) + latent codes.

### 6.16 — ZipNN Exponent Separation

- **Description**: For floating-point weights, separate exponent and mantissa bits before compression. Exponents tend to be highly clustered (small range), making them very compressible. Mantissa bits are more uniformly distributed.
- **Expected BPB impact**: -0.002 to -0.005 (medium confidence)
- **Source**: ZipNN (2024)
- **Complexity**: Low — bit manipulation before compression
- **Competition precedent**: Not reported
- **Dependencies**: Applies to fp16 weights (embeddings, control tensors)
- **Risk**: Minimal — only affects fp16 components
- **Implementation**: For fp16 tensors: separate into sign (1 bit), exponent (5 bits), mantissa (10 bits) arrays. Compress each separately with zstd. Recombine on decompression.

### 6.17 — CERWU Rate-Distortion Optimal Quantisation

- **Description**: Channel-wise Entropy-Regularised Weight Update. Formulate quantisation as a rate-distortion optimisation problem: minimise distortion (quantisation error) subject to a rate (bits) constraint. Uses Lagrangian relaxation to find the optimal bit allocation per channel.
- **Expected BPB impact**: -0.005 to -0.010 (medium confidence)
- **Source**: CERWU (2025); rate-distortion theory
- **Complexity**: Medium-high
- **Competition precedent**: Not widely reported
- **Dependencies**: None
- **Risk**: Optimisation adds time to export step.
- **Implementation**: For each weight matrix: compute per-column sensitivity (gradient magnitude). Solve: `min sum_j D_j(b_j) s.t. sum_j b_j <= B` where D_j is distortion of column j at b_j bits. Use dynamic programming or Lagrangian method.

### 6.18 — Bitmask-LZMA

- **Description**: Compress quantised weights using LZMA instead of zstd. LZMA typically achieves 5-15% better compression than zstd at the cost of slower decompression. For a one-time model load, decompression speed is less critical.
- **Expected BPB impact**: -0.001 to -0.003 on artifact size (medium confidence)
- **Source**: LZMA (Pavlov, 2004)
- **Complexity**: Trivial — swap compressor
- **Competition precedent**: Some submissions use LZMA
- **Dependencies**: None
- **Risk**: Slower decompression. Must verify within eval startup budget.
- **Implementation**: Replace `zstandard.compress(data, level=22)` with `lzma.compress(data, preset=9)`. Test decompression time.

### 6.19 — YAQA Adaptive Rounding

- **Description**: Yet Another Quantisation Algorithm. Instead of round-to-nearest, use adaptive rounding that considers reconstruction error across the entire layer. For each weight, choose floor or ceil to minimise the layer's output error (similar to AdaRound).
- **Expected BPB impact**: -0.003 to -0.006 (medium-high confidence)
- **Source**: AdaRound (Nagel et al., 2020); YAQA
- **Complexity**: Medium — requires calibration data and per-weight optimisation
- **Competition precedent**: Advanced quantisation submissions
- **Dependencies**: Calibration dataset (use validation data)
- **Risk**: Optimisation time. Use 1000 calibration samples.
- **Implementation**: For each layer, for each weight, compute `v = sigmoid(alpha)` where alpha is optimised to minimise `||Wx - Q(W)x||_2` over calibration data. Rounding decision: `floor(w/s) + (v > 0.5)`. Optimise alpha with 200 steps of Adam.

### 6.20 — Binary Asymmetric U-Net

- **Description**: Use different quantisation bit-widths for encoder and decoder halves of the U-Net. Encoder layers (which feed into skip connections) need higher precision; decoder layers can tolerate lower precision because skip connections provide corrective signal.
- **Expected BPB impact**: -0.002 to -0.005 (medium confidence)
- **Source**: Architecture-aware quantisation
- **Complexity**: Low — different bit width per layer
- **Competition precedent**: Not reported
- **Dependencies**: None
- **Risk**: Decoder degradation may not be compensated by skip connections.
- **Implementation**: Encoder layers 0-4: int6 (6 bits). Decoder layers 5-10: int5 (5 bits). Skip connections + embeddings: fp16.

---

## Phase 7: Tokenisation (Days)

### 7.1 — 2048-Token Picky BPE Optimised for FineWeb

- **Description**: Increase vocabulary from 1024 to 2048 tokens, selecting tokens specifically to maximise compression on FineWeb statistics. Each token added must "earn its keep" by reducing BPB more than the parameter cost of the extra embedding row.
- **Expected BPB impact**: -0.010 to -0.020 (medium-high confidence)
- **Source**: Optimal vocabulary selection literature; competition discussions
- **Complexity**: Medium — retrain SentencePiece with modified scoring
- **Competition precedent**: Top submissions use vocab sizes 1024-4096
- **Dependencies**: Must retrain model with new tokeniser
- **Risk**: Larger vocab = larger embedding table. At 2048 x 512 = 1M params (int6: ~750KB). Currently 1024 x 512 = 524K params (fp16: ~1MB). Actually the embedding is kept in fp16, so 2048 vocab costs 2MB vs 1MB — a 1MB penalty. Must verify net benefit.
- **Implementation**: Train SentencePiece with `--vocab_size=2048 --model_type=bpe --input=fineweb_sample.txt`. Score each token by its frequency-weighted byte savings. Remove tokens that don't save enough bytes.

### 7.2 — Gravity Tokenizer (Ablation Leverage Scoring)

- **Description**: Score each token in the vocabulary by its "gravitational pull" — how much removing it would hurt compression. Tokens with low gravity are candidates for removal; tokens with high gravity justify their embedding cost. Use this to prune the vocabulary to the optimal size.
- **Expected BPB impact**: -0.003 to -0.008 (medium confidence)
- **Source**: Gravity Tokenizer (2025); vocabulary pruning literature
- **Complexity**: Medium — requires ablation study per token
- **Competition precedent**: Not widely reported
- **Dependencies**: None
- **Risk**: Ablation study is expensive (one model eval per token candidate).
- **Implementation**: For each token t: retokenise validation set without t, evaluate BPB change. Tokens with smallest BPB increase are prunable. This is O(V * eval_cost) — use sampling to reduce cost.

### 7.3 — Successive Abstraction (Suffix-Trie OOV Handling)

- **Description**: Handle out-of-vocabulary bytes using a suffix trie that decomposes unknown byte sequences into known sub-sequences. This avoids the "unknown token" penalty by ensuring every byte sequence has a valid tokenisation, even if it requires falling back to byte-level tokens.
- **Expected BPB impact**: -0.002 to -0.005 (low-medium confidence)
- **Source**: Successive Abstraction (2024)
- **Complexity**: Medium
- **Competition precedent**: Not reported
- **Dependencies**: Custom tokeniser
- **Risk**: Adds complexity to tokenisation/detokenisation pipeline.
- **Implementation**: Build suffix trie from vocabulary. For each input byte sequence, greedily match longest prefix in trie. Fall back to byte-level tokens for unmatched suffixes. This is essentially what SentencePiece already does, but with explicit trie structure for better OOV handling.

### 7.4 — Byte-Level Sub-Model (1MB Alongside Main Model)

- **Description**: Ship a tiny byte-level model (~1MB, ~100K params) alongside the main token-level model. The byte model handles tokens that the main model struggles with (high-entropy byte sequences, code, URLs, etc.). Switch between models based on content type detection.
- **Expected BPB impact**: -0.005 to -0.015 (medium confidence)
- **Source**: Byte-level language modelling; hybrid tokenisation
- **Complexity**: High — train and ship second model, content-type detection
- **Competition precedent**: At least 1 submission uses byte-level auxiliary
- **Dependencies**: Content type detection (simple heuristic: ASCII ratio, entropy)
- **Risk**: 1MB is significant budget. Must verify improvement justifies cost.
- **Implementation**: Train a 3-layer, 128-dim byte-level transformer (~130K params, ~100KB at int6+zstd). At eval: if byte entropy > threshold, use byte model for that span. Mix with token model predictions.

### 7.5 — Custom SentencePiece Trained on FineWeb Statistics

- **Description**: Train a new SentencePiece model directly on a large sample of FineWeb, with coverage optimised for the validation set distribution. Use `--character_coverage=0.9999` and `--train_extremely_large_corpus=true`.
- **Expected BPB impact**: -0.005 to -0.010 (medium confidence)
- **Source**: Standard practice; SentencePiece documentation
- **Complexity**: Low — retrain SentencePiece
- **Competition precedent**: Universal
- **Dependencies**: Large FineWeb sample for training
- **Risk**: Overfitting to training distribution; may not generalise to val set.
- **Implementation**: `spm.SentencePieceTrainer.train(input='fineweb_100M_sample.txt', model_prefix='fineweb_1024_v2', vocab_size=1024, model_type='bpe', byte_fallback=True, character_coverage=0.9999)`.

---

## Phase 8: Wild Ideas (Weeks)

### 8.1 — Hypernetworks (Generate Weights from Compressed Embeddings)

- **Description**: Instead of storing all weights directly, store a small hypernetwork that generates the main model's weights from a compressed code. The hypernetwork takes a layer index and position code as input and outputs the weight matrix for that position. If the hypernetwork is 2MB, it can potentially generate 25MB of weights.
- **Expected BPB impact**: Unknown — potentially -0.010 to -0.030 (low confidence)
- **Source**: Hypernetworks (Ha et al., 2017); Neural Weight Generation
- **Complexity**: Very high
- **Competition precedent**: None known
- **Dependencies**: Complete architecture redesign
- **Risk**: Quality of generated weights is unknown. Hypernetwork training is unstable.
- **Implementation**: Hypernetwork: 3-layer MLP (256->512->512). Input: layer_idx (one-hot, 11-dim) + position_code (learned, 32-dim). Output: weight row (512-dim). Generate all weight matrices at load time. Store hypernetwork + position codes.

### 8.2 — Self-Referential Weight Matrix (SRWM)

- **Description**: The model's weights are partially determined by its own activations on a "self-reference" input. At init, feed the model a special token sequence; the resulting hidden states are used to construct some weight matrices. This creates a self-referential loop where the model defines part of its own structure.
- **Expected BPB impact**: Unknown (low confidence)
- **Source**: SRWM (Schmidhuber, 1993); self-modifying networks
- **Complexity**: Very high
- **Competition precedent**: None
- **Dependencies**: Careful bootstrapping to avoid degenerate fixed points
- **Risk**: Training instability. Degenerate self-referential loops. Academic curiosity.
- **Implementation**: Not recommended for competition. Academic exploration only.

### 8.3 — In-Place TTT (Fast Weights in MLP Projection)

- **Description**: Instead of running separate TTT passes, modify the MLP to accumulate fast weights during the forward pass itself. Each token's hidden state contributes to a running outer product that modifies the MLP projection matrix in-place. This is TTT without the separate adaptation step.
- **Expected BPB impact**: -0.005 to -0.015 (medium confidence)
- **Source**: Fast Weights (Ba et al., 2016); TTT Layers (Sun et al., 2024)
- **Complexity**: Medium-high
- **Competition precedent**: TTT layers paper explores this concept
- **Dependencies**: None
- **Risk**: Accumulating outer products is O(d^2) per token per layer. Memory cost.
- **Implementation**: In MLP forward: `self.fast_W += lr * x.unsqueeze(-1) @ target.unsqueeze(-2)`. Use `(W + fast_W) @ x` instead of `W @ x`. Reset fast_W at document boundaries.

### 8.4 — Implicit Neural Representation for Weights (SIREN)

- **Description**: Represent weight matrices as a continuous function parameterised by SIREN (Sinusoidal Representation Network). Instead of storing a discrete matrix, store the SIREN parameters and evaluate the function at integer grid points to reconstruct the weight matrix. SIREN's periodic activations capture weight structure efficiently.
- **Expected BPB impact**: -0.005 to -0.015 (low confidence)
- **Source**: SIREN (Sitzmann et al., 2020); INR for model compression
- **Complexity**: Very high
- **Competition precedent**: None known
- **Dependencies**: SIREN fitting pipeline
- **Risk**: SIREN fitting is itself an optimisation problem. Reconstruction quality is uncertain.
- **Implementation**: For each weight matrix (512 x 1536): train a 3-layer SIREN (128-dim hidden) to map (row, col) coordinates to weight values. SIREN params: ~50K per weight matrix. If model has 22 weight matrices: ~1.1M SIREN params total. Potential 20:1 compression if fitting quality is high.

### 8.5 — Compressive Context via Learned Summarisation

- **Description**: Instead of sliding window attention, compress distant context into a fixed-size "summary" vector using a learned compression module. This allows effective context lengths far beyond the training sequence length without increasing compute.
- **Expected BPB impact**: -0.005 to -0.015 (medium confidence)
- **Source**: Compressive Transformer (Rae et al., 2020); Memorizing Transformers
- **Complexity**: High — additional compression module, modified attention
- **Competition precedent**: Not reported
- **Dependencies**: Additional parameters for compression module
- **Risk**: Compression module adds parameters. Quality of compressed context is uncertain.
- **Implementation**: Add a small MLP that maps sequences of K=64 tokens to a single summary vector. During eval, maintain a buffer of summary vectors for past context. Attend to summaries + current window.

### 8.6 — Arithmetic Coding Integration for Weight Storage

- **Description**: Use arithmetic coding (instead of zstd or ANS) for maximum-entropy weight compression. Arithmetic coding achieves the information-theoretic lower bound on code length for any symbol distribution.
- **Expected BPB impact**: -0.002 to -0.005 on artifact size (medium confidence)
- **Source**: Arithmetic coding (Witten et al., 1987)
- **Complexity**: Medium — implement arithmetic encoder/decoder
- **Competition precedent**: Some submissions
- **Dependencies**: None
- **Risk**: Slower than ANS. Must verify decompression speed.
- **Implementation**: Use `arithmeticcoding` Python library or implement from scratch. Encode int6 weight stream with adaptive probability model. Achieves ~0.05 bits/symbol better than tANS in theory.

### 8.7 — Ensemble of Tiny Routed Models

- **Description**: Instead of one 16MB model, train 4 tiny 4MB models, each specialised for different content types (prose, code, structured data, conversational). Route each document to the best model using a simple classifier. The routing classifier is ~50KB.
- **Expected BPB impact**: -0.010 to -0.025 (medium confidence)
- **Source**: Mixture of Experts; specialised model ensembles
- **Complexity**: High — train 4 models, router, packaging
- **Competition precedent**: At least 1 submission uses model ensembles
- **Dependencies**: Content type classifier
- **Risk**: Uneven content distribution in FineWeb may leave some models under-trained. Routing errors are catastrophic.
- **Implementation**: Classify FineWeb docs into 4 clusters by topic (K-means on TF-IDF). Train 4 separate models. Ship all 4 + router. At eval: classify document, route to best model.

### 8.8 — V:N:M Structured Sparsity with Sparse Matmul

- **Description**: Apply V:N:M structured sparsity (e.g., 2:4 — keep 2 of every 4 weights) to MLP layers. H100 has hardware support for 2:4 sparse matrix multiplication, which runs at 2x the throughput of dense matmul. This allows doubling the MLP size within the same compute budget.
- **Expected BPB impact**: -0.005 to -0.015 (medium confidence)
- **Source**: NVIDIA Sparse Tensor Cores; 2:4 sparsity literature
- **Complexity**: Medium — sparsity masks, sparse matmul kernels
- **Competition precedent**: Some submissions explore sparsity
- **Dependencies**: H100 sparse tensor core support in PyTorch
- **Risk**: 50% sparsity hurts quality. Must train with sparsity from scratch (gradual pruning).
- **Implementation**: During warmdown, prune each group of 4 weights to keep the 2 with largest magnitude. Use `torch.nn.utils.prune.ln_structured` or custom implementation. At inference: use `torch.sparse.mm` with 2:4 format.

### 8.9 — V:N:M Activation Sparsity Exploitation

- **Description**: LeakyReLU(0.9)² creates activation sparsity (small values get crushed by squaring). Exploit this by skipping computation for near-zero activations. This isn't structured sparsity — it's dynamic, input-dependent sparsity.
- **Expected BPB impact**: -0.003 to -0.008 (low-medium confidence — indirect via speed)
- **Source**: Dynamic sparsity; ReLU sparsity exploitation
- **Complexity**: Medium — custom sparse matmul kernel
- **Competition precedent**: Not reported
- **Dependencies**: Custom CUDA kernel
- **Risk**: Activation sparsity varies by input. May not be consistently exploitable.
- **Implementation**: After LeakyReLU²: threshold activations below eps=1e-4 to zero. Use sparse matmul for the projection layer. Effective speedup depends on actual sparsity ratio.

---

## Phase 9: Hardware and Deployment

### 9.1 — H100 Deployment via RunPod

- **Description**: Deploy training on 8xH100 SXM via RunPod or similar cloud provider. The competition scoring hardware is 8xH100, so training must be validated on this exact configuration.
- **Expected BPB impact**: -0.10 to -0.15 vs M4 Max (high confidence — from 7x more training steps)
- **Source**: Competition rules
- **Complexity**: Low — launch cloud instance, run training script
- **Competition precedent**: Required
- **Dependencies**: RunPod account, GPU availability
- **Risk**: Cost ($24/hr for 8xH100). Training bugs that only appear on multi-GPU.
- **Implementation**: `runpodctl create pod --gpu 8 --gpu-type H100_SXM --image pytorch/pytorch:2.5-cuda12.4-cudnn9-devel`. Upload code, run `torchrun --nproc_per_node=8 train_gpt.py`.

### 9.2 — Parallel Muon for Multi-GPU

- **Description**: Current Muon implementation distributes parameters across GPUs (parameter `i` assigned to GPU `i % world_size`). Verify this scaling is optimal. Consider: all-gather gradients before Newton-Schulz vs current approach of distributing and all-reducing.
- **Expected BPB impact**: Indirect (training speed -> more steps -> better BPB)
- **Source**: Current implementation in train_gpt.py
- **Complexity**: Low — profiling and tuning
- **Competition precedent**: Universal
- **Dependencies**: Multi-GPU access
- **Risk**: None
- **Implementation**: Profile with `torch.cuda.Event` to identify bottlenecks. Test all-gather vs current all-reduce pattern. Overlap Newton-Schulz computation with communication.

### 9.3 — MLX Zero-Copy Unified Memory Programming

- **Description**: Optimise the MLX training script to exploit Apple Silicon's unified memory architecture. Avoid unnecessary copies between CPU and GPU memory. Use `mx.eval()` strategically to control graph materialisation.
- **Expected BPB impact**: Indirect (faster local iteration -> more experiments)
- **Source**: MLX documentation; Apple Silicon architecture
- **Complexity**: Low-medium
- **Competition precedent**: N/A (local dev only)
- **Dependencies**: M4 Max hardware
- **Risk**: None
- **Implementation**: Profile with `mx.metal.device_info()`. Ensure no hidden copies. Use `mx.stream(mx.gpu)` for all compute. Batch `mx.eval()` calls.

### 9.4 — Custom Metal Shading Language Megakernels

- **Description**: Write fused MSL kernels for common operation sequences: dequant+matmul, RMSNorm+attention, etc. Avoids kernel launch overhead and memory traffic between operations.
- **Expected BPB impact**: Indirect (faster eval -> more time for TTT/n-gram processing)
- **Source**: Metal Performance Shaders documentation
- **Complexity**: High — MSL programming
- **Competition precedent**: N/A (local dev only; competition uses CUDA)
- **Dependencies**: Metal API knowledge
- **Risk**: MSL debugging is painful. Diminishing returns if MLX already fuses well.
- **Implementation**: Write fused `dequant_int6_matmul` kernel in MSL. Register as custom MLX operation.

### 9.5 — Fused Dequant+Attention CUDA Kernel

- **Description**: For H100 deployment, write a fused CUDA kernel that dequantises int6 weights and performs attention in a single kernel launch. Eliminates memory round-trip for dequantised weights.
- **Expected BPB impact**: Indirect (faster eval)
- **Source**: FlashAttention codebase; CUTLASS
- **Complexity**: Very high — CUDA kernel programming
- **Competition precedent**: Advanced submissions
- **Dependencies**: CUDA expertise
- **Risk**: Extreme implementation complexity. Bugs hard to debug.
- **Implementation**: Extend FlashAttention kernel to accept int6 weights + scales. Dequantise in registers before matmul. Use CUTLASS for template-based kernel generation.

---

## Cumulative Projection

Estimated BPB progression assuming successful implementation of high-confidence techniques in each phase. All estimates are cumulative from our current 1.1458 baseline.

| Phase | Key Additions | Est. BPB | Delta | Confidence |
|-------|--------------|----------|-------|------------|
| **1. Foundation** (done) | Current best | **1.1458** | — | Measured |
| **2. Immediate Wins** | 11 layers, C-Muon, tuned n-grams, adaptive stride | **1.110 - 1.125** | -0.020 to -0.035 | Medium-High |
| **3. Architecture** | Differential Attention, VRL, depth recurrence (2x), Partial RoPE | **1.080 - 1.105** | -0.020 to -0.045 | Medium |
| **4. Training** | Late QAT+STE, Soft-Round, MAML-TTT, RPS | **1.060 - 1.090** | -0.015 to -0.030 | Medium |
| **5. Eval-Time** | Match model, PPMd, CTW, LoRA-TTT+Adam, logistic mixing | **1.020 - 1.060** | -0.030 to -0.050 | Medium |
| **6. Compression** | Hadamard rotation, GPTQ-lite, mixed int5/int6, tANS | **1.000 - 1.045** | -0.015 to -0.025 | Medium |
| **7. Tokenisation** | 2048-vocab Picky BPE | **0.990 - 1.035** | -0.010 to -0.020 | Medium |
| **8. Wild Ideas** | Best 2-3 that work | **0.970 - 1.020** | -0.010 to -0.020 | Low |
| **Theoretical floor** | Perfect model + perfect compression | **~0.75 - 0.85** | — | — |

**Key insight**: The largest remaining gains are in eval-time techniques (Phase 5), especially the combination of match model + PPMd + improved n-gram mixing. These are "free" in that they don't consume model parameter budget.

---

## Appendix: Dead Ends

### DE.1 — Full Depth Recurrence (3+ Loops)

- **Status**: Investigated and rejected
- **Reason**: Quantisation error amplifies exponentially with loop count. At int6, each loop multiplies the quantisation noise by the model's gain factor (~30x). At 2 loops: 900x amplification is manageable (+0.003 BPB). At 3 loops: ~27,000x amplification is catastrophic (+0.08 BPB).
- **Source**: Internal ablations, March 2026
- **Recommendation**: Hard cap at 2 loops. Consider partial recurrence (recurse only middle layers).

### DE.2 — Mamba/SSM Hybrid (Hymba)

- **Status**: Investigated and rejected
- **Reason**: SSMs (Mamba, S4, etc.) save *compute* not *parameters*. In a parameter-limited setting, the SSM's state transition matrices consume the same parameter budget as attention weights but with worse quality at 28M scale. SSMs shine at >1B params where compute is the bottleneck.
- **Source**: Hymba (NVIDIA, 2024); internal analysis
- **Recommendation**: Only reconsider if parameter budget increases significantly.

### DE.3 — KAN (Kolmogorov-Arnold Networks)

- **Status**: Investigated and rejected
- **Reason**: KAN replaces MLPs with learned activation functions on edges. While theoretically more expressive per parameter, KAN underperforms standard MLPs on NLP tasks at this scale. The spline-based activation functions add parameter overhead without compensating quality improvement.
- **Source**: KAN (Liu et al., 2024); NLP benchmarks
- **Recommendation**: Not suitable for language modelling at 28M scale.

### DE.4 — Full BitNet at 16MB

- **Status**: Investigated and rejected
- **Reason**: BitNet (1.58-bit ternary weights) requires 100M+ parameters to match the quality of 6-bit models at 28M params. A ternary 28M-param model achieves ~1.8 BPB — far worse than int6.
- **Source**: BitNet scaling results; internal estimates
- **Recommendation**: Only viable if the artifact budget were 64MB+ (allowing 100M+ ternary params).

### DE.5 — MLA (Multi-Head Latent Attention)

- **Status**: Investigated and rejected
- **Reason**: MLA (from DeepSeek-V2) compresses KV cache into a low-rank latent space. This saves inference *memory*, not parameters. The latent projection matrices add parameters. At 28M scale, the overhead exceeds the savings.
- **Source**: DeepSeek-V2 (2024)
- **Recommendation**: Useful for KV cache compression at large scale; irrelevant here.

### DE.6 — SwiGLU

- **Status**: Investigated and rejected
- **Reason**: SwiGLU requires 3 weight matrices (gate, up, down) vs 2 for standard MLP (up, down). At fixed parameter budget, SwiGLU's hidden dimension must be 2/3 of the standard MLP's, cancelling the quality advantage. Our `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch` record confirmed no improvement.
- **Source**: Internal ablation; PaLM (2022) analysis of gate overhead
- **Recommendation**: Only beneficial when parameter budget is not binding.

### DE.7 — INT4 Quantisation

- **Status**: Investigated and rejected
- **Reason**: INT4 adds +0.065 BPB penalty over INT6 for attention weights. Even with QAT, the penalty is +0.035 BPB. The parameter savings do not compensate for the quality loss at 28M scale.
- **Source**: Internal ablation, March 2026
- **Recommendation**: Consider only for MLP layers in a mixed-precision scheme (see 6.1).

### DE.8 — Orthogonal Residuals

- **Status**: Investigated and rejected
- **Reason**: Constraining residual updates to be orthogonal to the residual stream (via Gram-Schmidt) added +0.05 BPB. The orthogonality constraint prevents the model from reinforcing features across layers.
- **Source**: Internal ablation
- **Recommendation**: Rejected. Standard additive residuals are better.

### DE.9 — MUD Naive Implementation

- **Status**: Partially rejected (naive version only)
- **Reason**: Naive MUD (Cholesky factorisation instead of Newton-Schulz) was 4.5x slower per step. The theoretical quality is identical, but the wall-clock cost is prohibitive within the 10-minute training budget. An optimised MUD implementation (custom CUDA kernel) might be viable.
- **Source**: Internal benchmarking
- **Recommendation**: Only revisit with custom CUDA kernel. Otherwise, stick with Newton-Schulz Muon.

---

## Priority Order (Recommended Implementation Sequence)

1. **2.2** — 11 layers (trivial, high confidence)
2. **2.1** — BiggerBigramHash + enable TrigramHash (trivial)
3. **2.3** — Cautious Muon (one line)
4. **4.1** — Late QAT with STE (high confidence, low complexity)
5. **3.6** — Differential Attention (medium complexity, high impact)
6. **3.2** — Value Residual Learning (low complexity)
7. **3.7** — Depth recurrence 2x (medium complexity, potentially highest single gain)
8. **5.1** — Match model (medium complexity, reliable gain)
9. **5.4** — PPMd auxiliary model (reliable gain)
10. **5.12** — LoRA-TTT with Adam (medium complexity)
11. **6.3** — Hadamard rotation before quant (medium complexity, reliable gain)
12. **5.6** — Logistic-domain mixing (low complexity)
13. **5.17** — Pre-computed n-gram table (medium complexity, high impact)
14. **9.1** — H100 deployment (required for submission)
15. Everything else in priority order of (expected_impact * confidence / complexity)

---

*This document is a living roadmap. Update after each experiment with measured BPB results.*
