"""
modules_training.py — Four targeted diffs porting MLX improvements to train_gpt.py.

Each section shows the ORIGINAL lines from train_gpt.py (with line numbers),
followed by the REPLACEMENT code.  Apply in order; changes are independent.
"""

# ============================================================================
# 1. CAUTIOUS MUON  (train_gpt.py lines 142-156, inside Muon.step())
# ============================================================================
#
# ORIGINAL (lines 149-155):
#
#     buf = state["momentum_buffer"]
#     buf.mul_(momentum).add_(g)
#     if nesterov:
#         g = g.add(buf, alpha=momentum)
#     g = zeropower_via_newtonschulz5(g, steps=backend_steps)
#     # Scale correction from Muon reference implementations.
#     g *= max(1, g.size(0) / g.size(1)) ** 0.5
#
# REPLACEMENT:
#
#     buf = state["momentum_buffer"]
#     buf.mul_(momentum).add_(g)
#     # Cautious Muon: mask out momentum that conflicts with gradient direction.
#     buf_masked = buf * (buf * g > 0).float()
#     if nesterov:
#         g = g.add(buf_masked, alpha=momentum)
#     g = zeropower_via_newtonschulz5(g, steps=backend_steps)
#     # Scale correction from Muon reference implementations.
#     g *= max(1, g.size(0) / g.size(1)) ** 0.5
#
# Summary: After updating the momentum buffer, we create buf_masked that zeros
# out any momentum component whose sign disagrees with the current gradient.
# The Nesterov lookahead then uses buf_masked instead of raw buf.  This
# prevents the optimiser from overshooting when momentum and gradient conflict,
# which the MLX branch found is worth ~0.002 BPB.


# ============================================================================
# 2. LeakyReLU(0.9)^2  (train_gpt.py lines 606-617, MLP class)
# ============================================================================
#
# ORIGINAL (lines 615-617):
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.relu(self.fc(x))
#         return self.proj(x.square())
#
# REPLACEMENT:
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = F.leaky_relu(self.fc(x), negative_slope=0.9)
#         return self.proj(x.square())
#
# Summary: Swapping ReLU for LeakyReLU(0.9) keeps the squared activation
# pathway but allows the negative half-plane to contribute a scaled signal.
# This eliminates dead neurons entirely.  The MLX branch measured a 0.013 BPB
# improvement over the previous negative_slope=0.5 setting.
# Requires: torch.nn.functional already imported as F (line 26).


# ============================================================================
# 3. zstd-22 COMPRESSION  (train_gpt.py lines 19, 1080, 1089, 1098)
# ============================================================================
#
# ORIGINAL import (line 19):
#
#     import zlib
#
# REPLACEMENT import (after line 19):
#
#     import zlib
#     try:
#         import zstandard
#         _COMPRESSOR = "zstd"
#     except ImportError:
#         _COMPRESSOR = "zlib"
#
# ---
#
# ORIGINAL compression (line 1080):
#
#     quant_blob = zlib.compress(quant_raw, level=9)
#
# REPLACEMENT compression:
#
#     if _COMPRESSOR == "zstd":
#         quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
#     else:
#         quant_blob = zlib.compress(quant_raw, level=9)
#
# ---
#
# ORIGINAL log message (line 1089):
#
#         log0(
#             f"Serialized model int8+zlib: {quant_file_bytes} bytes "
#             ...
#         )
#
# REPLACEMENT log message:
#
#         log0(
#             f"Serialized model int8+{_COMPRESSOR}: {quant_file_bytes} bytes "
#             ...
#         )
#
# ---
#
# ORIGINAL decompression (line 1098):
#
#     quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
#
# REPLACEMENT decompression:
#
#     if _COMPRESSOR == "zstd":
#         quant_decompressed = zstandard.ZstdDecompressor().decompress(quant_blob_disk)
#     else:
#         quant_decompressed = zlib.decompress(quant_blob_disk)
#     quant_state = torch.load(io.BytesIO(quant_decompressed), map_location="cpu")
#
# Summary: zstd at level 22 achieves ~5-8% better compression than zlib-9 on
# quantised int8 weight blobs, shaving bytes off the 16 MB submission cap.
# Falls back to zlib transparently if zstandard is not installed.


# ============================================================================
# 4. DUAL-PATH lr_mul  (train_gpt.py lines 924-933, lr_mul function)
# ============================================================================
#
# ORIGINAL (lines 924-933):
#
#     def lr_mul(step: int, elapsed_ms: float) -> float:
#         if args.warmdown_iters <= 0:
#             return 1.0
#         if max_wallclock_ms is None:
#             warmdown_start = max(args.iterations - args.warmdown_iters, 0)
#             return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
#         step_ms = elapsed_ms / max(step, 1)
#         warmdown_ms = args.warmdown_iters * step_ms
#         remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
#         return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
#
# REPLACEMENT:
#
#     def lr_mul(step: int, elapsed_ms: float) -> float:
#         if args.warmdown_iters <= 0:
#             return 1.0
#         # Iteration-based warmdown
#         iter_scale = 1.0
#         warmdown_start = max(args.iterations - args.warmdown_iters, 0)
#         if warmdown_start <= step < args.iterations:
#             iter_scale = max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
#         elif step >= args.iterations:
#             iter_scale = 0.0
#         if max_wallclock_ms is None:
#             return iter_scale
#         # Wallclock-based warmdown
#         step_ms = elapsed_ms / max(step, 1)
#         warmdown_ms = args.warmdown_iters * step_ms
#         remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
#         wall_scale = remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
#         # Use whichever warmdown is further along (lower value), so SWA triggers
#         # regardless of whether the run is iteration-limited or wallclock-limited
#         return min(iter_scale, wall_scale)
#
# Summary: The original code uses an either/or branch — if max_wallclock_ms is
# set it ONLY uses wallclock warmdown, otherwise ONLY iteration warmdown.  The
# MLX branch computes both and takes the minimum.  This ensures LR ramps down
# correctly (and SWA kicks in on time) no matter which limit the run hits first.
