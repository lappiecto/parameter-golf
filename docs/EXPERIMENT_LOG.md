# Parameter Golf -- Complete Experiment Audit Trail

Generated: 2026-03-26
Source data: `logs/*.txt`, `/tmp/*.log`, `dashboard/experiments.json`
Platform: Apple M4 Max / 128 GB / MLX 0.31.1 / Python 3.13.2

---

## Summary of All Runs (Chronological)

| # | Run ID | Date | Layers | Dim | MLP | Seq | Iters | Steps Done | Params | SmearGate | Bigram | Trigram | Muon Mom | matrix_lr | grad_clip | SWA | EMA | Quant | Pre-Q val_bpb | Post-Q val_bpb | Artifact (bytes) | Elapsed | Status |
|---|--------|------|--------|-----|-----|-----|-------|------------|--------|-----------|--------|---------|----------|-----------|-----------|-----|-----|-------|---------------|----------------|-------------------|---------|--------|
| 1 | mlx_smoke | 2026-03-22 ~09:00 | 9 | 512 | 2x | 1024 | 200 | 200/200 | 17,059,912 | No | -- | -- | 0.95 | 0.04 | 0 | No | No | int8 | 2.4071 | 2.4077 | 11,258,337 | ~52s train + ~392s eval | Completed |
| 2 | baseline_1774174275 | 2026-03-22 ~12:17 | 9 | 512 | 2x | 1024 | 200 | 200/200 | 17,059,912 | No | -- | -- | 0.95 | 0.04 | 0 | No | No | int8 | -- | -- | -- | ~53s train, killed during val | Killed (SIGKILL during val) |
| 3 | baseline_1774178244 | 2026-03-22 ~13:04 | 9 | 512 | 2x | 1024 | 200 | 200/200 | 17,059,912 | No | -- | -- | 0.95 | 0.04 | 0 | No | No | int8 | -- | -- | -- | ~51s train, killed during val | Killed (exit -9, SIGKILL during val) |
| 4 | baseline_1774179869 | 2026-03-22 ~13:44 | 9 | 512 | 2x | 1024 | 200 | 200/200 | 17,059,912 | No | -- | -- | 0.95 | 0.04 | 0 | No | No | int8 | 2.4087 | 2.4087 | 11,259,453 | ~52s train + ~817s total | Completed |
| 5 | baseline_..._1774265496 | 2026-03-23 ~13:31 | 10 | 512 | 3x | 2048 | 500 | 0/500 | 24,140,368 | No | -- | -- | 0.99 | 0.02 | 0 | No | No | -- | -- | -- | -- | 0.4s | Crashed (microbatch_batch_size:0) |
| 6 | baseline_num_layers_1774265634 | 2026-03-23 ~13:34 | 10 | 512 | 3x | 2048 | 500 | ~20/500 | 24,140,368 | No | -- | -- | 0.99 | 0.02 | 0 | No | No | -- | -- | -- | -- | 12.3s | Failed (warmup only, no train steps logged) |
| 7 | baseline_num_layers_1774269207 | 2026-03-23 ~14:33 | 10 | 512 | 3x | 2048 | 500 | 500/500 | 24,140,368 | No | -- | -- | 0.99 | 0.02 | 0 | No | No | int8 | 1.9344 | 1.9364 | 14,483,310 | ~353s train + ~1363s total | Completed |
| 8 | advisor_grad_clip_1774273946 | 2026-03-23 ~15:52 | 10 | 512 | 3x | 2048 | 500 | 99/500 | 24,140,368 | No | -- | -- | 0.99 | 0.02 | 0.3 | No | No | int8 | 3.2387 | 3.2705 | 7,073,205 | ~604s train (wallclock stopped) + ~1558s total | Wallclock stopped |
| 9 | advisor_grad_clip_1774278020 | 2026-03-23 ~16:00 | 10 | 512 | 3x | 2048 | 500 | 500/500 | 24,140,368 | No | -- | -- | 0.99 | 0.02 | 0.3 | No | No | int8 | 1.8987 | 1.9048 | 11,966,487 | ~345s train + ~1309s total | Completed |
| 10 | advisor_grad_clip_1774290778 | 2026-03-23 ~20:33 | 10 | 512 | 3x | 2048 | 500 | 500/500 | 24,140,368 | No | -- | -- | 0.99 | 0.02 | 0.3 | No | No | int8 | 1.9030 | 1.9089 | 12,059,513 | ~331s train + ~1342s total | Completed |
| 11 | advisor_grad_clip_1774292267 | 2026-03-23 ~20:57 | 10 | 512 | 3x | 2048 | 500 | 500/500 | 24,140,368 | No | -- | -- | 0.99 | 0.02 | 0.3 | No | No | int8 | 1.9277 | 1.9343 | 11,764,903 | ~374s train + ~1383s total | Completed |
| 12 | sota_stack_1774299828 | 2026-03-23 ~23:04 | 10 | 512 | 3x | 2048 | 1000 | 1000/1000 | 24,730,705 | Yes | 4096x128 | -- | 0.99 | 0.02 | 0.3 | No | No | int8 | 1.6881 | 1.6893 | 16,708,061 | ~962s train + ~2156s total | Completed |
| 13 | sota_stack_1774313082 | 2026-03-24 ~02:45 | 10 | 512 | 3x | 2048 | 1000 | 700/1000 | 24,730,705 | Yes | 4096x128 | -- | 0.99 | 0.02 | 0.3 | No | No | -- | -- | -- | -- | ~682s | Killed (log truncated at step 700) |
| 14 | sota_stack_1774314889 | 2026-03-24 ~03:14 | 10 | 512 | 3x | 2048 | 3000 | 2400/3000 | 25,517,137 | Yes | 10240x128 | -- | 0.99 | 0.02 | 0.3 | Yes (frac 0.4, every 50) | No | mixed 6/6 | -- | -- | -- | ~2466s | Killed (log truncated at step 2400) |
| 15 | sota_stack_1774380145 | 2026-03-24 ~21:22 | 10 | 512 | 3x | 2048 | 3000 | 2100/3000 | 25,517,137 | Yes | 10240x128 | -- | 0.99 | 0.02 | 0.3 | Yes (frac 0.4, every 50) | No | mixed 6/6 | -- | -- | -- | ~2359s | Killed (log truncated at step 2100) |
| 16 | sota_full_3k | 2026-03-24 ~23:15 | 10 | 512 | 3x | 2048 | 3000 | 2000/3000 | 25,517,137 | Yes | 10240x128 | -- | 0.99 | 0.02 | 0.3 | Yes (frac 0.4, every 50) | No | mixed 6/6 | -- | -- | -- | ~2223s train | Crashed (RuntimeError: numpy dtype mismatch on serialisation) |
| 17 | sota_full_3k_v2 | 2026-03-25 ~08:34 | 10 | 512 | 3x | 2048 | 3000 | 3000/3000 | 25,517,137 | Yes | 10240x128 | -- | 0.99 | 0.02 | 0.3 | Yes (frac 0.4, every 50), 18 ckpts | Yes (0.997) | mixed 6/6 | 1.4922 | 1.4739 | 13,669,403 | ~3241s train + ~21194s eval | Completed |
| 18 | beast_0point9 | 2026-03-25 ~20:48 | 11 | 512 | 3x | 2048 | 3000 | 1850/3000 | 28,173,402 | Yes | 10240x128 | 4096x64 | 0.99 | 0.02 | 0.3 | Yes (frac 0.4, every 50), 26 ckpts | Yes (0.997) | mixed 5/6 | 1.5487 | 1.6215 | 8,064,297 | 7200s train (wallclock cap) + ~28790s eval | Wallclock stopped |
| 19 | ngram_beast | 2026-03-25 ~23:33 | 11 | 512 | 3x | 2048 | 3000 | 200/3000 | 28,173,402 | Yes | 10240x128 | -- | 0.99 | 0.02 | 0.3 | -- | Yes (0.997) | -- | -- | -- | -- | ~196s | Killed (log truncated at step 200) |
| 20 | ultimate | 2026-03-26 ~00:00 | 11 | 512 | 3x | 2048 | 3000 | 3000/3000 | 28,173,402 | Yes | 10240x128 | -- | 0.99 | 0.02 | 0.3 | Skipped (0 ckpts) | Yes (0.997) | mixed 6/6 | 1.6238 | -- (TTT eval stalled) | 11,248,555 | ~3098s train | Stalled (TTT eval at doc 1/50000) |
| 21 | ngram_only | 2026-03-26 ~01:50 | 11 | 512 | 3x | 2048 | 3000 | 3000/3000 | 28,173,402 | Yes | 10240x128 | -- | 0.99 | 0.02 | 0.3 | Skipped (0 ckpts) | No | mixed 6/6 | 1.6209 | 1.6895 | 15,617,065 | ~3093s train + ~24902s eval | Completed |
| 22 | clean_fixed | 2026-03-26 ~09:30 | 11 | 512 | 3x | 2048 | 3000 | 10/3000 | 28,173,402 | Yes | 10240x128 | -- | 0.99 | 0.02 | 0.3 | -- | No | -- | -- | -- | -- | ~10s | Killed (log truncated at step 10) |
| 23 | smasher | 2026-03-26 ~10:25 | 11 | 512 | 3x | 2048 | 5000 | 5000/5000 | 28,173,402 | Yes | 10240x128 | -- | 0.99 | 0.02 | 0.3 | Yes (frac 0.77, every 50), 23 ckpts | No | mixed 6/6 | 1.4351 | ~1.4257 (in progress) | 14,746,571 | ~6098s train + eval in progress | Eval in progress |
| 24 | definitive_20k | 2026-03-28 | 11 | 512 | 3x | 2048 | 20000 | 20000/20000 | 28,255,422 | Yes | 10240x128 | 4096x64, Quad 2048 | 0.99 | 0.02 | 0.3 | Yes (start 18850, every 50), 23 ckpts | No | mixed 6/6 + Late QAT + GPTQ clip | 1.4109 | **1.3471** | 11,575,498 | 21172s train + 24883s eval (12.79h total) | **Completed — NEW RECORD** |

**Notes on the table:**
- "Pre-Q val_bpb" = the final validation BPB measured with float weights (the `val_loss:X val_bpb:Y` line at end of training).
- "Post-Q val_bpb" = the `final_int8_zlib_roundtrip` BPB after quantisation, serialisation, decompression, and dequantisation. This is the official competition metric.
- "Artifact (bytes)" = the `serialized_model_int8_zlib` compressed file size.
- All runs used Muon+Adam optimiser, bfloat16 compute, tied embeddings, logit_softcap=30 (except run 10: softcap=29, run 11: qk_gain=1.9 + softcap=29).
- Runs 1-11 used 1 train shard. Runs 12 used 5 shards. Runs 13-23 used 20 shards. Run 24 used 80 shards (8B tokens).

---

## Detailed Per-Run Narratives

### Run 1: mlx_smoke (2026-03-22, ~09:59)

**Purpose:** Initial smoke test to verify the MLX training pipeline works end-to-end on M4 Max.

**Configuration:** Stock baseline -- 9 layers, dim 512, 2x MLP, seq_len 1024, 200 iterations, Muon momentum 0.95, matrix_lr 0.04, no grad clipping. Wallclock cap 600s. Single train shard (1/195).

**Results:**
- Training completed all 200 steps in 51s (255ms/step, 33,136 tok/s).
- Pre-quant val_bpb: 2.4071 (val_loss: 4.0643).
- Post-quant val_bpb: 2.4077 (val_loss: 4.0653). Quantisation degradation: +0.0006 bpb.
- Artifact size: 11,258,337 bytes (int8 zlib). Payload ratio: 3.91x.
- Total elapsed: ~443s (bulk spent on full-dataset validation at stride 1).

**Key observations:** Pipeline works. Quantisation degradation is negligible at int8. The baseline is far from competitive but establishes the floor. Validation takes ~7.5x longer than training because it evaluates the entire 62M-token val set.

---

### Run 2: baseline_1774174275 (2026-03-22, ~12:17)

**Purpose:** Second baseline run, identical config to smoke test. Appears to be a re-run to verify reproducibility.

**Configuration:** Identical to Run 1.

**Results:**
- Training completed 200/200 steps in ~53s (263ms/step, 31,021 tok/s).
- Final train_loss: 3.8996.
- Process killed (SIGKILL) during validation pass -- log ends mid-way through val_progress (47,275/60,568).
- No final val_bpb or artifact produced.

**Key observations:** Training itself is stable and reproducible. The kill during validation suggests external termination (user interrupt, OOM, or process manager). The long validation pass on the full dataset is a liability.

---

### Run 3: baseline_1774178244 (2026-03-22, ~13:04)

**Purpose:** Third baseline attempt, again identical config. Attempting to complete what Run 2 could not.

**Configuration:** Identical to Run 1.

**Results:**
- Training completed 200/200 steps in ~51s (254ms/step).
- Final train_loss: 3.9041.
- Process killed (exit code -9, SIGKILL) during validation -- log ends at val_progress 9,525/60,568.
- experiments.json records elapsed: 115.4s, status: failed.

**Key observations:** Killed even earlier during validation than Run 2. This is likely a memory pressure issue during the val pass, or the user terminated it. The val_batch_size of 8192 on a 62M-token val set creates very long evaluation passes.

---

### Run 4: baseline_1774179869 (2026-03-22, ~13:44)

**Purpose:** Fourth baseline attempt. This one finally completed the full pipeline including validation and serialisation.

**Configuration:** Identical to Runs 1-3.

**Results:**
- Training completed 200/200 steps in 52s.
- Pre-quant val_bpb: 2.4087 (val_loss: 4.0670).
- Post-quant val_bpb: 2.4087 (val_loss: 4.0670). Zero quantisation degradation.
- Artifact size: 11,259,453 bytes. Payload ratio: 3.91x.
- Total elapsed: 869.3s.

**Key observations:** Establishes the definitive baseline at **2.4087 bpb**. Virtually identical to the smoke test (2.4077), confirming reproducibility. The baseline model has 17M params at 9 layers. From here, the experiments begin optimising.

---

### Run 5: baseline_..._1774265496 (2026-03-23, ~13:31)

**Purpose:** First attempt to scale up -- 10 layers, 3x MLP, seq_len 2048, 500 iterations, Muon momentum 0.99, matrix_lr 0.02.

**Configuration:** num_layers=10, mlp_mult=3, train_seq_len=2048, iterations=500, muon_momentum=0.99, matrix_lr=0.02. train_batch_tokens=8192 (unchanged from baseline).

**Results:**
- Immediate crash after 0.4s. Zero training steps completed.
- Error: `microbatch_batch_size:0` -- the batch token budget (8192) was too small for the new seq_len (2048) after grad accumulation division, resulting in zero-length batches.

**Key observations:** Critical lesson: when doubling seq_len from 1024 to 2048, the batch token budget must increase proportionally. train_batch_tokens=8192 with 8 grad_accum steps gives microbatch_tokens=1024, which is less than seq_len=2048. This produces microbatch_batch_size=0 (floor division). The fix is trivial: increase train_batch_tokens.

---

### Run 6: baseline_num_layers_1774265634 (2026-03-23, ~13:34)

**Purpose:** Quick fix of Run 5 -- doubled train_batch_tokens to 16384 while keeping everything else the same.

**Configuration:** Same as Run 5 but train_batch_tokens=16384.

**Results:**
- Completed warmup (20/20 warmup steps logged) but no training steps appear in the log.
- experiments.json records elapsed: 12.3s, exit_code: 1.

**Key observations:** The warmup completed but training itself failed immediately. Likely a val_batch_size mismatch issue (val_batch_size was still 8192 while train used 16384). The specific error is not captured in the log -- the process may have exited during the first compiled step. The fix: also update val_batch_size to match.

---

### Run 7: baseline_num_layers_1774269207 (2026-03-23, ~14:33)

**Purpose:** Corrected version of Runs 5/6 -- val_batch_size bumped to 16384 to match train_batch_tokens.

**Configuration:** 10 layers, mlp_mult=3, seq_len=2048, 500 iters, train_batch_tokens=16384, val_batch_size=16384, muon_momentum=0.99, matrix_lr=0.02.

**Results:**
- All 500 steps completed in 353s (706ms/step, 22,768 tok/s).
- Pre-quant val_bpb: 1.9344 (val_loss: 3.2661).
- Post-quant val_bpb: 1.9364 (val_loss: 3.2695). Quantisation degradation: +0.0020 bpb.
- Artifact size: 14,483,310 bytes. Payload ratio: 3.93x.
- Total params: 24,140,368 (up from 17M).
- Total elapsed: 1362.8s.

**Key observations:** Massive improvement over baseline: **1.9364 vs 2.4087 bpb** (0.47 bpb gain). The combination of +1 layer, 3x MLP (vs 2x), doubled seq_len, higher Muon momentum (0.99 vs 0.95), and halved matrix_lr (0.02 vs 0.04) was transformative. This becomes the new base config for all subsequent runs. Step time increased from ~260ms to ~706ms due to larger model and longer sequences.

---

### Run 8: advisor_grad_clip_1774273946 (2026-03-23, ~15:52)

**Purpose:** Test gradient clipping (norm 0.3) and much larger batch size (train_batch_tokens=163840, a 10x increase).

**Configuration:** Same as Run 7 plus grad_clip_norm=0.3, train_batch_tokens=163840, val_batch_size=163840, warmdown_iters=3000.

**Results:**
- Only 99/500 steps completed before 600s wallclock cap.
- Step average: 6104ms/step -- extremely slow due to massive batch size (10 sequences of 2048 per microbatch).
- Pre-quant val_bpb: 3.2387 at step 99 (barely started training).
- Post-quant val_bpb: 3.2705. Far worse than baseline.
- Artifact size: 7,073,205 bytes.

**Key observations:** The 10x batch size increase was catastrophic. At 6.1s/step, only 99 steps fit in the 600s wallclock. The model barely moved from random initialisation. The small artifact size (7MB vs 14MB) likely reflects how little the model diverged from its initialised state (compresses better when weights are near-random). Lesson: batch size must be balanced against step budget within wallclock constraints. Grad clipping itself was not tested in isolation here.

---

### Run 9: advisor_grad_clip_1774278020 (2026-03-23, ~16:00)

**Purpose:** Revert batch size to 16384, keep grad_clip_norm=0.3 and warmdown_iters=3000. Isolate the effect of gradient clipping.

**Configuration:** Same as Run 7 plus grad_clip_norm=0.3, warmdown_iters=3000. Batch tokens back to 16384.

**Results:**
- All 500 steps completed in 345s (690ms/step).
- Pre-quant val_bpb: 1.8987 (val_loss: 3.2058).
- Post-quant val_bpb: 1.9048 (val_loss: 3.2162). Quantisation degradation: +0.0061 bpb.
- Artifact size: 11,966,487 bytes.
- Total elapsed: 1309.2s.

**Key observations:** Gradient clipping at 0.3 provides a small but real improvement: **1.9048 vs 1.9364 bpb** (-0.032 bpb). The warmdown_iters increase (1200 -> 3000) may also contribute. This becomes the new best non-SOTA config.

---

### Run 10: advisor_grad_clip_1774290778 (2026-03-23, ~20:33)

**Purpose:** Test logit_softcap=29 (reduced from 30). All else same as Run 9.

**Configuration:** Same as Run 9 but logit_softcap=29 (from experiments.json).

**Results:**
- All 500 steps completed in 331s (662ms/step).
- Pre-quant val_bpb: 1.9030.
- Post-quant val_bpb: 1.9089. Quantisation degradation: +0.0059 bpb.
- Artifact size: 12,059,513 bytes.

**Key observations:** logit_softcap reduction from 30 to 29 gave slightly worse results: 1.9089 vs 1.9048 (+0.004 bpb). The difference is marginal but the original softcap=30 is retained as better.

---

### Run 11: advisor_grad_clip_1774292267 (2026-03-23, ~20:57)

**Purpose:** Test qk_gain_init=1.9 (up from 1.5) combined with logit_softcap=29. Exploring attention scaling.

**Configuration:** Same as Run 10 but qk_gain_init=1.9 (from experiments.json).

**Results:**
- All 500 steps completed in 374s (747ms/step).
- Pre-quant val_bpb: 1.9277.
- Post-quant val_bpb: 1.9343. Quantisation degradation: +0.0066 bpb.
- Artifact size: 11,764,903 bytes.

**Key observations:** Higher qk_gain_init hurt: 1.9343 vs 1.9048 (+0.030 bpb over best). The increased attention scaling combined with reduced softcap moved away from the optimum. Both modifications are reverted for subsequent runs.

---

### Run 12: sota_stack_1774299828 (2026-03-23, ~23:04)

**Purpose:** First SOTA-stack attempt -- introduce SmearGate and BigramHash. Double iterations to 1000. Wallclock raised to 1800s.

**Configuration:** 10 layers, SmearGate enabled, bigram_vocab_size=4096, bigram_dim=128, muon_weight_decay=0.04, iterations=1000, max_wallclock=1800s, ortho_init=True. 5 train shards (up from 1).

**Results:**
- All 1000 steps completed in 962s (962ms/step, 16,488 tok/s).
- Pre-quant val_bpb: 1.6881 (val_loss: 2.8502).
- Post-quant val_bpb: 1.6893 (val_loss: 2.8523). Quantisation degradation: +0.0012 bpb.
- Artifact size: 16,708,061 bytes. Payload ratio: 3.92x.
- Params: 24,730,705 (590K more than base due to bigram hash table + smeargate params).
- Total elapsed: 2156.1s.

**Key observations:** Enormous leap: **1.6893 vs 1.9048 bpb** (-0.22 bpb). SmearGate (gated skip connections) and BigramHash (learned bigram embedding lookup) are extremely effective. The bigram table adds a statistical language model component that captures local token co-occurrence patterns the transformer alone struggles with at this scale. Ortho initialisation and weight decay (0.04) also contribute to training stability.

---

### Run 13: sota_stack_1774313082 (2026-03-24, ~02:45)

**Purpose:** Re-run of the SOTA stack, possibly with code changes to the serialisation or SWA pipeline. Same config as Run 12.

**Configuration:** Same as Run 12. 20 train shards (up from 5).

**Results:**
- Log truncated at step 700/1000 (train_loss: 3.2120, train_time: 663s).
- experiments.json records elapsed: 681.9s, exit_code: 1.
- No error message captured in log.

**Key observations:** The process was killed or crashed at step 700. Given exit_code 1 (not -9), this is likely a code error rather than OOM. The train_time of 663s was well within the 1800s wallclock. Possibly a bug in new SWA or serialisation code that was being tested. The increased shard count (20 vs 5) was carried forward.

---

### Run 14: sota_stack_1774314889 (2026-03-24, ~03:14)

**Purpose:** Scale to 3000 iterations, 3600s wallclock. Add SWA (start_frac=0.4, every 50), mixed quantisation (6-bit MLP + 6-bit attn), eval_stride=64. Bigram expanded to 10240 vocab.

**Configuration:** iterations=3000, bigram_vocab_size=10240, swa_enabled=1, swa_start_frac=0.4, swa_every=50, quant_bits_mlp=6, quant_bits_attn=6, eval_stride=64, max_wallclock=3600s. 25,517,137 params.

**Results:**
- Log truncated at step 2400/3000 (train_loss: 2.6007, train_time: 2396s).
- experiments.json records elapsed: 2465.0s, exit_code: 1.
- No error message captured.

**Key observations:** Got 80% through training before failure. The train_loss of 2.6 at step 2400 is promising. The crash likely occurred during the SWA or serialisation phase after step 2400 -- the new SWA/quantisation code had bugs. This is the first run with 6-bit mixed quantisation and SWA, which are novel code paths.

---

### Run 15: sota_stack_1774380145 (2026-03-24, ~21:22)

**Purpose:** Retry of Run 14 after code fixes. Same configuration.

**Configuration:** Identical to Run 14.

**Results:**
- Log truncated at step 2100/3000 (train_loss: 2.6277, train_time: 2285s).
- experiments.json records elapsed: 2359.3s, exit_code: 1.
- No error message captured.

**Key observations:** Failed even earlier than Run 14 (step 2100 vs 2400). The bug in the SWA/serialisation pipeline was not fully fixed. The train loss trajectory (2.6 at step 2100) is consistent with Run 14, confirming the model architecture is sound. The infrastructure around it needs debugging.

---

### Run 16: sota_full_3k (2026-03-24, ~23:15)

**Purpose:** Third attempt at the 3000-iteration SOTA run. Same configuration as Runs 14/15.

**Configuration:** Identical to Runs 14/15. 20 train shards.

**Results:**
- Reached step 2000/3000 (train_loss: 2.4319, train_time: 2223s).
- **Crashed with explicit error:**
  ```
  RuntimeError: Item size 2 for PEP 3118 buffer format string B does not match the dtype B item size 1.
  ```
  at `flat_np = {k: np.array(v, copy=True) for k, v in tree_flatten(model.state)}` (line 1289).
- The error occurs during serialisation when converting MLX bfloat16 tensors to numpy arrays.

**Key observations:** The root cause is identified: MLX bfloat16 tensors (2 bytes/item) cannot be directly converted to numpy via the PEP 3118 buffer protocol because numpy interprets the format string 'B' as uint8 (1 byte/item). The fix requires explicit dtype casting (e.g., `.astype(mx.float32)`) before numpy conversion. This bug affected Runs 14, 15, and 16. The train_loss of 2.43 at step 2000 shows the model was training well.

---

### Run 17: sota_full_3k_v2 (2026-03-25, ~08:34)

**Purpose:** Fixed the serialisation dtype bug. Added EMA (decay 0.997). Full 3000-iteration run with all SOTA features.

**Configuration:** Same base as Run 16 plus ema_enabled=1, ema_decay=0.997. Serialisation bug fixed. SWA start at step 2100, collecting every 50 steps.

**Results:**
- All 3000 steps completed in 3241s (1080ms/step, 16,748 tok/s).
- SWA applied: averaged 18 checkpoints (steps 2100-2950, every 50).
- Pre-quant val_bpb: 1.4922 (val_loss: 2.5195).
- Post-quant val_bpb: **1.4739** (val_loss: 2.4886). Quantisation **improved** bpb by -0.018!
- Artifact size: 13,669,403 bytes. Payload ratio: 3.92x.
- Mixed quantisation: 6-bit MLP, 6-bit attention.
- Final eval: sliding window with stride 64.
- Eval time: 21,194s (~5.9 hours).
- Total elapsed: ~24,435s.

**Key observations:** Major breakthrough: **1.4739 bpb**, down from 1.6893 (-0.22 bpb). The combination of EMA smoothing, SWA averaging (18 checkpoints), and the sliding-window eval with stride 64 all contributed. Remarkably, post-quant bpb was *better* than pre-quant (1.4739 vs 1.4922), suggesting the SWA-averaged weights regularise well under quantisation. This is the first run where quantisation helped rather than hurt.

---

### Run 18: beast_0point9 (2026-03-25, ~20:48)

**Purpose:** Push further -- 11 layers, add TrigramHash, aggressive 5-bit MLP quantisation, 7200s wallclock.

**Configuration:** num_layers=11, trigram_vocab_size=4096, trigram_dim=64, quant_bits_mlp=5, quant_bits_attn=6, max_wallclock=7200s, swa_enabled, ema_enabled. 28,173,402 params.

**Results:**
- Hit wallclock cap at step 1850/3000 (train_time: 7201s, 3892ms/step, 3450 tok/s).
- SWA applied: averaged 26 checkpoints (from step 550 onwards).
- EMA applied.
- Pre-quant val_bpb: 1.5487 (val_loss: 2.6149).
- Post-quant val_bpb: **1.6215** (val_loss: 2.7379). Quantisation degradation: **+0.073 bpb**.
- Artifact size: 8,064,297 bytes.
- Mixed quantisation: 5-bit MLP, 6-bit attention.
- Eval time: 28,790s (~8 hours).

**Key observations:** The 5-bit MLP quantisation was too aggressive -- it caused +0.073 bpb degradation (vs -0.018 for 6-bit in Run 17). Despite the pre-quant bpb of 1.5487 being reasonable, the post-quant 1.6215 is worse than Run 17's 1.4739. The 11th layer + trigram table added ~2.7M params but the step time exploded from ~1080ms to ~3892ms. The trigram computation appears to create a severe bottleneck. Only 1850/3000 steps fit in 7200s. Key lesson: **5-bit MLP quantisation destroys too much information; stick with 6-bit**.

---

### Run 19: ngram_beast (2026-03-25, ~23:33)

**Purpose:** Retry the 11-layer config with EMA enabled, likely with code adjustments from beast_0point9 learnings.

**Configuration:** 11 layers, smeargate, bigram 10240x128, no trigram (dropped from beast), EMA 0.997, wallclock 14400s. Same as beast but without trigram and with longer wallclock.

**Results:**
- Log truncated at step 200/3000 (train_loss: 3.6696, train_time: 196s).
- No final results, no error captured.

**Key observations:** Killed very early. Likely user-terminated to try a different approach after seeing early training dynamics. The step times (~980ms) are much better than beast's 3892ms, confirming the trigram removal fixed the throughput issue.

---

### Run 20: ultimate (2026-03-26, ~00:00)

**Purpose:** Full 3000-iteration run with 11 layers, EMA, and test-time training (TTT) during eval.

**Configuration:** 11 layers, smeargate, bigram 10240x128, EMA 0.997, wallclock 14400s, iterations 3000. TTT eval mode: lr=1.0, epochs=3, SGD with momentum. Mixed quant 6/6.

**Results:**
- All 3000 steps completed in 3098s (1033ms/step, 14,803 tok/s).
- EMA applied. SWA skipped (0 checkpoints collected -- SWA was not properly enabled).
- Pre-quant val_bpb: 1.6238 (val_loss: 2.7417).
- Artifact size: 11,248,555 bytes.
- TTT eval started but **stalled at document 1/50000** (running_bpb: 1.6541 at doc 1).
- No final post-quant bpb produced.

**Key observations:** Two problems. First, SWA collected 0 checkpoints despite config intending it -- the SWA start fraction or trigger logic was misconfigured. Without SWA, the pre-quant bpb (1.6238) is worse than Run 17's SWA-averaged 1.4922. Second, the TTT (test-time training) eval mode was prohibitively slow -- adapting model weights per-document via SGD for 50,000 documents would take days. The run effectively produced no usable final metric.

---

### Run 21: ngram_only (2026-03-26, ~01:50)

**Purpose:** Test n-gram cache during eval (order 7) instead of TTT. Full 3000 iterations, 11 layers.

**Configuration:** 11 layers, smeargate, bigram 10240x128, mixed quant 6/6, eval_stride=64, ngram_cache enabled (order 7). No EMA, no SWA (collected 0 ckpts).

**Results:**
- All 3000 steps completed in 3093s (1031ms/step, 16,010 tok/s).
- SWA skipped (0 checkpoints).
- Pre-quant val_bpb: 1.6209 (val_loss: 2.7367).
- Post-quant val_bpb: **1.6895** (val_loss: 2.8526). Quantisation degradation: +0.069 bpb.
- N-gram cache hit rate: 38.0%.
- Artifact size: 15,617,065 bytes.
- Eval time: 24,902s (~6.9 hours).

**Key observations:** The n-gram cache at order 7 achieved a 38% hit rate but the final bpb of 1.6895 is disappointing -- worse than Run 17's 1.4739. The n-gram cache helps but cannot compensate for the lack of SWA (which gave Run 17 its edge). Without SWA, the 11-layer model's pre-quant bpb (1.6209) is already much worse than Run 17's SWA-averaged 1.4922. The large artifact (15.6MB) also suggests the weights are less compressible without SWA averaging.

---

### Run 22: clean_fixed (2026-03-26, ~09:30)

**Purpose:** Likely a code-fix attempt to resolve the SWA configuration issue discovered in Runs 20-21.

**Configuration:** 11 layers, smeargate, bigram 10240x128, wallclock 14400s, iterations 3000. Same base as Run 20/21.

**Results:**
- Only 10/3000 steps completed before termination.
- Final logged: step 10, train_loss: 6.4372, train_time: 9826ms.
- No error captured.

**Key observations:** Killed almost immediately -- likely a test run that was manually terminated after confirming the code changes compiled and started training, before committing to a full multi-hour run.

---

### Run 23: smasher (2026-03-26, ~10:25) -- CURRENT BEST

**Purpose:** The definitive run. 5000 iterations (up from 3000), working SWA (start_frac 0.77, every 50 steps), 11 layers, 14400s wallclock.

**Configuration:** 11 layers, smeargate, bigram 10240x128, iterations=5000, max_wallclock=14400s, SWA start at step 3850 (collecting from 77% onwards), mixed quant 6/6, eval_stride=64, no EMA, no trigram. 20 train shards. 28,173,402 params.

**Results:**
- All 5000 steps completed in 6098s (1220ms/step, 12,401 tok/s).
- SWA applied: averaged 23 checkpoints (steps 3850-4950, every 50).
- Pre-quant val_bpb: **1.4351** (val_loss: 2.4231).
- Artifact size: 14,746,571 bytes.
- Final eval: sliding window with stride 64, currently in progress.
- Running eval bpb at position 343k/969k: **~1.4257** (extrapolating from running_bpb values ~1.425).
- Total elapsed: ~6098s train + eval still running.

**Key observations:** New personal best at pre-quant level: **1.4351 bpb** (down from Run 17's 1.4922). The key differences from Run 17: +1 layer (11 vs 10), +2000 iterations (5000 vs 3000), and SWA delayed to the final 23% of training (collecting only the most converged checkpoints). The running eval bpb of ~1.4257 suggests the post-quant score will likely be around **1.42-1.43 bpb**, which would beat Run 17's 1.4739 by ~0.05 bpb. The eval is approximately 35% complete as of this writing.

---

### Run 24: definitive_20k (2026-03-28) -- NEW ALL-TIME RECORD

**Purpose:** Definitive run combining every proven technique plus new additions: QuadgramHash, Late QAT, GPTQ-lite clip search, 80 training shards, and 20,000 iterations.

**Configuration:** 11 layers, 512d, 8H/4KV, 3x MLP, LeakyReLU(0.9) squared activation, SmearGate, BigramHash 10240, TrigramHash 4096, QuadgramHash 2048, VRL (Variable-Rate Layers), Gated Attention, Partial RoPE 16/64, LN Scale, Cautious Muon optimiser, Late QAT (quantisation-aware training), GPTQ-lite clip search. 80 train shards (8B tokens), 20,000 iterations, wallclock 43,200s. Mixed quant 6/6. SWA starting at step 18850.

**Results:**
- All 20,000 steps completed in 21,172s (5.88 hours, 1059ms/step avg).
- SWA applied: averaged 23 checkpoints (started step 18850).
- Pre-quant val_bpb: **1.4109** (val_loss: 2.3822).
- Post-quant val_bpb: **1.3471** (val_loss: 2.2745). Quantisation *improved* score by -0.0638 bpb.
- Artifact size: 11,575,498 bytes (payload: 28,621,356, ratio: 3.91x).
- Eval time: 24,883s (6.91 hours), sliding window stride 64.
- Total wall time: 12.79 hours.
- Model params: 28,255,422.

**Key observations:** First run with QuadgramHash, Late QAT, and GPTQ clip search. The combination of Late QAT and GPTQ clip search produced a remarkable result: post-quant bpb (1.3471) is substantially *better* than pre-quant (1.4109), a -0.0638 bpb improvement from quantisation. This is the largest quantisation-helps-score effect observed across all runs — Run 17 showed -0.018, this run shows -0.064. The 20,000 iterations (4x Run 23's 5,000) and 80 shards (4x Run 23's 20) confirm that more data and more training continue to pay dividends. The pre-quant bpb of 1.4109 is already better than Run 23's 1.4351, and the post-quant 1.3471 crushes the previous best (Run 17's 1.4739) by 0.127 bpb. New all-time record.

---

## Evolution Summary

```
2.41 bpb  Baseline (9L, 200 iters, no features)
  |
  | +1 layer, 3x MLP, 2x seq_len, tune LR/momentum
  v
1.94 bpb  Scaled baseline (10L, 500 iters)
  |
  | +grad clipping, warmdown
  v
1.90 bpb  With grad clip
  |
  | +SmearGate, BigramHash, weight decay, ortho init
  v
1.69 bpb  SOTA stack (1000 iters)
  |
  | +SWA, +EMA, 3000 iters, sliding-window eval
  v
1.47 bpb  sota_full_3k_v2 (10L, 3000 iters, SWA+EMA)
  |
  | +1 layer, +2000 iters, SWA tuning
  v
~1.43 bpb  smasher (11L, 5000 iters, SWA)
  |
  | +QuadgramHash, Late QAT, GPTQ clip, 80 shards, 20k iters
  v
1.35 bpb  definitive_20k (11L, 20000 iters, SWA, Late QAT+GPTQ) ← NEW RECORD
```

## Dead Ends and Failed Hypotheses

1. **10x batch size** (Run 8): Catastrophic. Reduced steps from 500 to 99, wasted the entire wallclock budget. Lesson: match batch size to wallclock constraints.
2. **logit_softcap=29** (Run 10): Marginal regression. Reverted.
3. **qk_gain_init=1.9** (Run 11): Clear regression (+0.030 bpb). Attention scaling is sensitive; default 1.5 is better.
4. **5-bit MLP quantisation** (Run 18): Too aggressive. +0.073 bpb degradation vs +0.018 improvement at 6-bit. Stick with 6-bit.
5. **TrigramHash** (Run 18): Added ~2.7M params but created a 3.6x throughput penalty (3892ms/step vs 1080ms/step). The trigram computation is not optimised for MLX.
6. **Test-Time Training (TTT)** (Run 20): Prohibitively slow -- adapting per-document for 50,000 docs would take days. Not viable within competition constraints.
7. **N-gram cache** (Run 21): 38% hit rate but bpb improvement minimal without SWA. The overhead of the cache is not justified.
8. **SWA misconfiguration** (Runs 20-21): Collected 0 checkpoints due to missing or misconfigured start parameters, negating the primary advantage discovered in Run 17.

## Key Discoveries

1. **SWA is the single most impactful technique**: Run 17 with SWA achieved 1.4739 bpb; the same model without SWA (Runs 20-21) scored ~1.62 bpb. SWA accounts for ~0.15 bpb of improvement.
2. **Late QAT + GPTQ clip search is transformative**: Run 24 showed post-quant bpb *0.064 below* pre-quant (1.3471 vs 1.4109). This is the largest quantisation-helps effect observed, far exceeding Run 17's -0.018. Late QAT trains the model to be quantisation-friendly; GPTQ clip search finds optimal clipping thresholds.
3. **More iterations always help**: 200 -> 500 -> 1000 -> 3000 -> 5000 -> 20000 iterations, each step improved results. Training is not saturating even at 20k steps.
4. **More data always helps**: 80 shards (8B tokens) at 20k iterations produced measurably better results than 20 shards at 5k iterations.
5. **SmearGate + BigramHash**: Together worth ~0.22 bpb improvement (1.90 -> 1.69). These are the highest-value architectural additions.
6. **Mixed 6-bit quantisation**: The sweet spot. 5-bit destroys too much information; 6-bit loses almost nothing.
7. **Sliding-window eval with stride 64**: Essential for proper validation on long sequences. Full-stride eval would be impossibly slow.
