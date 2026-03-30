#!/bin/bash
# =============================================================================
# PARAMETER GOLF — GCP A100 RUNBOOK
# =============================================================================
# Copy-paste each section into your terminal.
# Every command is on a single line — no backslashes.
#
# Cost: ~$1.10/hr spot for 1x A100
# Expected run time: ~2 hours for full 20k iteration run
# =============================================================================


# =============================================================================
# PHASE 1: CREATE THE VM (run on your Mac)
# =============================================================================

# Create 1x A100 spot instance
gcloud compute instances create param-golf-a100 --zone=us-central1-a --machine-type=a2-highgpu-1g --accelerator="type=nvidia-tesla-a100,count=1" --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 --image-project=deeplearning-platform-release --boot-disk-size=200GB --maintenance-policy=TERMINATE --provisioning-model=SPOT


# =============================================================================
# PHASE 2: SSH INTO THE VM (run on your Mac)
# =============================================================================

gcloud compute ssh param-golf-a100 --zone=us-central1-a


# =============================================================================
# PHASE 3: SETUP ON THE VM (run inside the VM after SSH)
# =============================================================================

# Clone repo, install deps, download data
cd ~ && git clone -b lappie-submission https://github.com/lappiecto/parameter-golf.git && cd parameter-golf && pip install sentencepiece zstandard huggingface-hub && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10 && cp records/track_10min_16mb/lappie-submission/train_gpt.py .


# =============================================================================
# PHASE 4a: QUICK SMOKE TEST — 10 iterations (~30 seconds)
# =============================================================================

RUN_ID=a100_smoke ITERATIONS=10 TRAIN_BATCH_TOKENS=16384 VAL_BATCH_SIZE=16384 TRAIN_SEQ_LEN=1024 TRAIN_LOG_EVERY=1 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=120 SWA_ENABLED=0 QAT_ENABLED=0 WARMDOWN_ITERS=10 torchrun --standalone --nproc_per_node=1 train_gpt.py


# Check it worked:
grep "final_int8" logs/a100_smoke.txt


# =============================================================================
# PHASE 4b: FULL RUN — 20k iterations, 2hr wallclock (~$2.20)
# =============================================================================

# Download more data for the full run (10 shards already downloaded, get all 80)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# Full training run — matches our best Mac config
RUN_ID=a100_full ITERATIONS=20000 TRAIN_BATCH_TOKENS=262144 VAL_BATCH_SIZE=262144 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=200 MAX_WALLCLOCK_SECONDS=7200 SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 WARMDOWN_ITERS=3000 QAT_ENABLED=1 QAT_LR_THRESHOLD=0.15 QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6 EVAL_STRIDE=64 MUON_WEIGHT_DECAY=0.04 GRAD_CLIP_NORM=0.3 torchrun --standalone --nproc_per_node=1 train_gpt.py


# =============================================================================
# PHASE 4c: COMPETITION SIMULATION — 10 min wallclock, big batches
# =============================================================================

# This simulates what happens on 8xH100 in the competition (but on 1 GPU)
RUN_ID=a100_comp ITERATIONS=20000 TRAIN_BATCH_TOKENS=262144 VAL_BATCH_SIZE=262144 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100 MAX_WALLCLOCK_SECONDS=600 SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 WARMDOWN_ITERS=500 QAT_ENABLED=1 QAT_LR_THRESHOLD=0.15 QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6 EVAL_STRIDE=64 MUON_WEIGHT_DECAY=0.04 GRAD_CLIP_NORM=0.3 torchrun --standalone --nproc_per_node=1 train_gpt.py


# =============================================================================
# PHASE 4d: 3-SEED SUBMISSION RUNS (for statistical significance)
# =============================================================================

RUN_ID=submission_seed42 SEED=42 ITERATIONS=20000 TRAIN_BATCH_TOKENS=262144 VAL_BATCH_SIZE=262144 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=200 MAX_WALLCLOCK_SECONDS=7200 SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 WARMDOWN_ITERS=3000 QAT_ENABLED=1 QAT_LR_THRESHOLD=0.15 QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6 EVAL_STRIDE=64 MUON_WEIGHT_DECAY=0.04 GRAD_CLIP_NORM=0.3 torchrun --standalone --nproc_per_node=1 train_gpt.py

RUN_ID=submission_seed1337 SEED=1337 ITERATIONS=20000 TRAIN_BATCH_TOKENS=262144 VAL_BATCH_SIZE=262144 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=200 MAX_WALLCLOCK_SECONDS=7200 SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 WARMDOWN_ITERS=3000 QAT_ENABLED=1 QAT_LR_THRESHOLD=0.15 QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6 EVAL_STRIDE=64 MUON_WEIGHT_DECAY=0.04 GRAD_CLIP_NORM=0.3 torchrun --standalone --nproc_per_node=1 train_gpt.py

RUN_ID=submission_seed2024 SEED=2024 ITERATIONS=20000 TRAIN_BATCH_TOKENS=262144 VAL_BATCH_SIZE=262144 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=200 MAX_WALLCLOCK_SECONDS=7200 SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 WARMDOWN_ITERS=3000 QAT_ENABLED=1 QAT_LR_THRESHOLD=0.15 QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6 EVAL_STRIDE=64 MUON_WEIGHT_DECAY=0.04 GRAD_CLIP_NORM=0.3 torchrun --standalone --nproc_per_node=1 train_gpt.py

# Check all scores:
grep "final_int8.*val_bpb" logs/submission_seed*.txt


# =============================================================================
# PHASE 5: STREAM LOGS TO MAC (run on your Mac in a separate terminal)
# =============================================================================

# This streams the remote training log to your local machine
# The dashboard at localhost:8888 will pick it up automatically
gcloud compute ssh param-golf-a100 --zone=us-central1-a -- "tail -f ~/parameter-golf/logs/*.txt" > /tmp/cloud_a100.log 2>/dev/null &


# =============================================================================
# PHASE 6: DOWNLOAD RESULTS (run on your Mac)
# =============================================================================

# Download training logs for submission
gcloud compute scp param-golf-a100:~/parameter-golf/logs/submission_seed*.txt ~/codex-root/parameter-golf/records/track_10min_16mb/lappie-submission/ --zone=us-central1-a

# Download the quantised model artifact
gcloud compute scp param-golf-a100:~/parameter-golf/final_model.int8.ptz ~/codex-root/parameter-golf/records/track_10min_16mb/lappie-submission/ --zone=us-central1-a


# =============================================================================
# PHASE 7: DELETE THE VM — STOP CHARGES (run on your Mac)
# =============================================================================

gcloud compute instances delete param-golf-a100 --zone=us-central1-a --quiet


# =============================================================================
# NOTES
# =============================================================================
#
# Costs:
#   1x A100 spot: ~$1.10/hr
#   Smoke test: ~$0.05 (3 minutes)
#   Full run: ~$2.20 (2 hours)
#   3-seed submission: ~$6.60 (6 hours total)
#   Total: ~$9 for complete validation + submission
#
# If spot VM gets preempted:
#   Just recreate with the same Phase 1 command and re-run
#   Training checkpoints are NOT saved — you restart from scratch
#
# To check GPU usage while running:
#   SSH in from another terminal and run: nvidia-smi
#
# To check training progress:
#   SSH in and run: tail -5 ~/parameter-golf/logs/*.txt
#
