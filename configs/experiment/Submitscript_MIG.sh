#!/bin/bash
# SLURM submit script for VAE training on H100 MIG (3g.40gb).
# Use 1 MIG device per job as recommended by the cluster guidance.

# --- Customize these for your environment ---
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/slurm_logs}"
PARTITION="${PARTITION:-mig}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3_3g.40gb}"
MEM="${MEM:-32G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
TIME="${TIME:-12:00:00}"
# CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/logs/train/runs/2026-04-20_13-30-10/checkpoints/last.ckpt}"
CKPT_PATH=null

mkdir -p "$LOG_DIR"

# Submit the job to MIG partition with exactly 1 MIG device.
sbatch \
    --job-name="LMM_res_2mT_mig" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --partition="$PARTITION" \
    --gres=gpu:${GPU_TYPE}:1 \
    --time=0 \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    --wrap="cd $REPO_ROOT && \
            export PROJECT_ROOT=$REPO_ROOT && \
            export PYTHONPATH=$REPO_ROOT:\$PYTHONPATH && \
            $REPO_ROOT/.venv/bin/python src/train.py \
                    experiment=downscaling_LMM_res_2mT_MIG \
                    ckpt_path=$CKPT_PATH \
                    trainer.max_epochs=100 \
                    paths.data_dir=$REPO_ROOT/LDM-downscaling/full_Dataset/ \
                    callbacks.rich_progress_bar=null"
