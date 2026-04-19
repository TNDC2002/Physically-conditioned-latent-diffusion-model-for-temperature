#!/bin/bash
# SLURM submit script for LMM (latent MeanFlow) on H100 MIG (3g.40gb).
# Reuses the same partition / GRES / env pattern as Submitscript_MIG.sh.
#
# Pretrained weights (UNET_2mT.ckpt, VAE_residual_2mT.ckpt) are read from
# ${REPO_ROOT}/pretrained_models/ via configs (paths.pretrained_models_dir).
#
# Usage:
#   bash configs/experiment/Submitscript_MIG_LMM.sh
#
# Parity + metrics + micro-train smoke (separate script): bash configs/experiment/Submitscript_LMM_MIG.sh
#
# Full training (disable one-batch smoke):
#   LMM_FULL_TRAIN=1 bash configs/experiment/Submitscript_MIG_LMM.sh
#
# Optional resume:
#   CKPT_PATH=/path/to/lmm.ckpt bash configs/experiment/Submitscript_MIG_LMM.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/slurm_logs}"
PARTITION="${PARTITION:-mig}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3_3g.40gb}"
MEM="${MEM:-32G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
# Match Submitscript_MIG.sh: 0 often means “no limit” on this cluster; override if needed.
TIME="${TIME:-0}"
CKPT_PATH="${CKPT_PATH:-}"

# Default: Lightning fast_dev_run (1 train batch, 1 val batch) to validate MIG + data + ckpts.
# Set LMM_FULL_TRAIN=1 for normal max_epochs from trainer config.
LMM_FULL_TRAIN="${LMM_FULL_TRAIN:-0}"
SMOKE_ARGS=()
if [ "$LMM_FULL_TRAIN" != "1" ]; then
  SMOKE_ARGS+=(trainer.fast_dev_run=true)
fi

CKPT_ARGS=()
if [ -n "$CKPT_PATH" ]; then
  CKPT_ARGS+=(ckpt_path="$CKPT_PATH")
fi

mkdir -p "$LOG_DIR"

sbatch \
    --job-name="LMM_res_2mT_mig" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --partition="$PARTITION" \
    --gres=gpu:${GPU_TYPE}:1 \
    --time="$TIME" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    --wrap="cd $REPO_ROOT && \
            export PROJECT_ROOT=$REPO_ROOT && \
            export PYTHONPATH=$REPO_ROOT:\$PYTHONPATH && \
            $REPO_ROOT/.venv/bin/python src/train.py \
                    experiment=downscaling_LMM_res_2mT_MIG \
                    paths.data_dir=$REPO_ROOT/LDM-downscaling/full_Dataset/ \
                    callbacks.rich_progress_bar=null \
                    ${SMOKE_ARGS[*]:-} \
                    ${CKPT_ARGS[*]:-}"
