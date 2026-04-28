#!/bin/bash
# SLURM submit script to compute u_pred/u_tgt mean+std on val/test for an LMM checkpoint.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/slurm_logs}"
PARTITION="${PARTITION:-main}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3}"
MEM="${MEM:-64G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"

# By default, inspect the same run you were continuing from.
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/logs/train/runs/2026-04-24_18-27-35/checkpoints/last.ckpt}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-8}"

if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Checkpoint file not found: $CKPT_PATH" >&2
    exit 1
fi

if [[ "$DATA_NUM_WORKERS" -ge "$CPUS_PER_TASK" ]]; then
    DATA_NUM_WORKERS=$((CPUS_PER_TASK - 1))
fi
if [[ "$DATA_NUM_WORKERS" -lt 1 ]]; then
    DATA_NUM_WORKERS=1
fi

mkdir -p "$LOG_DIR"

sbatch \
    --job-name="LMM_u_stats" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --partition="$PARTITION" \
    --gres=gpu:${GPU_TYPE}:${NUM_GPUS} \
    --time=0 \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    --wrap="cd $REPO_ROOT && \
            export PROJECT_ROOT=$REPO_ROOT && \
            export PYTHONPATH=$REPO_ROOT:\$PYTHONPATH && \
            export OMP_NUM_THREADS=1 && \
            $REPO_ROOT/.venv/bin/python scripts/lmm_u_stats.py \
                    experiment=downscaling_LMM_res_2mT.yaml \
                    ckpt_path=$CKPT_PATH \
                    data.num_workers=$DATA_NUM_WORKERS \
                    paths.data_dir=$REPO_ROOT/LDM-downscaling/full_Dataset/"
