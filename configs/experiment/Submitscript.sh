#!/bin/bash
# SLURM wrapper for experiment=downscaling_LMM_res_2mT.
# Training (``ckpt_path``, ``load_optimizer_state``, scheduler/LR resume flags, optional ``hydra.run.dir``)
# lives in configs/experiment/downscaling_LMM_res_2mT.yaml — this file is only cluster + one data override.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# LOG_DIR="${LOG_DIR:-$REPO_ROOT/slurm_logs}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/LDM-downscaling/slurm_logs}"
PARTITION="${PARTITION:-main}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3}"
MEM="${MEM:-64G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"

# Optional: override dataloader workers for the allocation (everything else from the experiment YAML).
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-8}"

if [[ "$DATA_NUM_WORKERS" -ge "$CPUS_PER_TASK" ]]; then
    DATA_NUM_WORKERS=$((CPUS_PER_TASK - 1))
fi
if [[ "$DATA_NUM_WORKERS" -lt 1 ]]; then
    DATA_NUM_WORKERS=1
fi

mkdir -p "$LOG_DIR"

sbatch \
    --job-name="LMM_res_2mT" \
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
            $REPO_ROOT/.venv/bin/python src/train.py \
                    experiment=downscaling_LMM_res_2mT_pretrain \
                    data.num_workers=$DATA_NUM_WORKERS"
