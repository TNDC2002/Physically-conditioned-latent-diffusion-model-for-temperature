#!/bin/bash
# SLURM: full LMM smoke (parity + metrics + micro-train) on **one MIG GPU**.
# Do not run the smoke suite on CPU for parity/metrics/train — it is intentionally skipped
# without CUDA (see scripts/lmm_smoke_suite.py).

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/slurm_logs}"
PARTITION="${PARTITION:-mig}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3_3g.40gb}"
MEM="${MEM:-32G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
TIME="${TIME:-02:00:00}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/LDM-downscaling/full_Dataset/}"

mkdir -p "$LOG_DIR"

sbatch \
    --job-name="LMM_smoke_2mT_mig" \
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
            export LMM_SMOKE_DATA_DIR=$DATA_DIR && \
            export LMM_SMOKE_MAX_STEPS=10 && \
            export LMM_SMOKE_QUIET=1 && \
            unset LMM_SMOKE_SKIP_TRAIN && \
            $REPO_ROOT/.venv/bin/python $REPO_ROOT/scripts/lmm_smoke_suite.py \
                experiment=downscaling_LMM_res_2mT_smoke \
                model=lmm \
                paths.data_dir=$DATA_DIR \
                paths.pretrained_models_dir=$REPO_ROOT/pretrained_models/ \
                callbacks.rich_progress_bar=null"
