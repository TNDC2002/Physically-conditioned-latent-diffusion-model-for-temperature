#!/bin/bash
# SLURM wrapper for experiment=downscaling_LMM_res_2mT.
# Training defaults live in configs/experiment/downscaling_LMM_res_2mT.yaml — this file only sets
# cluster resources, PROJECT_ROOT, resume path, and a few overrides that depend on the allocation.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/slurm_logs}"
PARTITION="${PARTITION:-main}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3}"
MEM="${MEM:-64G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"

LOAD_FROM_CHECKPOINT="${LOAD_FROM_CHECKPOINT:-true}"
REUSE_RUN_DIR="${REUSE_RUN_DIR:-true}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/logs/train/runs/2026-04-24_18-27-35/checkpoints/last-v1.ckpt}"
LOAD_OPTIMIZER_STATE="${LOAD_OPTIMIZER_STATE:-true}"
RESET_SCHEDULER_AND_LR="${RESET_SCHEDULER_AND_LR:-false}"
RESET_LR_VALUE="${RESET_LR_VALUE:-1e-4}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-8}"

to_bool() {
    local v
    v="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    case "$v" in
        true|1|yes|y) echo "true" ;;
        false|0|no|n) echo "false" ;;
        *)
            echo "Invalid boolean value: '$1' (expected true/false, 1/0, yes/no)" >&2
            exit 2
            ;;
    esac
}

LOAD_FROM_CHECKPOINT="$(to_bool "$LOAD_FROM_CHECKPOINT")"
REUSE_RUN_DIR="$(to_bool "$REUSE_RUN_DIR")"
LOAD_OPTIMIZER_STATE="$(to_bool "$LOAD_OPTIMIZER_STATE")"
RESET_SCHEDULER_AND_LR="$(to_bool "$RESET_SCHEDULER_AND_LR")"

CKPT_ARG="null"
if [[ "$LOAD_FROM_CHECKPOINT" == "true" ]]; then
    if [[ ! -f "$CKPT_PATH" ]]; then
        echo "Checkpoint file not found: $CKPT_PATH" >&2
        exit 1
    fi
    CKPT_ARG="$CKPT_PATH"
elif [[ "$LOAD_OPTIMIZER_STATE" == "true" ]]; then
    echo "LOAD_OPTIMIZER_STATE=true requires LOAD_FROM_CHECKPOINT=true. Forcing false." >&2
    LOAD_OPTIMIZER_STATE="false"
fi

if [[ "$RESET_SCHEDULER_AND_LR" == "true" && "$LOAD_OPTIMIZER_STATE" != "true" ]]; then
    echo "RESET_SCHEDULER_AND_LR=true requires LOAD_OPTIMIZER_STATE=true. Forcing RESET_SCHEDULER_AND_LR=false." >&2
    RESET_SCHEDULER_AND_LR="false"
fi

RESUME_RUN_DIR=""
if [[ "$REUSE_RUN_DIR" == "true" ]]; then
    if [[ "$LOAD_FROM_CHECKPOINT" != "true" ]]; then
        echo "REUSE_RUN_DIR=true requires LOAD_FROM_CHECKPOINT=true (to infer run dir from CKPT_PATH)." >&2
        exit 1
    fi
    RESUME_RUN_DIR="$(dirname "$(dirname "$CKPT_PATH")")"
fi

DEFAULT_HYDRA_RUN_DIR='\${paths.log_dir}/\${task_name}/runs/\${now:%Y-%m-%d}_\${now:%H-%M-%S}'
HYDRA_RUN_DIR="${RESUME_RUN_DIR:-$DEFAULT_HYDRA_RUN_DIR}"

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
                    experiment=downscaling_LMM_res_2mT \
                    ckpt_path=$CKPT_ARG \
                    load_optimizer_state=$LOAD_OPTIMIZER_STATE \
                    reset_scheduler_on_resume=$RESET_SCHEDULER_AND_LR \
                    reset_lr_to_default_on_resume=$RESET_SCHEDULER_AND_LR \
                    reset_lr_default_value=${RESET_LR_VALUE:-null} \
                    hydra.run.dir=$HYDRA_RUN_DIR \
                    data.num_workers=$DATA_NUM_WORKERS"
