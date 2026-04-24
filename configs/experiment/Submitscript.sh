#!/bin/bash
# SLURM submit script for LDM_res training on H100 GPU(s).
# Edit the variables below to match your cluster and run.

# --- Customize these for your environment ---
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/slurm_logs}"
PARTITION="${PARTITION:-main}"   # default partition on this cluster
NUM_GPUS="${NUM_GPUS:-1}"               # 1 H100 (80GB) for compute-limited; use more if allowed
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3}"            # GRES name on this cluster
MEM="${MEM:-64G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
TIME="${TIME:-12:00:00}"
# Independent toggles:
# - LOAD_FROM_CHECKPOINT: load model state from CKPT_PATH (ckpt_path=<path>).
# - REUSE_RUN_DIR: reuse CKPT_PATH's run folder; otherwise create a new timestamped run folder.
# These toggles are intentionally independent for clarity.
LOAD_FROM_CHECKPOINT="${LOAD_FROM_CHECKPOINT:-false}"
REUSE_RUN_DIR="${REUSE_RUN_DIR:-false}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/logs/train/runs/2026-04-24_08-42-44/checkpoints/last.ckpt}"
LOAD_OPTIMIZER_STATE="${LOAD_OPTIMIZER_STATE:-false}"  # true => restore optimizer/scheduler/epoch from checkpoint
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-8}"

# Normalize boolean-like values to strict true/false.
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

# Keep dataloader workers below requested CPUs to avoid oversubscription.
if [[ "$DATA_NUM_WORKERS" -ge "$CPUS_PER_TASK" ]]; then
    DATA_NUM_WORKERS=$((CPUS_PER_TASK - 1))
fi
if [[ "$DATA_NUM_WORKERS" -lt 1 ]]; then
    DATA_NUM_WORKERS=1
fi

mkdir -p "$LOG_DIR"

# Submit the job (request H100 GPU(s); adjust GPU_TYPE if your cluster uses a different GRES name)
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
                    experiment=downscaling_LMM_res_2mT.yaml \
                    ckpt_path=$CKPT_ARG \
                    load_optimizer_state=$LOAD_OPTIMIZER_STATE \
                    hydra.run.dir=$HYDRA_RUN_DIR \
                    trainer.max_epochs=100 \
                    data.num_workers=$DATA_NUM_WORKERS \
                    paths.data_dir=$REPO_ROOT/LDM-downscaling/full_Dataset/ \
                    callbacks.rich_progress_bar=null"