#!/bin/bash
# SLURM submit script for VAE training on H100 MIG (3g.40gb).
# Use 1 MIG device per job as recommended by the cluster guidance.

# --- Customize these for your environment ---
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/slurm_logs}"
PARTITION="${PARTITION:-mig}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3_3g.40gb}"
MEM="${MEM:-64G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
TIME="${TIME:-12:00:00}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-false}"  # true => continue, false => train from scratch
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/logs/train/runs/2026-04-22_17-55-14/checkpoints/last.ckpt}"
LOAD_OPTIMIZER_STATE="${LOAD_OPTIMIZER_STATE:-true}"  # true => resume epoch/optimizer/scheduler state
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-8}"

# If resuming from a checkpoint, keep writing logs into the same Hydra run directory.
# If training from scratch, force ckpt_path=null and use a new timestamped Hydra run dir.
if [[ "$RESUME_FROM_CHECKPOINT" != "true" ]]; then
    CKPT_PATH="null"
    LOAD_OPTIMIZER_STATE="false"
fi

RESUME_RUN_DIR=""
if [[ "$CKPT_PATH" != "null" && -f "$CKPT_PATH" ]]; then
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
            export OMP_NUM_THREADS=1 && \
            $REPO_ROOT/.venv/bin/python src/train.py \
                    experiment=downscaling_LMM_res_2mT_MIG.yaml \
                    ckpt_path=$CKPT_PATH \
                    load_optimizer_state=$LOAD_OPTIMIZER_STATE \
                    hydra.run.dir=$HYDRA_RUN_DIR \
                    trainer.max_epochs=100 \
                    data.num_workers=$DATA_NUM_WORKERS \
                    paths.data_dir=$REPO_ROOT/LDM-downscaling/full_Dataset/ \
                    callbacks.rich_progress_bar=null"
