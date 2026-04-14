#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/submit_post_train_pipeline.sh <train_job_id>
#
# Example:
#   bash scripts/submit_post_train_pipeline.sh 6834

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <train_job_id>"
  exit 1
fi

TRAIN_JOB_ID="$1"
REPO_DIR="/home/chuongtnd/git-repo/Physically-conditioned-latent-diffusion-model-for-temperature"
OLD_CKPT="$REPO_DIR/pretrained_models/LDM_residual_2mT.ckpt"
NEW_CKPT="$REPO_DIR/logs/train/runs/2026-03-23_15-19-17/checkpoints/last.ckpt"

POST_JOB_ID="$(
  sbatch --parsable \
    --job-name=post_train_replace_and_submit_infer \
    --dependency=afterok:${TRAIN_JOB_ID} \
    --partition=main \
    --mem=32G \
    --time=00:15:00 \
    --output="$REPO_DIR/slurm_logs/post_train-%j.out" \
    --error="$REPO_DIR/slurm_logs/post_train-%j.err" \
    --wrap="set -euo pipefail; \
      rm -f \"$OLD_CKPT\"; \
      cp \"$NEW_CKPT\" \"$OLD_CKPT\"; \
      sbatch --job-name=run_inference_nb --partition=main --gres=gpu:nvidia_h100_80gb_hbm3:1 --mem=64G --time=1:00:00 --output=$REPO_DIR/slurm_logs/inference_nb-%j.out --error=$REPO_DIR/slurm_logs/inference_nb-%j.err --wrap=\"cd $REPO_DIR && export PYTHONPATH=$REPO_DIR:\\\$PYTHONPATH && .venv/bin/jupyter nbconvert --to notebook --execute notebooks/models_inference.ipynb --output=models_inference_executed.ipynb\""
)"

echo "Submitted post-train job: ${POST_JOB_ID}"
echo "It will run after training job ${TRAIN_JOB_ID} completes successfully."
