#!/bin/bash
# One-sample LMM schedule debug → slurm_logs/lmm_sched_debug-%j.out
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$REPO_ROOT/slurm_logs}"
PARTITION="${PARTITION:-mig}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3_3g.40gb}"
MEM="${MEM:-32G}"
TIME="${TIME:-0:15:00}"

mkdir -p "$SLURM_LOG_DIR"

LDM_DATA_ROOT="${LDM_DATA_ROOT:-$REPO_ROOT/LDM-downscaling/full_Dataset}"
LMM_CKPT="${LMM_CKPT:-}"

if [[ -z "$LMM_CKPT" ]]; then
  echo "Set LMM_CKPT to your Lightning checkpoint, e.g.:" >&2
  echo "  LMM_CKPT=./logs/train/runs/.../checkpoints/last.ckpt bash $0" >&2
  exit 1
fi

echo "Submitting LMM schedule debug:"
echo "  LMM_CKPT=$LMM_CKPT"
echo "  LDM_DATA_ROOT=$LDM_DATA_ROOT"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lmm_sched_dbg
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:${GPU_TYPE}:1
#SBATCH --mem=$MEM
#SBATCH --cpus-per-task=4
#SBATCH --time=$TIME
#SBATCH --output=$SLURM_LOG_DIR/lmm_sched_debug-%j.out
#SBATCH --error=$SLURM_LOG_DIR/lmm_sched_debug-%j.err

set -euo pipefail
cd "$REPO_ROOT"
export PROJECT_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:\${PYTHONPATH:-}"
export LDM_DATA_ROOT="$LDM_DATA_ROOT"
export LMM_CKPT="$LMM_CKPT"
export PYTHONUNBUFFERED=1

source .venv/bin/activate
python -u scripts/lmm_inference_schedule_debug.py --n-steps 3 --schedule "\${LMM_SCHEDULE:-uniform}" --ckpt "\$LMM_CKPT"
EOF
