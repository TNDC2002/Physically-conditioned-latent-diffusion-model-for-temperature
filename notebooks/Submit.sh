#!/bin/bash
# Submit models_inference.ipynb (or NOTEBOOK=...) to Slurm with GPU; writes executed notebook under ./outputs/
#
# Usage (from repo root; this script calls sbatch):
#   bash notebooks/Submit.sh
# Note: `bash -n notebooks/Submit.sh` only checks syntax — it does NOT submit a job.
# Optional env overrides:
#   NOTEBOOK=... PARTITION=... TIME=8:00:00 MEM=64G ...

# --- Config ---
# Submit.sh lives in <repo>/notebooks/ → repo root is one level up (not ../..).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARTITION="${PARTITION:-main}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3}"
MEM="${MEM:-64G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
# Many clusters reject 0:00:00; set a walltime if your site requires it (e.g. TIME=8:00:00).
TIME="${TIME:-0:00:00}"

NOTEBOOK="${NOTEBOOK:-$REPO_ROOT/notebooks/models_inference.ipynb}"

# Output with timestamp (avoid overwrite); path is relative to REPO_ROOT after job cds
BASENAME=$(basename "$NOTEBOOK" .ipynb)
OUTPUT_NOTEBOOK="outputs/${BASENAME}_$(date +%Y%m%d_%H%M%S).ipynb"

echo "Submitting notebook job:"
echo "  REPO_ROOT: $REPO_ROOT"
echo "  Input:     $NOTEBOOK"
echo "  Output:    $OUTPUT_NOTEBOOK"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/outputs"

# --- Submit job ---
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=nb_${BASENAME}
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:${GPU_TYPE}:${NUM_GPUS}
#SBATCH --mem=$MEM
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --time=$TIME
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
echo "Running on node: \$(hostname)"

cd "$REPO_ROOT"
mkdir -p logs outputs

export PROJECT_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

source .venv/bin/activate

# Execute notebook (paths below are relative to REPO_ROOT)
jupyter nbconvert \\
    --to notebook \\
    --execute "$NOTEBOOK" \\
    --output "$OUTPUT_NOTEBOOK" \\
    --ExecutePreprocessor.timeout=-1 \\
    --ExecutePreprocessor.kernel_name=python3

echo "Done. Output saved to $REPO_ROOT/$OUTPUT_NOTEBOOK"
EOF
