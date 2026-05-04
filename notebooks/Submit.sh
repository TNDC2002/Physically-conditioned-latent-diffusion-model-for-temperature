#!/bin/bash
# Submit models_inference.ipynb (or NOTEBOOK=...) to Slurm with GPU; writes executed notebook under \$REPO_ROOT/outputs/
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

NOTEBOOK="${NOTEBOOK:-$REPO_ROOT/notebooks/metric_computation.ipynb}"
# Slurm stdout/stderr (match configs/experiment/Submitscript.sh → slurm_logs/)
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$REPO_ROOT/slurm_logs}"

BASENAME=$(basename "$NOTEBOOK" .ipynb)

echo "Submitting notebook job:"
echo "  REPO_ROOT:     $REPO_ROOT"
echo "  SLURM_LOG_DIR: $SLURM_LOG_DIR"
echo "  Input:         $NOTEBOOK"
echo "  Output dir:    $REPO_ROOT/outputs/  (timestamped ${BASENAME}_*.ipynb)"

mkdir -p "$SLURM_LOG_DIR" "$REPO_ROOT/outputs"

# Dataset lives under the project mount: LDM-downscaling/full_Dataset/ (see training yaml paths.data_dir).
# Override if your mount is elsewhere: LDM_DATA_ROOT=/path/to/full_Dataset bash notebooks/Submit.sh
LDM_DATA_ROOT_DEFAULT="$REPO_ROOT/LDM-downscaling/full_Dataset"
LDM_DATA_RESOLVED="${LDM_DATA_ROOT:-$LDM_DATA_ROOT_DEFAULT}"
if [[ ! -f "$LDM_DATA_RESOLVED/normalization_data.pkl" ]]; then
    echo "ERROR: Missing normalization_data.pkl under data root:" >&2
    echo "  $LDM_DATA_RESOLVED" >&2
    echo "Set LDM_DATA_ROOT to your full_Dataset directory, or ensure \$REPO_ROOT/LDM-downscaling/full_Dataset exists." >&2
    exit 1
fi

# --- Submit job ---
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=nb_${BASENAME}
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:${GPU_TYPE}:${NUM_GPUS}
#SBATCH --mem=$MEM
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --time=$TIME
#SBATCH --output=$SLURM_LOG_DIR/%x-%j.out
#SBATCH --error=$SLURM_LOG_DIR/%x-%j.err

set -euo pipefail
echo "Running on node: \$(hostname)"

cd "$REPO_ROOT"
mkdir -p "$SLURM_LOG_DIR" "$REPO_ROOT/outputs"

export PROJECT_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
# Notebook reads this in \"Set paths\" (default: under project mount full_Dataset).
export LDM_DATA_ROOT="${LDM_DATA_ROOT:-$REPO_ROOT/LDM-downscaling/full_Dataset}"

source .venv/bin/activate

OUT_DIR="$REPO_ROOT/outputs"
OUT_FILE="${BASENAME}_\$(date +%Y%m%d_%H%M%S).ipynb"
mkdir -p "\$OUT_DIR"

jupyter nbconvert \\
    --to notebook \\
    --execute "$NOTEBOOK" \\
    --output-dir "\$OUT_DIR" \\
    --output "\$OUT_FILE" \\
    --ExecutePreprocessor.timeout=-1 \\
    --ExecutePreprocessor.iopub_timeout=86400 \\
    --ExecutePreprocessor.kernel_name=python3

echo "Done. Output: \$OUT_DIR/\$OUT_FILE"
EOF
