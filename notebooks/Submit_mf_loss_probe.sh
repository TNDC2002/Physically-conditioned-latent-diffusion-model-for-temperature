#!/bin/bash
# Submit meanflow_loss_deviation_probe.ipynb on 1× H100 MIG (3g.40gb).
# Writes executed notebook (with outputs) under $REPO_ROOT/outputs/ — same pipeline as Submit_metrics.sh.
#
# Usage (from repo root):
#   bash notebooks/Submit_mf_loss_probe.sh
#
# Optional env:
#   MAX_VAL_BATCHES=50              # cap batches (unset = full val split ~1105 batches)
#   PARTITION=mig TIME=8:00:00 MEM=64G
#   PARTITION=main GPU_TYPE=nvidia_h100_80gb_hbm3
#   KEEP_EXEC_OUTPUTS=1             # default: keep plots/tables in final .ipynb
#   KEEP_EXEC_OUTPUTS=0             # clear outputs for smaller notebook file

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARTITION="${PARTITION:-mig}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3_3g.40gb}"
MEM="${MEM:-64G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
TIME="${TIME:-0:00:00}"

NOTEBOOK="${NOTEBOOK:-$REPO_ROOT/notebooks/meanflow_loss_deviation_probe.ipynb}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$REPO_ROOT/slurm_logs}"
BASENAME=$(basename "$NOTEBOOK" .ipynb)

echo "Submitting MeanFlow loss probe:"
echo "  REPO_ROOT:       $REPO_ROOT"
echo "  SLURM_LOG_DIR:   $SLURM_LOG_DIR"
echo "  Input:           $NOTEBOOK"
echo "  Output dir:      $REPO_ROOT/outputs/  (timestamped ${BASENAME}_*.ipynb)"
echo "  partition:       $PARTITION"
echo "  GPUs:            ${NUM_GPUS} x ${GPU_TYPE}"
echo "  MAX_VAL_BATCHES: ${MAX_VAL_BATCHES:-<all val batches>}"
echo "  KEEP_EXEC_OUTPUTS: ${KEEP_EXEC_OUTPUTS:-1}"

mkdir -p "$SLURM_LOG_DIR" "$REPO_ROOT/outputs"

LDM_DATA_ROOT_DEFAULT="$REPO_ROOT/LDM-downscaling/full_Dataset"
LDM_DATA_RESOLVED="${LDM_DATA_ROOT:-$LDM_DATA_ROOT_DEFAULT}"
if [[ ! -f "$LDM_DATA_RESOLVED/normalization_data.pkl" ]]; then
    echo "ERROR: Missing normalization_data.pkl under data root:" >&2
    echo "  $LDM_DATA_RESOLVED" >&2
    echo "Set LDM_DATA_ROOT to your full_Dataset directory." >&2
    exit 1
fi

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
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<default>}"

cd "$REPO_ROOT"
mkdir -p "$SLURM_LOG_DIR" "$REPO_ROOT/outputs"

export PROJECT_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export LDM_DATA_ROOT="${LDM_DATA_ROOT:-$REPO_ROOT/LDM-downscaling/full_Dataset}"
export MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"

source .venv/bin/activate

OUT_DIR="$REPO_ROOT/outputs"
OUT_FILE="${BASENAME}_\$(date +%Y%m%d_%H%M%S).ipynb"
TMP_OUT_FILE=".tmp_\${OUT_FILE}"
mkdir -p "\$OUT_DIR"

export TQDM_DISABLE=1
export MPLBACKEND=module://matplotlib_inline.backend_inline

TMP_NB="\$(mktemp --suffix=.ipynb)"
cp "$NOTEBOOK" "\$TMP_NB"
python "$REPO_ROOT/scripts/sanitize_notebook.py" "\$TMP_NB" --clear-outputs

jupyter nbconvert \\
    --to notebook \\
    --execute "\$TMP_NB" \\
    --output-dir "\$OUT_DIR" \\
    --output "\$TMP_OUT_FILE" \\
    --ExecutePreprocessor.timeout=-1 \\
    --ExecutePreprocessor.iopub_timeout=86400 \\
    --ExecutePreprocessor.store_widget_state=False \\
    --ExecutePreprocessor.kernel_name=python3

rm -f "\$TMP_NB"

python -c "import json; json.load(open('$REPO_ROOT/outputs/\$TMP_OUT_FILE', encoding='utf-8')); print('Validated executed notebook JSON')"

KEEP_EXEC_OUTPUTS="${KEEP_EXEC_OUTPUTS:-1}"
if [[ "\$KEEP_EXEC_OUTPUTS" == "1" ]]; then
    mv "\$OUT_DIR/\$TMP_OUT_FILE" "\$OUT_DIR/\$OUT_FILE"
else
    jupyter nbconvert \\
        --to notebook \\
        --ClearOutputPreprocessor.enabled=True \\
        --output-dir "\$OUT_DIR" \\
        --output "\$OUT_FILE" \\
        "\$OUT_DIR/\$TMP_OUT_FILE"
    rm -f "\$OUT_DIR/\$TMP_OUT_FILE"
fi
echo "Done. Output: \$OUT_DIR/\$OUT_FILE"
echo "Stats JSON: \$OUT_DIR/mf_loss_probe_stats.json (if probe completed)"
echo "Histogram:  \$OUT_DIR/mf_loss_probe_dev_hist.png (if probe completed)"
EOF
