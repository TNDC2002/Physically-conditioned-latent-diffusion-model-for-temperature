#!/bin/bash
# Request an INTERACTIVE GPU session for running Jupyter and executing notebooks block-by-block.
# After this script runs, you will get a shell ON a compute node. Then start Jupyter there.
#
# Usage: bash configs/experiment/Submitscript_jupyter_interactive.sh
# Then in the new shell: cd $REPO_ROOT && export PYTHONPATH=$REPO_ROOT:$PYTHONPATH && .venv/bin/jupyter notebook --no-browser --port=8888

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PARTITION="${PARTITION:-main}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3}"
MEM="${MEM:-64G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
TIME="${TIME:-2:00:00}"

echo "Requesting interactive GPU node (partition=$PARTITION, gpu=$GPU_TYPE, time=$TIME)."
echo "After you get a shell on the compute node, run:"
echo "  cd $REPO_ROOT && export PROJECT_ROOT=$REPO_ROOT PYTHONPATH=$REPO_ROOT:\$PYTHONPATH && .venv/bin/jupyter notebook --no-browser --port=8888"
echo "Then from your local machine: ssh -L 8888:localhost:8888 $(whoami)@<LOGIN_NODE>"
echo ""

salloc \
    --partition="$PARTITION" \
    --gres=gpu:${GPU_TYPE}:${NUM_GPUS} \
    --mem="$MEM" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --time="$TIME"
