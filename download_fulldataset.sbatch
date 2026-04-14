#!/bin/bash
#SBATCH --job-name=ldm_download
#SBATCH --output=/home/chuongtnd/slurm-logs/%x-%j.out
#SBATCH --error=/home/chuongtnd/slurm-logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

# Use Slurm's submit directory so we write into the real repo,
# not the node-local spool/work directory.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT"

# Hard-coded destination directory under repository root.
DOWNLOAD_DIR="$REPO_ROOT/LDM-downscaling/full_Dataset"

mkdir -p "$DOWNLOAD_DIR"

echo "Node: $(hostname)"
echo "PWD: $(pwd)"
echo "Repo: $REPO_ROOT"
echo "DOWNLOAD_DIR=$DOWNLOAD_DIR"

# Try modules if available; ignore if your cluster doesn't use environment modules.
module -q load wget unzip 2>/dev/null || true

DOWNLOAD_DIR="$DOWNLOAD_DIR" bash data/download_full_dataset.sh
