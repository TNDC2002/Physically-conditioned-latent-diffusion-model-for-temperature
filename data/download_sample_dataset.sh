#!/bin/bash

# Download to a mount/large storage to avoid quota. Set via env or use repo's LDM-downscaling.
# Example: DOWNLOAD_DIR=/mnt/LDM-downscaling bash download_sample_dataset.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$REPO_ROOT/LDM-downscaling}"

mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# download data via SSH reverse proxy (zenodo.org -> localhost:8080)
# -c/--continue: resume partial download if sample_dataset.zip already exists here
wget --no-check-certificate \
     --header="Host: zenodo.org" \
     --continue \
     "https://localhost:8080/records/12934521/files/sample_data.zip?download=1" \
     -O sample_dataset.zip

# unzip data (into DOWNLOAD_DIR)
unzip -o sample_dataset.zip -d ./
rm sample_dataset.zip

# Symlink so repo's data/ still sees the sample (configs use data_dir: .../data/)
DATA_LINK="$SCRIPT_DIR/sample_dataset"
rm -f "$DATA_LINK"
if [ -d "$DOWNLOAD_DIR/sample_data" ]; then
    ln -sf "$DOWNLOAD_DIR/sample_data" "$DATA_LINK"
    echo "Created symlink: $DATA_LINK -> $DOWNLOAD_DIR/sample_data"
elif [ -d "$DOWNLOAD_DIR/sample_dataset" ]; then
    ln -sf "$DOWNLOAD_DIR/sample_dataset" "$DATA_LINK"
    echo "Created symlink: $DATA_LINK -> $DOWNLOAD_DIR/sample_dataset"
else
    echo "Extracted folder not found as sample_data or sample_dataset in $DOWNLOAD_DIR"
fi
