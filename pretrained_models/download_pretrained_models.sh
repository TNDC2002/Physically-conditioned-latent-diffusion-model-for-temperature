#!/bin/bash

# Download via SSH reverse proxy (zenodo.org -> localhost:8080). Same as data/*.sh — use when you cannot reach Zenodo directly.
# Ensure the proxy is running (e.g. ssh -R 8080:zenodo.org:443 your-jump-host) before running this script.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Main pretrained models (UNET_*.ckpt, VAE_residual_*.ckpt, LDM_residual_*.ckpt, etc.)
wget --no-check-certificate \
     --header="Host: zenodo.org" \
     --continue \
     "https://localhost:8080/records/12941117/files/pretrained_models.zip?download=1" \
     -O pretrained_models.zip

# unzip main pretrained models
unzip -o pretrained_models.zip -d ./
rm pretrained_models.zip

# PDE Loss Model (single .ckpt file, not a zip)
wget --no-check-certificate \
     --header="Host: zenodo.org" \
     --continue \
     "https://localhost:8080/records/15460090/files/pde_loss_model_checkpoint.ckpt?download=1" \
     -O pde_loss_model_checkpoint.ckpt

