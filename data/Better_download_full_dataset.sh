#!/bin/bash
set -e

# Download to a mount/large storage. Set via env or use repo's LDM-downscaling.
# Example: DOWNLOAD_DIR=/mnt/LDM-downscaling bash Better_download_full_dataset.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$REPO_ROOT/LDM-downscaling/better_full_dataset}"

mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# Zenodo download mode:
#   USE_ZENODO_SSH_PROXY=0 (default) — direct HTTPS to zenodo.org (no tunnel).
#   USE_ZENODO_SSH_PROXY=1 — via SSH reverse proxy (zenodo.org -> localhost:8080, Host header).
USE_ZENODO_SSH_PROXY="${USE_ZENODO_SSH_PROXY:-0}"

# Function to download a file with robust options.
# -c: continue partial download if file already exists
download_file() {
    local url="$1"
    local outfile="$2"
    echo "Downloading $outfile ..."

    if [[ "${USE_ZENODO_SSH_PROXY}" == "1" ]]; then
        # Route through the SSH reverse proxy (keep tunnel listening on localhost:8080)
        local proxied_url="${url/https:\/\/zenodo.org/https:\/\/localhost:8080}"
        wget --no-check-certificate \
             --header="Host: zenodo.org" \
             -c -t 0 --waitretry=5 --read-timeout=20 --timeout=30 \
             "$proxied_url" -O "$outfile"
    else
        wget -c -t 0 --waitretry=5 --read-timeout=20 --timeout=30 \
             "$url" -O "$outfile"
    fi
}

# Download data files from Zenodo (into DOWNLOAD_DIR)
download_file "https://zenodo.org/records/12944960/files/2000-2002.zip?download=1" "2000-2002.zip"
download_file "https://zenodo.org/records/12945014/files/2003-2005.zip?download=1" "2003-2005.zip"
download_file "https://zenodo.org/records/12945028/files/2006-2008.zip?download=1" "2006-2008.zip"
download_file "https://zenodo.org/records/12945040/files/2009-2011.zip?download=1" "2009-2011.zip"
download_file "https://zenodo.org/records/12945050/files/2012-2014.zip?download=1" "2012-2014.zip"
download_file "https://zenodo.org/records/12945058/files/2015-2017.zip?download=1" "2015-2017.zip"
download_file "https://zenodo.org/records/12945066/files/2018-2020.zip?download=1" "2018-2020.zip"

# Unzip each downloaded file and remove the zip file afterwards
for file in *.zip; do
    echo "Unzipping $file ..."
    unzip -o "$file" -d ./
    rm "$file"
done

echo "All files downloaded and extracted under $DOWNLOAD_DIR"
