#!/bin/bash
set -e

# Download to a mount/large storage. Set via env or use repo's LDM-downscaling.
# Example: DOWNLOAD_DIR=/mnt/LDM-downscaling bash Better_download_full_dataset.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$REPO_ROOT/LDM-downscaling/better_full_dataset}"

# Disk / resume behaviour (helps when the filesystem fills mid-job):
#   WAIT_FOR_DISK=1 (default) — poll until free space is enough, then continue; retry wget after failures.
#   WAIT_FOR_DISK=0 — no waiting (original behaviour aside from per-archive ordering).
#   MIN_FREE_GB — headroom required on the DOWNLOAD_DIR filesystem before starting a download or unzip.
#   DISK_POLL_INTERVAL_SEC — sleep between disk checks when waiting.
#   MAX_DISK_WAIT_SEC — give up waiting for disk (0 = never; wait until space appears).
#   MAX_WGET_ATTEMPTS — wget retries after failure (0 = unlimited; uses disk wait between tries).
#   UNZIP_SPACE_FACTOR — before unzip, require min_free + (zip size × this factor) free bytes (expansion buffer).
MIN_FREE_GB="${MIN_FREE_GB:-2}"
DISK_POLL_INTERVAL_SEC="${DISK_POLL_INTERVAL_SEC:-120}"
MAX_DISK_WAIT_SEC="${MAX_DISK_WAIT_SEC:-0}"
WAIT_FOR_DISK="${WAIT_FOR_DISK:-1}"
MAX_WGET_ATTEMPTS="${MAX_WGET_ATTEMPTS:-0}"
UNZIP_SPACE_FACTOR="${UNZIP_SPACE_FACTOR:-3}"

mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# Zenodo download mode:
#   USE_ZENODO_SSH_PROXY=0 (default) — direct HTTPS to zenodo.org (no tunnel).
#   USE_ZENODO_SSH_PROXY=1 — via SSH reverse proxy (zenodo.org -> localhost:8080, Host header).
USE_ZENODO_SSH_PROXY="${USE_ZENODO_SSH_PROXY:-0}"

min_free_bytes() {
    echo $((MIN_FREE_GB * 1024 * 1024 * 1024))
}

# Free space (bytes) on the filesystem that holds DOWNLOAD_DIR (POSIX df -k).
avail_bytes() {
    df -Pk "$DOWNLOAD_DIR" | awk 'NR==2 {print ($4 + 0) * 1024}'
}

# Block until at least $1 bytes are free, or MAX_DISK_WAIT_SEC exceeded (if set).
wait_for_disk_space() {
    local need="${1:?}"
    [[ "$WAIT_FOR_DISK" == "1" ]] || return 0
    local start
    start=$(date +%s)
    while true; do
        local avail
        avail=$(avail_bytes)
        if ((avail >= need)); then
            echo "Disk OK under $DOWNLOAD_DIR: ${avail} bytes free (required ${need})."
            return 0
        fi
        echo "Low disk under $DOWNLOAD_DIR: ${avail} bytes free, need ${need}. Next check in ${DISK_POLL_INTERVAL_SEC}s."
        if ((MAX_DISK_WAIT_SEC > 0)); then
            local now
            now=$(date +%s)
            if ((now - start >= MAX_DISK_WAIT_SEC)); then
                echo "Exceeded MAX_DISK_WAIT_SEC=${MAX_DISK_WAIT_SEC} while waiting for disk space." >&2
                return 1
            fi
        fi
        sleep "$DISK_POLL_INTERVAL_SEC"
    done
}

# Run wget with proxy/direct logic; stdout/stderr as usual.
_wget_zenodo() {
    local url="$1"
    local outfile="$2"
    if [[ "${USE_ZENODO_SSH_PROXY}" == "1" ]]; then
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

# Download one file; optionally wait for disk and retry wget until success (-c resumes partial files).
download_file() {
    local url="$1"
    local outfile="$2"
    local need
    need=$(min_free_bytes)
    echo "Downloading $outfile ..."
    wait_for_disk_space "$need" || return 1

    local attempt=0
    while true; do
        if _wget_zenodo "$url" "$outfile"; then
            return 0
        fi
        attempt=$((attempt + 1))
        if [[ "$MAX_WGET_ATTEMPTS" != "0" && "$attempt" -ge "$MAX_WGET_ATTEMPTS" ]]; then
            echo "wget gave up after ${MAX_WGET_ATTEMPTS} attempt(s) for $outfile." >&2
            return 1
        fi
        echo "wget failed for $outfile (attempt ${attempt}); waiting for disk space then retrying (-c resumes) ..."
        sleep "$DISK_POLL_INTERVAL_SEC"
        wait_for_disk_space "$need" || return 1
    done
}

# Unzip then remove zip; wait for space first (extracted size is often similar to or larger than the zip).
unzip_and_remove() {
    local zip="$1"
    local zipsize need
    zipsize=$(stat -c '%s' "$zip")
    # Rough buffer: extracted footprint often within ~2x compressed size for many datasets.
    need=$(($(min_free_bytes) + zipsize * 2))
    echo "Unzipping $zip ..."
    wait_for_disk_space "$need" || return 1
    unzip -o "$zip" -d ./
    rm -f "$zip"
}

# Download → unzip → delete zip for each archive (lower peak disk than downloading all zips first).
process_archive() {
    local url="$1"
    local zip="$2"
    download_file "$url" "$zip" || return 1
    unzip_and_remove "$zip" || return 1
}

process_archive "https://zenodo.org/records/12944960/files/2000-2002.zip?download=1" "2000-2002.zip"
process_archive "https://zenodo.org/records/12945014/files/2003-2005.zip?download=1" "2003-2005.zip"
process_archive "https://zenodo.org/records/12945028/files/2006-2008.zip?download=1" "2006-2008.zip"
process_archive "https://zenodo.org/records/12945040/files/2009-2011.zip?download=1" "2009-2011.zip"
process_archive "https://zenodo.org/records/12945050/files/2012-2014.zip?download=1" "2012-2014.zip"
process_archive "https://zenodo.org/records/12945058/files/2015-2017.zip?download=1" "2015-2017.zip"
process_archive "https://zenodo.org/records/12945066/files/2018-2020.zip?download=1" "2018-2020.zip"

echo "All files downloaded and extracted under $DOWNLOAD_DIR"
