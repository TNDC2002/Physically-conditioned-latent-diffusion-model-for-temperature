#!/bin/bash

# Download to a mount/large storage. Set via env or use repo's LDM-downscaling.
# Example: DOWNLOAD_DIR=/mnt/LDM-downscaling bash download_full_dataset.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$REPO_ROOT/LDM-downscaling}"

mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# download data via SSH reverse proxy (zenodo.org -> localhost:8080)
# --continue: resume partial downloads if .zip files already exist
wget --no-check-certificate --header="Host: zenodo.org" --continue "https://zenodo.org/records/12944960/files/2000-2002.zip?download=1" -O 2000-2002.zip
wget --no-check-certificate --header="Host: zenodo.org" --continue "https://zenodo.org/records/12945014/files/2003-2005.zip?download=1" -O 2003-2005.zip
wget --no-check-certificate --header="Host: zenodo.org" --continue "https://zenodo.org/records/12945028/files/2006-2008.zip?download=1" -O 2006-2008.zip
wget --no-check-certificate --header="Host: zenodo.org" --continue "https://zenodo.org/records/12945040/files/2009-2011.zip?download=1" -O 2009-2011.zip
wget --no-check-certificate --header="Host: zenodo.org" --continue "https://zenodo.org/records/12945050/files/2012-2014.zip?download=1" -O 2012-2014.zip
wget --no-check-certificate --header="Host: zenodo.org" --continue "https://zenodo.org/records/12945058/files/2015-2017.zip?download=1" -O 2015-2017.zip
wget --no-check-certificate --header="Host: zenodo.org" --continue "https://zenodo.org/records/12945066/files/2018-2020.zip?download=1" -O 2018-2020.zip

# unzip data (into DOWNLOAD_DIR)
unzip -o 2000-2002.zip -d ./
rm 2000-2002.zip
unzip -o 2003-2005.zip -d ./
rm 2003-2005.zip
unzip -o 2006-2008.zip -d ./
rm 2006-2008.zip
unzip -o 2009-2011.zip -d ./
rm 2009-2011.zip
unzip -o 2012-2014.zip -d ./
rm 2012-2014.zip
unzip -o 2015-2017.zip -d ./
rm 2015-2017.zip
unzip -o 2018-2020.zip -d ./
rm 2018-2020.zip

echo "Full dataset extracted under $DOWNLOAD_DIR"
echo "Point your config data_dir to $DOWNLOAD_DIR or symlink from $SCRIPT_DIR as needed."
