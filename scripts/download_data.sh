#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root regardless of current working directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/datasets/TMQM_SPECTO/raw"
BLOCKS_DIR="${RAW_DIR}/tmc_blocks"

mkdir -p "${RAW_DIR}" "${BLOCKS_DIR}"

download_file() {
    local url="$1"
    local dest="$2"

    if [[ -f "${dest}" ]]; then
        echo "[INFO] File already exists, skipping download: ${dest}"
        return 0
    fi

    echo "[INFO] Downloading ${url} ->${dest}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar "${url}" -o "${dest}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${dest}" "${url}"
    else
        echo "[ERROR] Neither 'curl' nor 'wget' is installed. Please install one and retry." >&2
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# 1. Download tmQMg* spectroscopic data (tmqmg_star.csv)
# Source: https://github.com/uiocompcat/tmQMg_star/blob/main/tmQMg*.csv
# -----------------------------------------------------------------------------
TMQMG_STAR_URL="https://raw.githubusercontent.com/uiocompcat/tmQMg_star/main/tmQMg%2A.csv"
TMQMG_STAR_PATH="${RAW_DIR}/tmqmg_star.csv"

download_file "${TMQMG_STAR_URL}" "${TMQMG_STAR_PATH}"

# -----------------------------------------------------------------------------
# 2. Download tmQM geometries & metadata (tmqm_all.csv)
# -----------------------------------------------------------------------------
# TODO: Replace TMQM_ALL_URL with the verified download link for tmqm_all.csv
TMQM_ALL_URL=""
TMQM_ALL_PATH="${RAW_DIR}/tmqm_all.csv"

if [[ -n "${TMQM_ALL_URL}" ]]; then
    download_file "${TMQM_ALL_URL}" "${TMQM_ALL_PATH}"
elif [[ ! -f "${TMQM_ALL_PATH}" ]]; then
    echo "[TODO] Automated download URL for 'tmqm_all.csv' is not configured yet."
    echo "       Please place 'tmqm_all.csv' manually at: ${TMQM_ALL_PATH}"
    echo "       Then re-run this script to generate d-block splits."
    exit 1
else
    echo "[INFO] Found existing ${TMQM_ALL_PATH}"
fi

# -----------------------------------------------------------------------------
# 3. Split transition metal complexes into 3d / 4d / 5d blocks
# Expected outputs:
#   - ${BLOCKS_DIR}/3d.csv
#   - ${BLOCKS_DIR}/4d.csv
#   - ${BLOCKS_DIR}/5d.csv
# -----------------------------------------------------------------------------
SPLIT_SCRIPT="${REPO_ROOT}/scripts/split_tmc_blocks.py"

if [[ -f "${SPLIT_SCRIPT}" ]]; then
    echo "[INFO] Running ${SPLIT_SCRIPT}..."
    python "${SPLIT_SCRIPT}"
else
    echo "[TODO] Split script not found at: ${SPLIT_SCRIPT}"
    echo "       Once added, run: python scripts/split_tmc_blocks.py"
    exit 1
fi

echo "[DONE] Raw dataset preparation complete."
