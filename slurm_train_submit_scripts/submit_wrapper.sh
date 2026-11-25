#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

DEPENDENCY=""
PYTHON_ARGS=()

# Separate optional --dependency flag from arguments forwarded to script_train_executer.py
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dependency)
            DEPENDENCY="$2"
            shift 2
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build folder path and job name from Python arguments only
PATH_SUFFIX=""
JOB_NAME_SUFFIX=""

for arg in "${PYTHON_ARGS[@]}"; do
    if [[ "$arg" == --* ]]; then
        clean_arg="${arg#--}"
        clean_arg=$(echo "$clean_arg" | sed 's/[^a-zA-Z0-9_-]/_/g')
        PATH_SUFFIX="${PATH_SUFFIX}/${clean_arg}"
        if [[ -z "$JOB_NAME_SUFFIX" ]]; then
            JOB_NAME_SUFFIX="${clean_arg}"
        else
            JOB_NAME_SUFFIX="${JOB_NAME_SUFFIX}_${clean_arg}"
        fi
    else
        clean_arg=$(echo "$arg" | sed 's/[^a-zA-Z0-9_-]/_/g')
        PATH_SUFFIX="${PATH_SUFFIX}/${clean_arg}"
        JOB_NAME_SUFFIX="${JOB_NAME_SUFFIX}-${clean_arg}"
    fi
done

# Create output/error directories
OUT_DIR="output${PATH_SUFFIX}"
ERR_DIR="error${PATH_SUFFIX}"
mkdir -p "$OUT_DIR" "$ERR_DIR"

OUT_FILE="${OUT_DIR}/%j.log"
ERR_FILE="${ERR_DIR}/%j.log"
FINAL_JOB_NAME="train_${JOB_NAME_SUFFIX}"

# Assemble sbatch flags
SBATCH_ARGS=(
    --parsable
    --job-name="$FINAL_JOB_NAME"
    --output="$OUT_FILE"
    --error="$ERR_FILE"
)

if [[ -n "$DEPENDENCY" ]]; then
    SBATCH_ARGS+=(
        --dependency="$DEPENDENCY"
        --kill-on-invalid-dep=yes
    )
fi

# Send human-readable logs to stderr so stdout only contains the job ID
echo "Submitting job: $FINAL_JOB_NAME${DEPENDENCY:+ (dep: $DEPENDENCY)}" >&2
echo "Output mapped to: $OUT_FILE" >&2

# Submit and strip any ';cluster_name' suffix returned by multi-cluster Slurm (e.g. PLGrid)
RAW_JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "$DIR/submit.sh" "${PYTHON_ARGS[@]}")
JOB_ID="${RAW_JOB_ID%%;*}"

echo "$JOB_ID"
