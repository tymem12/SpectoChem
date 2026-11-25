#!/bin/bash
set -euo pipefail

MODELS=("schnet" "gine" "gat" "gcn")
SEEDS=(2137 42 1234)
BLOCK_SPLITS=("345" "3test" "none")

# Declarative intra-sweep dependencies: split -> prerequisite split (must appear earlier in BLOCK_SPLITS)
declare -A SPLIT_DEPENDS_ON=(
    ["3test"]="345"
)

DIR="$(cd "$(dirname "$0")" && pwd)"

EXP=""
EXTRA_DEPENDENCY=""
EXTRA_PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)
            EXP="$2"
            shift 2
            ;;
        --dependency)
            EXTRA_DEPENDENCY="$2"
            shift 2
            ;;
        *)
            EXTRA_PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$EXP" ]]; then
    echo "Usage: bash $0 --exp <binary|f_regression|lambda_regression|multi_regression> [--dependency <slurm_dep>] [extra_python_args...]" >&2
    exit 1
fi

declare -A job_ids=()

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "==== CONFIG: $model | $EXP | seed=$seed ===="
        job_ids=()

        for block in "${BLOCK_SPLITS[@]}"; do
            dep_parts=()

            # Optional external dependency passed to this helper (e.g., preprocessing job)
            if [[ -n "$EXTRA_DEPENDENCY" ]]; then
                dep_parts+=("$EXTRA_DEPENDENCY")
            fi

            # Automatic dependency on an earlier split in the same (model, seed) group
            parent_split="${SPLIT_DEPENDS_ON[$block]:-}"
            if [[ -n "$parent_split" ]]; then
                parent_job_id="${job_ids[$parent_split]:-}"
                if [[ -z "$parent_job_id" ]]; then
                    echo "[ERROR] Split '$block' requires '$parent_split' to appear earlier in BLOCK_SPLITS." >&2
                    exit 1
                fi
                dep_parts+=("afterok:${parent_job_id}")
            fi

            # Join multiple dependencies with ',' (Slurm requires all comma-separated conditions)
            joined_dep=""
            wrapper_args=()
            if [[ ${#dep_parts[@]} -gt 0 ]]; then
                joined_dep=$(IFS=,; echo "${dep_parts[*]}")
                wrapper_args+=(--dependency "$joined_dep")
            fi

            wrapper_args+=(
                --model-name "$model"
                --exp "$EXP"
                --block_3_split "$block"
                --seed "$seed"
            )

            if [[ ${#EXTRA_PYTHON_ARGS[@]} -gt 0 ]]; then
                wrapper_args+=("${EXTRA_PYTHON_ARGS[@]}")
            fi

            job_id=$(bash "$DIR/submit_wrapper.sh" "${wrapper_args[@]}")
            job_ids["$block"]="$job_id"

            echo "  -> block=${block}: job_id=${job_id}${joined_dep:+ (dep: ${joined_dep})}"
            sleep 1
        done
    done
done

echo "All ${EXP} jobs submitted!"
