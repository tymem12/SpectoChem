#!/bin/bash

# Define the lists of parameters
MODELS=("invariant_projection" "geometric_gating" "moe")
EXPERIMENTS=("individual_pairs_f" "individual_pairs_lambda")
OUTLIER_FLAGS=("--outliers" "--no-outliers")

DIR="$(dirname "$0")"

# Loop through all combinations and submit
for model in "${MODELS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        for outlier_flag in "${OUTLIER_FLAGS[@]}"; do
            
            echo "Submitting: $model | $exp | $outlier_flag"
            sh "$DIR/submit_wrapper.sh" --model-name "$model" --exp "$exp" $outlier_flag
            
            # Optional: Add a tiny sleep to avoid overwhelming the slurm scheduler instantly
            sleep 1
            
        done
    done
done

echo "All jobs submitted!"