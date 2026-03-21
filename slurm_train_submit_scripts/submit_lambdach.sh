#!/bin/bash

MODELS=("schnet" "gin" "gat" "gcn")
BLOCK_SPLITS=("none" "test")
SEEDS=(2137 42 1234 420 1337)

STANDARDIZATION_FLAGS=("--standarization" "--no-standarization")
NORMALIZE_FLAGS=("--normalize_eV" "--no-normalize_eV")

DIR="$(dirname "$0")"

for model in "${MODELS[@]}"; do
    for block in "${BLOCK_SPLITS[@]}"; do
        for std_flag in "${STANDARDIZATION_FLAGS[@]}"; do
            for norm_flag in "${NORMALIZE_FLAGS[@]}"; do

                echo "==== CONFIG: $model | block=$block | $std_flag | $norm_flag ===="

                for seed in "${SEEDS[@]}"; do

                    echo "Submitting: $model | lambda | block=$block | seed=$seed | $std_flag | $norm_flag"

                    sh "$DIR/submit_wrapper.sh" \
                        --model-name "$model" \
                        --exp lambda_regression \
                        --block_3_split "$block" \
                        --seed "$seed" \
                        $std_flag \
                        $norm_flag

                    sleep 1
                done
            done
        done
    done
done

echo "All lambda regression jobs submitted!"