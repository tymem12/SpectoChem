#!/bin/bash

MODELS=("schnet" "gin" "gat" "gcn")
BLOCK_SPLITS=("none" "test")
SEEDS=(2137 42 1234 420 1337)

DIR="$(dirname "$0")"

for model in "${MODELS[@]}"; do
    for block in "${BLOCK_SPLITS[@]}"; do
        
        echo "==== CONFIG: $model | block=$block ===="
        
        for seed in "${SEEDS[@]}"; do
            
            echo "Submitting: $model | binary | block=$block | seed=$seed"
            
            sh "$DIR/submit_wrapper.sh" \
                --model-name "$model" \
                --exp binary \
                --block_3_split "$block" \
                --seed "$seed"
            
            sleep 1
        done
    done
done

echo "All binary jobs submitted!"