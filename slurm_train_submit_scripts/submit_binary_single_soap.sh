#!/bin/bash

BLOCK_SPLITS=("none" "test")
SEEDS=(2137 42 1234 420 1337)
NORM_EV_OPTS=("True" "False") 
R_CUTS=(10.0)
NL_PAIRS=("4 2")
MODELS=("xgboost" "random_forest" "svm" "mlp" "logistic_regression" "dummy")

DIR="$(dirname "$0")"

for model in "${MODELS[@]}"; do
    for r in "${R_CUTS[@]}"; do
        for nl in "${NL_PAIRS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for block_split in "${BLOCK_SPLITS[@]}"; do
                    read -r n l <<< "$nl"

                    echo "==== SUBMITTING: model=$model | r_cut=$r | n_max=$n | l_max=$l | seed=$seed | block_3_split=$block_split ===="

                    sh "$DIR/submit_wrapper.sh" \
                        --r_cut "$r" \
                        --n_max "$n" \
                        --l_max "$l" \
                        --seed "$seed" \
                        --block_3_split "$block_split" \
                        --model "$model" \

                    sleep 1
                done
            done
        done
    done
done
echo "All jobs for all models submitted!"