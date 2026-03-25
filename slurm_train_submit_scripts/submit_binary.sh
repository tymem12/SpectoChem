#!/bin/bash

# Single-element variables as requested
BLOCK_SPLIT="none"
SEED=1234

# Models to run
MODELS=(
    #"xgboost"
    "random_forest"
    "svm"
    "logistic_regression"
    "mlp"
)

# Grid parameters
R_CUTS=(5.0 10.0)

# Pairs of "n_max l_max" designed to hit your target feature sizes
# Formula for 14 species: Features = (l + 1) * (14*n * (14*n + 1)) / 2
NL_PAIRS=(
    "1 4" # Yields ~525 features   (Target: 500)

    # --- 1k Features ---
    "1 9"  # 1,050 (Best balance)
    "3 0"  # 903   (High radial, no angular)
    "1 8"  # 945   (High angular)

    # --- 2k Features ---
    "2 4"  # 2,030 (Standard)
    "1 18" # 1,995 (Max angular resolution)
    "3 1"  # 1,806 (Max radial resolution)

    # --- 5k Features ---
    "5 1"  # 4,970 (Best radial focus)
    "2 11" # 4,872 (High angular focus)
    "7 0"  # 4,851 (Extreme radial, no angular)
    "4 2"  # 4,788 (Balanced)
    "3 4" # Yields ~4,515 features (Target: 5k)

    # --- 10k Features ---
    "5 3"  # 9,940  (Well-balanced)
    "3 10" # 9,933  (Angular heavy)
    "10 0" # 9,870  (Purely radial)
    "2 20" # 10,150 (Extreme angular)
    "7 1"  # 9,702  (Radial heavy)
)

DIR="$(dirname "$0")"

for model in "${MODELS[@]}"; do
    for r in "${R_CUTS[@]}"; do
        for nl in "${NL_PAIRS[@]}"; do
            
            # Split the string into n and l variables
            read -r n l <<< "$nl"
            
            echo "==== CONFIG: model=$model | r_cut=$r | n_max=$n | l_max=$l | seed=$SEED | block=$BLOCK_SPLIT ===="
            
            # Call your wrapper (passing the new arguments)
            sh "$DIR/submit_wrapper.sh" \
                --model-name "$model" \
                --r_cut "$r" \
                --n_max "$n" \
                --l_max "$l" \
                --seed "$SEED" \
                --block_3_split "$BLOCK_SPLIT"
            
            sleep 1
        done
    done
done

echo "All SOAP model jobs submitted!"