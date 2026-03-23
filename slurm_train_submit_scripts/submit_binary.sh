#!/bin/bash

# Single-element variables as requested
BLOCK_SPLIT="none"
SEED=1234

# Grid parameters
R_CUTS=(5.0 10.0)

# Pairs of "n_max l_max" designed to hit your target feature sizes
# Formula for 14 species: Features = (l + 1) * (14*n * (14*n + 1)) / 2
NL_PAIRS=(
    "1 4" # Yields ~525 features   (Target: 500)
    "1 9" # Yields ~1,050 features (Target: 1k)
    "2 4" # Yields ~2,030 features (Target: 2k)
    "3 4" # Yields ~4,515 features (Target: 5k)
)

DIR="$(dirname "$0")"

for r in "${R_CUTS[@]}"; do
    for nl in "${NL_PAIRS[@]}"; do
        
        # Split the string into n and l variables
        read -r n l <<< "$nl"
        
        echo "==== CONFIG: r_cut=$r | n_max=$n | l_max=$l | seed=$SEED | block=$BLOCK_SPLIT ===="
        
        # Call your wrapper (passing the new arguments)
        sh "$DIR/submit_wrapper.sh" \
            --r_cut "$r" \
            --n_max "$n" \
            --l_max "$l" \
            --seed "$SEED" \
            --block_3_split "$BLOCK_SPLIT"
        
        sleep 1
    done
done

echo "All SOAP XGBoost jobs submitted!"