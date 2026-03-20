#!/bin/bash

# 1. Loop through all passed arguments to build the folder path and job name
PATH_SUFFIX=""
JOB_NAME_SUFFIX=""

DIR="$(dirname "$0")"

for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        # It's a parameter key (starts with --)
        clean_arg="${arg#--}"
        clean_arg=$(echo "$clean_arg" | sed 's/[^a-zA-Z0-9_-]/_/g')
        
        PATH_SUFFIX="${PATH_SUFFIX}/${clean_arg}"
        
        if [ -z "$JOB_NAME_SUFFIX" ]; then
            JOB_NAME_SUFFIX="${clean_arg}"
        else
            JOB_NAME_SUFFIX="${JOB_NAME_SUFFIX}_${clean_arg}"
        fi
    else
        # It's a parameter value
        clean_arg=$(echo "$arg" | sed 's/[^a-zA-Z0-9_-]/_/g')
        
        PATH_SUFFIX="${PATH_SUFFIX}/${clean_arg}"
        JOB_NAME_SUFFIX="${JOB_NAME_SUFFIX}-${clean_arg}"
    fi
done

# 2. Define full output and error directories
OUT_DIR="output${PATH_SUFFIX}"
ERR_DIR="error${PATH_SUFFIX}"

# 3. Slurm needs the folders to exist before starting the job, so we create them here
mkdir -p "$OUT_DIR"
mkdir -p "$ERR_DIR"

# 4. Define the final file paths and job name
OUT_FILE="${OUT_DIR}/%j.log"
ERR_FILE="${ERR_DIR}/%j.log"
FINAL_JOB_NAME="train_${JOB_NAME_SUFFIX}"

echo "Submitting job: $FINAL_JOB_NAME"
echo "Output mapped to: $OUT_FILE"

# 5. Submit the job! Command-line flags override the #SBATCH lines in submit.sh
sbatch --job-name="$FINAL_JOB_NAME" --output="$OUT_FILE" --error="$ERR_FILE" "$DIR/submit.sh" "$@"