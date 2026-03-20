#!/bin/bash
#SBATCH --partition=plgrid-lem-gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:hopper:1

source /etc/profile

module load Python/3.11.3-GCCcore-12.3.0

source .venv/bin/activate

# "$@" passes all command-line arguments straight to the Python script
PYTHONPATH=. python script_train_executer.py "$@"

deactivate