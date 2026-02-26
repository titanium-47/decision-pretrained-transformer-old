#!/bin/bash
#SBATCH --job-name=klsweep
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Initialize conda (adjust path if your conda install is elsewhere)
source /gscratch/weirdlab/sidhraja/miniconda3/etc/profile.d/conda.sh
conda activate dpt


# Run the same bash script you were using interactively
bash /gscratch/weirdlab/sidhraja/projects/decision-pretrained-transformer-old/run_kl_sweep.sh