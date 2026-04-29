#!/bin/bash
#SBATCH --job-name=deltabot_data
#SBATCH --partition=llnn
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/deltabot_%j.out
#SBATCH --error=logs/deltabot_%j.err

set -e

echo "Job started on $(hostname)"
echo "Date: $(date)"

# environment
module purge
module load python/3.10
module load cuda/12
conda activate deltabot

echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

python ml4deltabot/deltabot_data.py

echo "Job finished at $(date)"
