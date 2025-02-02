#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=deepqbert
#SBATCH --output=train-%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=9

echo "Loading modules..."
module update
module purge

module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load OpenCV/4.6.0-foss-2022a-contrib
module load Boost/1.79.0-GCC-11.3.0

module list

export PATH=$HOME/.local/bin:$PATH

uv sync

uv run deepqbert.py

deactivate