#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=deepqbert
#SBATCH --output=train-%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=9

module purge

export PATH=$HOME/.local/bin:$PATH

# Uncomment to ensure standard python
# uv python install --reinstall

echo "Setting up environment..."
uv venv
source .venv/bin/activate

echo "Syncing packages..."
uv sync

echo "Running training script..."
uv run deepqbert.py --load-checkpoint latest --num-episodes 1000000 --checkpoint-freq 20 --max-frames 10000000 --env-name ALE/Qbert-v5

deactivate