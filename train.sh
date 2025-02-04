#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --job-name=deepqbert
#SBATCH --output=train-%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=20G

export PYTHONUNBUFFERED=TRUE

# Get arguments from command line
while getopts ":e:" opt; do
    case ${opt} in
        e)
            ENV_NAME=$OPTARG
            ;;
    esac
done

if [ -z "$ENV_NAME" ]; then
    echo "Usage: $0 -e <env_name>"
    exit 1
fi

if [ "$ENV_NAME" == "pong" ]; then
    ENV_NAME="ALE/Pong-v5"
elif [ "$ENV_NAME" == "qbert" ]; then
    ENV_NAME="ALE/Qbert-v5"
else
    echo "Invalid environment name"
    exit 1
fi

module purge

# save the current directory
START_DIR=$(pwd)

# Ensure we can use uv
export PATH=$HOME/.local/bin:$PATH

# Uncomment to ensure standard python
# uv python install --reinstall

# Sanitize environment name for folder naming
SANITIZED_ENV_NAME=$(echo "$ENV_NAME" | tr -cd '[:alnum:]_.-')

# create the environment directory if it doesn't exist
mkdir -p $HOME/.deepqbert/$SANITIZED_ENV_NAME

# copy the working directory to the HOME/.deepqbert space except for the .venv, __pycache__, .git, logs, and videos directories
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='logs' --exclude='videos' $(pwd)/* $HOME/.deepqbert/$SANITIZED_ENV_NAME/

cd $HOME/.deepqbert/$SANITIZED_ENV_NAME

echo "Setting up environment..."
uv venv
source .venv/bin/activate

echo "Syncing packages..."
uv sync

echo "Running training script..."
uv run python -u deepqbert.py --load-checkpoint latest --num-episodes 10000000 --checkpoint-freq 20 --max-frames 10000000 --env-name $ENV_NAME --no-recording --output-dir $START_DIR/output/$SANITIZED_ENV_NAME

deactivate