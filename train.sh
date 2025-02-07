#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --job-name=deepqbert
#SBATCH --output=train-%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=30G

export PYTHONUNBUFFERED=TRUE

# Get arguments from command line
REINSTALL=false
while getopts ":e:r" opt; do
    case ${opt} in
        e)
            ENV_NAME=$OPTARG
            ;;
        r)
            REINSTALL=true
            ;;
    esac
done

if [ -z "$ENV_NAME" ]; then
    echo "Usage: $0 -e <env_name> [-r]"
    echo "  -e: Environment name (pong or qbert)"
    echo "  -r: Reinstall environment by removing existing folder"
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

# If reinstall flag is set, remove the existing environment directory
if [ "$REINSTALL" = true ] && [ -d "$HOME/.deepqbert/$SANITIZED_ENV_NAME" ]; then
    echo "Removing existing environment directory..."
    rm -rf "$HOME/.deepqbert/$SANITIZED_ENV_NAME"
fi

# create the environment directory if it doesn't exist
mkdir -p $HOME/.deepqbert/$SANITIZED_ENV_NAME

echo "Copying working directory to ~/.deepqbert..."
# copy the working directory to the HOME/.deepqbert space except for the .venv, __pycache__, .git, logs, and videos directories
rsync -av --exclude='output' --exclude='__pycache__' --exclude='.git' $(pwd)/* $HOME/.deepqbert/$SANITIZED_ENV_NAME/

echo "Changing directory to ~/.deepqbert/$SANITIZED_ENV_NAME"
cd $HOME/.deepqbert/$SANITIZED_ENV_NAME

echo "Setting up environment..."
# if the .venv directory doesn't exist, create it
if [ ! -d ".venv" ]; then
    uv venv
fi
source .venv/bin/activate

echo "Syncing packages..."
uv sync

echo "Running training script..."
uv run python -u deepqbert.py --resume --frames 10000000 --env $ENV_NAME --output $START_DIR/output/$SANITIZED_ENV_NAME

deactivate