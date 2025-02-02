import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import globals as g
import plotter
from env_manager import EnvManager
from model import DQN
from replay_memory import ReplayMemory
from train import train


def load_checkpoint(network: DQN, checkpoint_path: Path):
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Last Episode: {checkpoint['episode']}")
        if "best_reward" in checkpoint:
            print(f"Best reward: {checkpoint['best_reward']}")
        return (
            checkpoint["episodes_rewards"],
            checkpoint["episode"],
            checkpoint.get("checkpoint_rewards", {}),
        )
    return None, 0, {}


# Parse command line arguments
parser = argparse.ArgumentParser(description="Train DeepQbert")
parser.add_argument(
    "--load-checkpoint",
    choices=["best", "latest", "none"],
    default="none",
    help="Load from checkpoint: best model, latest model, or none",
)
parser.add_argument(
    "--num-episodes",
    type=int,
    default=5,
    help="Number of episodes to train for",
)
parser.add_argument(
    "--checkpoint-freq",
    type=int,
    default=2,
    help="Save checkpoint every N episodes",
)
args = parser.parse_args()

envManager = EnvManager()

# Make the network
network = DQN(g.QUEUE_N_FRAMES, envManager.env.action_space.n).to(g.DEVICE)

# Load from checkpoint if specified
checkpoint_dir = Path("checkpoints")
episodes_rewards = None
starting_episode = 0
checkpoint_rewards = {}

if args.load_checkpoint == "best":
    episodes_rewards, starting_episode, checkpoint_rewards = load_checkpoint(
        network, checkpoint_dir / "best_model.pt"
    )
elif args.load_checkpoint == "latest":
    episodes_rewards, starting_episode, checkpoint_rewards = load_checkpoint(
        network, checkpoint_dir / "latest_model.pt"
    )


envManager.setup_recording(starting_episode)

# Make the memory
memory = ReplayMemory(10000)

# Continue training from where we left off if we loaded a checkpoint
if episodes_rewards is None:
    episodes_rewards = []

episodes_rewards = np.concatenate(
    [
        episodes_rewards,
        train(
            envManager=envManager,
            network=network,
            num_episodes=args.num_episodes,
            memory=memory,
            checkpoint_dir="checkpoints",
            checkpoint_freq=args.checkpoint_freq,
            starting_episode=starting_episode + 1,
            checkpoint_rewards=checkpoint_rewards,
        ),
    ]
)

plotter.plot_data(
    x=np.arange(len(episodes_rewards)),
    y=episodes_rewards,
    config=plotter.PlotConfig(
        title="Episode Rewards",
        xlabel="Episode",
        ylabel="Reward",
        running_avg=True,
        window_size=100,
        filepath=f"plots/rewards_{len(episodes_rewards)}.png",
    ),
)

# Close the environment
envManager.env.close()
