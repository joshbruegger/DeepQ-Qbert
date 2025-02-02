import numpy as np

import globals as g
import plotter
from env_manager import EnvManager
from model import DQN
from replay_memory import ReplayMemory
from train import train

# Make the environment
envManager = EnvManager()

# Make the network
n_actions = envManager.env.action_space.n
network = DQN(g.QUEUE_N_FRAMES, n_actions).to(g.DEVICE)

# Make the memory
memory = ReplayMemory(10000)

episodes_rewards = train(
    envManager=envManager,
    network=network,
    num_episodes=5,
    memory=memory,
    checkpoint_dir="checkpoints",
    checkpoint_freq=2,  # Save every 2 episodes since we're only running 5 episodes
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
        filepath="plots/rewards.png",
    ),
)

# Close the environment
envManager.env.close()
