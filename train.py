from itertools import count
from pathlib import Path

import numpy as np
import torch
from progress_table import ProgressTable

import globals as g
from env_manager import EnvManager
from optimizer import optimize
from replay_memory import ReplayMemory


def train(
    envManager: EnvManager,
    network: torch.nn.Module,
    num_episodes: int,
    memory: ReplayMemory,
    checkpoint_dir: str = "checkpoints",
    checkpoint_freq: int = 100,  # Save every 100 episodes
    starting_episode: int = 0,  # Add starting episode parameter
    checkpoint_rewards: dict = None,  # Add checkpoint rewards parameter
):
    episodes_rewards = np.array([])
    best_reward = float("-inf")
    checkpoint_rewards = checkpoint_rewards or {}  # Initialize empty dict if None

    # Create checkpoint directory if it doesn't exist
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    # Initialize progress table
    table = ProgressTable(
        pbar_style="circle",
        pbar_embedded=False,
    )

    for episode in table(
        range(starting_episode, starting_episode + num_episodes), description="Training"
    ):
        # Initialise the environment
        obs, _ = envManager.env.reset()

        state = torch.tensor(obs, dtype=torch.float32, device=g.DEVICE).unsqueeze(0)

        episode_reward = 0
        # Start counting upwards. Every iteration is a step in the environment
        for t in count():
            # Select and perform an action
            action = envManager.select_action(state, network)
            # Take the action and get the next state and reward
            observation, reward, terminated, truncated, info = envManager.env.step(
                action.item()
            )
            episode_reward += reward

            reward = torch.tensor([reward], device=g.DEVICE)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=g.DEVICE
                ).unsqueeze(0)

            # Store the trastition in memory
            memory.push(state, action, next_state, reward)

            state = next_state

            # Learn from our mistakes
            optimize(network, memory)

            if terminated or truncated:
                episodes_rewards = np.append(episodes_rewards, episode_reward)
                # Update progress metrics
                table["Episode"] = episode
                table["Duration (steps)"] = t + 1
                table.update(
                    "Reward", episode_reward, aggregate="mean", color="blue bold"
                )

                # Save periodic checkpoint
                if (episode + 1) % checkpoint_freq == 0:
                    checkpoint_rewards[episode] = episode_reward
                    checkpoint = {
                        "episode": episode,
                        "model_state_dict": network.state_dict(),
                        "episodes_rewards": episodes_rewards,
                        "checkpoint_rewards": checkpoint_rewards,  # Save all checkpoint rewards
                        "current_reward": episode_reward,  # Save current episode reward
                    }
                    torch.save(
                        checkpoint, checkpoint_path / f"checkpoint_episode_{episode}.pt"
                    )

                # Save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    checkpoint = {
                        "episode": episode,
                        "model_state_dict": network.state_dict(),
                        "episodes_rewards": episodes_rewards,
                        "best_reward": best_reward,
                        "checkpoint_rewards": checkpoint_rewards,  # Include checkpoint rewards history
                    }
                    torch.save(checkpoint, checkpoint_path / "best_model.pt")

                table.next_row()
                break

    # Save latest checkpoint
    checkpoint_rewards[num_episodes - 1] = episode_reward
    checkpoint = {
        "episode": num_episodes - 1,
        "model_state_dict": network.state_dict(),
        "episodes_rewards": episodes_rewards,
        "checkpoint_rewards": checkpoint_rewards,  # Include complete checkpoint rewards history
        "latest_reward": episode_reward,  # Save latest episode reward
    }
    torch.save(checkpoint, checkpoint_path / "latest_model.pt")

    table.close()
    return episodes_rewards
