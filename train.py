from itertools import count
from pathlib import Path

import numpy as np
import torch

# from progress_table import ProgressTable
import globals as g
from env_manager import EnvManager
from optimizer import optimize
from replay_memory import ReplayMemory


def load_checkpoint(network: torch.nn.Module, checkpoint_path: Path):
    """Load model checkpoint and return training state.

    Args:
        network: The network to load weights into
        checkpoint_path: Path to the checkpoint file

    Returns:
        Tuple of (episodes_rewards, last_episode, checkpoint_rewards, total_frames)
    """
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Last Episode: {checkpoint['episode']}")
        return (
            checkpoint["episodes_rewards"],
            checkpoint["episode"],
            checkpoint.get("checkpoint_rewards", {}),
            checkpoint.get(
                "total_frames", 0
            ),  # Return total_frames, default to 0 if not found
        )
    return None, 0, {}, 0  # Return 0 for total_frames if no checkpoint exists


def train(
    envManager: EnvManager,
    network: torch.nn.Module,
    num_episodes: int,
    memory: ReplayMemory,
    checkpoint_dir: str = "checkpoints",
    checkpoint_freq: int = 100,  # Save every 100 episodes
    load_checkpoint_type: str = "none",  # Can be "best", "latest", or "none"
    max_frames: int = None,  # Maximum number of frames to train for
):
    # Create checkpoint directory if it doesn't exist
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    # Load checkpoint if specified
    episodes_rewards = np.array([])
    starting_episode = 0
    checkpoint_rewards = {}
    total_frames = 0  # Initialize total frame counter

    if load_checkpoint_type in ["best", "latest"]:
        checkpoint_file = (
            "best_model.pt" if load_checkpoint_type == "best" else "latest_model.pt"
        )
        loaded_rewards, last_episode, loaded_checkpoint_rewards, loaded_total_frames = (
            load_checkpoint(network, checkpoint_path / checkpoint_file)
        )
        if loaded_rewards is not None:
            episodes_rewards = np.array(loaded_rewards)
            starting_episode = last_episode + 1
            checkpoint_rewards = loaded_checkpoint_rewards
            total_frames = loaded_total_frames  # Restore total frames from checkpoint

    best_reward = float("-inf")

    # # Initialize progress table
    # table = ProgressTable(
    #     pbar_style="circle",
    #     pbar_embedded=False,
    # )

    for episode in range(starting_episode, starting_episode + num_episodes):
        # Check if we've exceeded max frames
        if max_frames is not None and total_frames >= max_frames:
            print(
                f"\nStopping training - reached {total_frames} frames (max: {max_frames})"
            )
            break

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
                total_frames += t + 1  # Update total frames
                # Update progress metrics
                print(
                    f"Episode: {episode}, Duration: {t + 1}, Total Frames: {total_frames}"
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
                        "total_frames": total_frames,  # Save total frames
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
                        "total_frames": total_frames,  # Save total frames
                    }
                    torch.save(checkpoint, checkpoint_path / "best_model.pt")

                # table.next_row()
                break

    # Save latest checkpoint
    checkpoint_rewards[num_episodes - 1] = episode_reward
    checkpoint = {
        "episode": num_episodes - 1,
        "model_state_dict": network.state_dict(),
        "episodes_rewards": episodes_rewards,
        "checkpoint_rewards": checkpoint_rewards,  # Include complete checkpoint rewards history
        "latest_reward": episode_reward,  # Save latest episode reward
        "total_frames": total_frames,  # Save total frames
    }
    torch.save(checkpoint, checkpoint_path / "latest_model.pt")

    # table.close()
    # print summary
    print(f"Training complete. Total frames: {total_frames}")
    print(f"Best reward: {best_reward}")
    print(f"Average reward: {np.mean(episodes_rewards)}")
    print(f"Total episodes: {len(episodes_rewards)}")
    return episodes_rewards
