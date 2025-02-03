from itertools import count
from pathlib import Path

import numpy as np
import torch

import globals as g
import plotter
from env_manager import EnvManager
from optimizer import optimize
from replay_memory import ReplayMemory


def load_checkpoint(network: torch.nn.Module, checkpoint_path: Path):
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


def save_checkpoint(
    checkpoint_path: Path,
    network: torch.nn.Module,
    episode: int,
    episodes_rewards: np.ndarray,
    checkpoint_rewards: dict,
    total_frames: int,
    filename: str,
    extra_data: dict = None,
):
    # Move model to CPU for saving to avoid GPU memory issues
    if next(network.parameters()).is_cuda:
        state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
    else:
        state_dict = network.state_dict()

    checkpoint = {
        "episode": episode,
        "model_state_dict": state_dict,
        "episodes_rewards": episodes_rewards,
        "checkpoint_rewards": checkpoint_rewards,
        "total_frames": total_frames,
    }
    if extra_data:
        checkpoint.update(extra_data)

    # save checkpoint
    torch.save(checkpoint, checkpoint_path / filename)
    print(f"Saved checkpoint to {checkpoint_path / filename}")


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

    # Initialize training state
    episodes_rewards = np.array([])
    episodes_loss = np.array([])
    episodes_max_q = np.array([])
    starting_episode = 0
    checkpoint_rewards = {}
    total_frames = 0  # Initialize total frame counter

    # Load checkpoint if specified
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
        avg_loss = 0
        avg_max_q = 0
        # Start counting upwards. Every iteration is a step in the environment
        for t in count():
            # Batch process actions
            with torch.no_grad():
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

            # Perform optimization step
            if len(memory) >= g.BATCH_SIZE:
                loss, max_q = optimize(network, memory)
                if loss is not None and max_q is not None:
                    avg_loss += loss
                    avg_max_q += max_q

            if terminated or truncated:
                break

        episodes_rewards = np.append(episodes_rewards, episode_reward)
        avg_loss /= t + 1
        avg_max_q /= t + 1
        episodes_loss = np.append(episodes_loss, avg_loss)
        episodes_max_q = np.append(episodes_max_q, avg_max_q)
        total_frames += t + 1

        print(
            f"Episode: {episode}, Duration: {t + 1}, Frames: {total_frames}, Loss: {avg_loss:.4f}, Max Q: {avg_max_q:.4f}, Reward: {episode_reward:.4f}"
        )

        # Save periodic checkpoint
        if (episode + 1) % checkpoint_freq == 0:
            checkpoint_rewards[episode] = episode_reward
            save_checkpoint(
                checkpoint_path,
                network,
                episode,
                episodes_rewards,
                checkpoint_rewards,
                total_frames,
                f"checkpoint_episode_{episode}.pt",
                {"current_reward": episode_reward},
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
                    filepath=f"plots/{envManager.clean_env_name}/rewards_{len(episodes_rewards)}.png",
                ),
            )

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_checkpoint(
                checkpoint_path,
                network,
                episode,
                episodes_rewards,
                checkpoint_rewards,
                total_frames,
                "best_model.pt",
                {"best_reward": best_reward},
            )

    # Save latest checkpoint
    checkpoint_rewards[num_episodes - 1] = episode_reward
    save_checkpoint(
        checkpoint_path,
        network,
        num_episodes - 1,
        episodes_rewards,
        checkpoint_rewards,
        total_frames,
        "latest_model.pt",
        {"latest_reward": episode_reward},
    )

    # print summary
    print(f"Training complete. Total frames: {total_frames}")
    print(f"Best reward: {best_reward}")
    print(f"Average reward: {np.mean(episodes_rewards)}")
    print(f"Total episodes: {len(episodes_rewards)}")
    return episodes_rewards
