from itertools import count
from pathlib import Path

import numpy as np
import torch

import globals as g
import plotter
from env_manager import EnvManager
from optimizer import optimize
from replay_memory import ReplayMemory


class Trainer:
    def __init__(
        self,
        env_manager: EnvManager,
        network: torch.nn.Module,
        memory: ReplayMemory,
        output_dir: str = "output",
    ):
        self.env_manager = env_manager
        self.network = network
        self.memory = memory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir = Path(self.output_dir / "checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.plots_dir = Path(self.output_dir / "plots")
        self.plots_dir.mkdir(exist_ok=True, parents=True)

        # Training state
        self.episodes_rewards = np.array([])
        self.episodes_loss = np.array([])
        self.episodes_max_q = np.array([])
        self.starting_episode = 1
        self.total_frames = 0
        self.best_reward = float("-inf")

    def load_checkpoint(self, checkpoint_type: str):
        """Load a checkpoint of specified type (best/latest)"""
        if checkpoint_type not in ["best", "latest"]:
            return

        checkpoint_file = (
            "best_model.pt" if checkpoint_type == "best" else "latest_model.pt"
        )
        checkpoint_path = self.checkpoint_dir / checkpoint_file

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.network.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}", flush=True)
            print(f"Last Episode: {checkpoint['episode']}", flush=True)

            self.episodes_loss = np.array(checkpoint["episodes_loss"])
            self.episodes_max_q = np.array(checkpoint["episodes_max_q"])
            self.episodes_rewards = np.array(checkpoint["episodes_rewards"])
            self.starting_episode = checkpoint["episode"] + 1
            self.total_frames = checkpoint.get("total_frames", 0)

    def save_checkpoint(
        self,
        episode: int,
        filename: str,
        extra_data: dict = None,
    ):
        """Save a checkpoint with current training state"""
        # Move model to CPU for saving to avoid GPU memory issues
        if next(self.network.parameters()).is_cuda:
            state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        else:
            state_dict = self.network.state_dict()

        checkpoint = {
            "episode": episode,
            "model_state_dict": state_dict,
            "episodes_rewards": self.episodes_rewards,
            "total_frames": self.total_frames,
            "episodes_loss": self.episodes_loss,
            "episodes_max_q": self.episodes_max_q,
        }
        if extra_data:
            checkpoint.update(extra_data)

        # save checkpoint
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Saved checkpoint to {self.checkpoint_dir / filename}", flush=True)

    def plot_data(self):
        data = {
            "rewards": self.episodes_rewards,
            "q_values": self.episodes_max_q,
            "losses": self.episodes_loss,
        }
        plotter.plot_data(
            x=np.arange(len(self.episodes_rewards)),
            data=data,
            config=plotter.PlotConfig(
                title="Training Metrics",
                xlabel="Episode",
                ylabel="Value",
                running_avg=True,
                window_size=100,
                filepath=f"{self.plots_dir}/metrics_{len(self.episodes_rewards)}.png",
            ),
        )

    def train_episode(self):
        """Run a single training episode and return statistics"""
        obs, _ = self.env_manager.env.reset()
        state = torch.tensor(obs, dtype=torch.float32, device=g.DEVICE).unsqueeze(0)

        avg_reward = 0
        avg_loss = 0
        avg_max_q = 0

        for t in count():
            # Select and perform action
            with torch.no_grad():
                action = self.env_manager.select_action(state, self.network)
            observation, reward, terminated, truncated, _ = self.env_manager.env.step(
                action.item()
            )
            avg_reward += reward

            reward = torch.tensor([reward], device=g.DEVICE)
            next_state = (
                None
                if terminated
                else torch.tensor(
                    observation, dtype=torch.float32, device=g.DEVICE
                ).unsqueeze(0)
            )

            # Store transition and optimize
            self.memory.push(state, action, next_state, reward)
            state = next_state

            if len(self.memory) >= g.BATCH_SIZE:
                loss, max_q = optimize(self.network, self.memory)
                if loss is not None and max_q is not None:
                    avg_loss += loss
                    avg_max_q += max_q

            if terminated or truncated:
                break

        torch.cuda.memory_summary()
        return avg_reward / (t + 1), avg_loss / (t + 1), avg_max_q / (t + 1), t + 1

    def train(
        self,
        num_episodes: int,
        checkpoint_freq: int = 100,
        load_checkpoint_type: str = "none",
        max_frames: int = None,
    ):
        """Main training loop"""
        # Load checkpoint if specified
        self.load_checkpoint(load_checkpoint_type)

        for episode in range(
            self.starting_episode, self.starting_episode + num_episodes
        ):
            # Check if we've exceeded max frames
            if max_frames is not None and self.total_frames >= max_frames:
                print(
                    f"\nStopping training - reached {self.total_frames} frames (max: {max_frames})",
                    flush=True,
                )
                break

            # Run episode
            avg_reward, avg_loss, avg_max_q, steps = self.train_episode()
            self.total_frames += steps

            # Update statistics
            self.episodes_rewards = np.append(self.episodes_rewards, avg_reward)
            self.episodes_loss = np.append(self.episodes_loss, avg_loss)
            self.episodes_max_q = np.append(self.episodes_max_q, avg_max_q)

            print(
                f"Episode: {episode}, Duration: {steps}, Frames: {self.total_frames}, "
                f"Loss: {avg_loss:.4f}, Max Q: {avg_max_q:.4f}, Reward: {avg_reward:.4f}",
                flush=True,
            )

            # Save periodic checkpoint
            if (episode + 1) % checkpoint_freq == 0:
                self.save_checkpoint(
                    episode,
                    f"checkpoint_episode_{episode}.pt",
                    {"current_reward": avg_reward},
                )

                self.plot_data()

            # Save best model
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.save_checkpoint(
                    episode,
                    "best_model.pt",
                    {"best_reward": self.best_reward},
                )

        # Save final checkpoint
        self.save_checkpoint(
            len(self.episodes_rewards),
            "latest_model.pt",
            {"latest_reward": avg_reward},
        )

        self.plot_data()

        # Print summary
        print(f"Training complete. Total frames: {self.total_frames}", flush=True)
        print(f"Best reward: {self.best_reward}", flush=True)
        print(f"Average reward: {np.mean(self.episodes_rewards)}", flush=True)
        print(f"Total episodes: {len(self.episodes_rewards)}", flush=True)
        return self.episodes_rewards
