import os
import random
import time

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer

from envs import make_envs
from logger import Logger
from model import DQN


def epsilon(t, eps_start, eps_end, eps_decay):
    slope = (eps_end - eps_start) / eps_decay
    return max(slope * t + eps_start, eps_end)


def train(
    num_frames: int,
    env_name: str,
    num_envs: int,
    record: bool,
    checkpoint_type: str,
    log_interval: int,
    save_interval: int,
    warmup_frames: int,
    max_frames: int,
    batch_size: int,
    lr: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay: int,
    memory_size: int,
    output_dir: str,
):
    print(f"Training on {env_name} with {num_envs} environments", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = time.strftime("%Y-%m-%d_%H:%M:%S")
    envs = make_envs(env_name, num_envs, record, run_name, f"{output_dir}/videos")
    network = DQN(4, envs.single_action_space.n).to(device)

    optimizer = torch.optim.RMSprop(network.parameters(), lr=lr)
    loss_fn = torch.nn.functional.mse_loss

    replay_memory = ReplayBuffer(
        memory_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    start_frame = 0
    start_log_dict = {}
    output_dir = f"{output_dir}/{env_name.replace('/', '_')}"
    plots_dir = f"{output_dir}/plots"
    ckpt_dir = f"{output_dir}/checkpoints"

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load checkpoint if specified
    if checkpoint_type and checkpoint_type in ["best", "latest"]:
        ckpt_path = f"{ckpt_dir}/{checkpoint_type}_model.pth"

        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path)
            network.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {ckpt_path}", flush=True)

            start_frame = checkpoint["frame"] + 1
            start_log_dict = checkpoint["log_dict"]
    else:
        print("No checkpoint specified, starting from scratch", flush=True)

    start_time = time.time()
    logger = Logger(start_log_dict)

    # Get initial observations and start training loop
    print("Starting training loop", flush=True)
    obs, _ = envs.reset()
    for frame in range(start_frame, start_frame + num_frames):
        if max_frames is not None and frame >= max_frames:
            print(f"\nStopping training: reached max frames ({max_frames})", flush=True)
            break

        # Select actions for each env, either random (eps probability) or from network
        eps = epsilon(frame, eps_start, eps_end, eps_decay)
        if random.random() < eps:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = network(torch.Tensor(obs).to(device))  # TODO: No grad?
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Perform actions
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # Check if any envs are done and log data if so
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    # If episode is not done, continue
                    continue
                print(f"Frame {frame}, reward {info['episode']['r']}", flush=True)
                logger.log("reward", frame, info["episode"]["r"])
                logger.log("length", frame, info["episode"]["l"])
                logger.log("epsilon", frame, eps)

        # When an episode terminates, the next observation is the initial observation of the next episode
        # We do not want this, as we want to store the final observation of the episode in the replay memory
        actual_next_obs = next_obs.copy()
        for i, done in enumerate(terminated):
            if done:
                print(infos.keys())
                actual_next_obs[i] = infos["next_observation"][i]

        # Store transition in replay memory
        replay_memory.add(obs, actual_next_obs, actions, rewards, terminated, infos)

        # update observation for next iteration
        obs = next_obs

        # Now we can optimize the network, but not before the warmup frames
        if frame < warmup_frames:
            continue

        # Sample a batch of transitions from the replay memory
        batch = replay_memory.sample(batch_size)

        # Compute the target Q-values
        with torch.no_grad():
            max_estimate, _ = network(batch.next_observations).max(dim=1)
            target_q_values = batch.rewards.flatten() + gamma * max_estimate * (
                1 - batch.dones.flatten()
            )

        # Predict the Q-values
        predicted_q_values = (
            network(batch.observations).gather(1, batch.actions).squeeze()
        )

        # Compute the loss
        loss = loss_fn(target_q_values, predicted_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log and save if necessary

        if frame % log_interval == 0:
            fps = int(frame / (time.time() - start_time))
            log_q_values = predicted_q_values.mean().item()
            logger.log("loss", frame, loss)
            logger.log("q_values", frame, log_q_values)
            logger.log("FPS", frame, fps)
            logger.save_plot(plots_dir, f"frame_{frame}")

            print(f"Frame: {frame}, q_values: {log_q_values}, FPS: {fps}", flush=True)

        if frame % save_interval == 0:
            torch.save(
                {
                    "model_state_dict": network.state_dict(),
                    "frame": frame,
                    "log_dict": logger.dict,
                },
                f"{ckpt_dir}/frame_{frame}.pth",
            )

    print(
        f"Training complete in {time.time() - start_time} seconds ({frame} frames)",
        flush=True,
    )

    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "frame": frame,
            "log_dict": logger.dict,
        },
        f"{ckpt_dir}/latest_model.pth",
    )

    fps = int(frame / (time.time() - start_time))
    logger.log("loss", frame, loss)
    logger.log("q_values", frame, predicted_q_values.mean().item())
    logger.log("FPS", frame, fps)
    logger.save_plot(plots_dir, f"frame_{frame}")

    return logger
