import gymnasium as gym
from gymnasium.wrappers import ResizeObservation


import ale_py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def main():
    # Pytorch setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    gym.register_envs(ale_py)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", obs_type="grayscale")

    # Wrap environment to preprocess observations
    env = ResizeObservation(env, (110, 84))

    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger=lambda num: num % 2 == 0,
        video_folder="videos",
        name_prefix="video-",
    )
    
    env.reset()
    for _ in range(100):

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
