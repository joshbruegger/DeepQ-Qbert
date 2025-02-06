import math
import random
from time import time

import ale_py
import gymnasium as gym
import torch
from gymnasium.wrappers import (
    FrameStackObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
)
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
)

import globals as g

gym.register_envs(ale_py)  # Tell the IDE that ale_py is used


class EnvManager:
    def __init__(self, env_name: str):
        self.steps = 0

        self.clean_env_name = env_name.replace("/", "-")

        # Make the environment
        self.env = gym.make(
            env_name, render_mode="rgb_array", obs_type="grayscale", frameskip=4
        )
        self.env = RecordEpisodeStatistics(self.env)

        self.env = EpisodicLifeEnv(self.env)

        if "FIRE" in self.env.unwrapped.get_action_meanings():
            self.env = FireResetEnv(self.env)

        self.env = ClipRewardEnv(self.env)

        self.env = ResizeObservation(self.env, (110, 84))
        self.env = FrameStackObservation(self.env, g.MEMORY_SIZE)

    def setup_recording(self, frequency: int = 2):
        self.env = gym.wrappers.RecordVideo(
            self.env,
            episode_trigger=lambda num: (num) % frequency == 0,
            video_folder=f"videos/{self.clean_env_name}",
            name_prefix=f"episode{time()}-",
        )

    def select_action(self, state, network):
        sample = random.random()
        eps_threshold = g.EPS_END + (g.EPS_START - g.EPS_END) * math.exp(
            -1 * self.steps / g.EPS_DECAY
        )
        self.steps += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return network(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.env.action_space.sample()]],
                device=g.DEVICE,
                dtype=torch.long,
            )
