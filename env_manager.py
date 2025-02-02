import math
import random

import ale_py
import gymnasium as gym
import torch
from gymnasium.wrappers import FrameStackObservation, ResizeObservation

import globals as g

gym.register_envs(ale_py)  # Tell the IDE that ale_py is used


class EnvManager:
    def __init__(self):
        self.steps = 0

        # Make the environment
        self.env = gym.make(
            "ALE/Pong-v5", render_mode="rgb_array", obs_type="grayscale", frameskip=4
        )
        self.env = ResizeObservation(self.env, (110, 84))
        self.env = FrameStackObservation(self.env, g.QUEUE_N_FRAMES)

        # Record what happens
        self.env = gym.wrappers.RecordVideo(
            self.env,
            episode_trigger=lambda num: num % 2 == 0,
            video_folder="videos",
            name_prefix="video-",
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
