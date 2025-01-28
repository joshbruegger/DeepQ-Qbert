from collections import deque
from itertools import count
import math
import random
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStackObservation


import ale_py

import torch

from model import DQN
from replay_memory import ReplayMemory

QUEUE_N_FRAMES = 4

BATCH_SIZE = 32
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000000


# def shift_state_queue(state, obs):
#     state = torch.roll(state, shifts=-1, dims=0)
#     state[0] = torch.tensor(obs, dtype=torch.float32, device=device)
#     return state


steps_done = 0
def select_action(state, network):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return network(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.long)

# Pytorch setup
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

gym.register_envs(ale_py)

# Prepare the environment
env = gym.make("ALE/Pong-v5", render_mode="rgb_array", obs_type="grayscale", frameskip=4)
env = ResizeObservation(env, (110, 84))
env = FrameStackObservation(env, QUEUE_N_FRAMES)

# Record what happens
env = gym.wrappers.RecordVideo(
    env,
    episode_trigger=lambda num: num % 2 == 0,
    video_folder="videos",
    name_prefix="video-",
)

# Make the network
n_actions = env.action_space.n
network = DQN(QUEUE_N_FRAMES, n_actions).to(device)

num_episodes = 5

memory = ReplayMemory(10000)

episodes_durations = []
for i_episode in range(num_episodes):
    # Initialise the environment
    obs, info = env.reset()

    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    # iterate indefinetly
    for t in count():
        action = select_action(state, network) # Infer an action
        # Take the action
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the trastition in memory
        memory.push(state, action, next_state, reward)

        state = next_state

        # Learn from our mistakes
        # optimize_model()

        if terminated or truncated:
            episodes_durations.append(t + 1)
            # plot_durations()
            break

print("Complete")

#plot stuff

env.close()


