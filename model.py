import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, obs_channels, n_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                obs_channels, 16, kernel_size=8, stride=4
            ),  # Input: 110x84 -> Output: 26x20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # Input: 26x20 -> Output: 12x9
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # Input: 12x9 -> Output: 10x8
            nn.ReLU(),
            nn.Linear(32 * 12 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)
