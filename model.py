import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Conv2d(
            n_obs, 16, kernel_size=8, stride=4
        )  # Input: 110x84 -> Output: 26x20
        self.layer2 = nn.Conv2d(
            16, 32, kernel_size=4, stride=2
        )  # Input: 26x20 -> Output: 12x9

        # Calculate the size of flattened features
        self.flatten_size = 32 * 12 * 9

        self.layer3 = nn.Linear(self.flatten_size, 256)
        self.layer4 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.layer3(x))
        return self.layer4(x)
