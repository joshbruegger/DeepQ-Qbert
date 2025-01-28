import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Conv2d(n_obs, 16, 8, 4)
        self.layer2 = nn.Conv2d(16, 32, 4, 2)
        self.layer3 = nn.Linear(32 * 9 * 9, 256)
        self.layer4 = nn.Linear(256, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)