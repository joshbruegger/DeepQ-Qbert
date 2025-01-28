import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        self.layer1 = nn.Conv2d(4, 16, 8, 4)
        self.layer2 = nn.Conv2d()