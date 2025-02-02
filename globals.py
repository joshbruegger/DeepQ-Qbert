import torch

QUEUE_N_FRAMES = 4

BATCH_SIZE = 32
LR = 1e-4  # Learning rate

GAMMA = 0.99  # Discount factor for cumulative reward. Lower values make rewards from uncertain future less important than immediate rewards

EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000000

# Pytorch setup
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
