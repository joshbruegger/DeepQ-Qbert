import os

import torch

# Set memory management environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

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

# Configure PyTorch to be more memory efficient
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Set memory allocator settings
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available memory

print(f"Using device: {DEVICE}", flush=True)
