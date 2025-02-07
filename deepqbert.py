import argparse

import train

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train DeepQbert")
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume training from latest checkpoint",
)
parser.add_argument(
    "--frames",
    type=int,
    default=10000000,
    help="Number of frames to train for",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning rate",
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=1,
    help="Number of environments to train on",
)
parser.add_argument(
    "--memory",
    type=int,
    default=1000000,
    help="the replay memory size",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="the discount factor gamma",
)
parser.add_argument(
    "--batch",
    type=int,
    default=32,
    help="the batch size of sample from the reply memory",
)
parser.add_argument(
    "--e-start",
    type=float,
    default=1,
    help="the starting epsilon",
)
parser.add_argument(
    "--e-end",
    type=float,
    default=0.1,
    help="the ending epsilon",
)
parser.add_argument(
    "--e-decay",
    type=int,
    default=1000000,
    help="the last frame of the epsilon decay",
)
parser.add_argument(
    "--warmup",
    type=int,
    default=1000,
    help="the number of frames to warm up for",
)
parser.add_argument(
    "--save-interval",
    type=int,
    default=10000,
    help="Save checkpoint every N episodes",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=200,
    help="Log interval",
)
parser.add_argument(
    "--max-frames",
    type=int,
    default=None,
    help="Maximum number of frames to train for (default: no limit)",
)
parser.add_argument(
    "--env",
    type=str,
    default="ALE/Pong-v5",
    help="Environment name. Can be: ALE/Pong-v5, ALE/BeamRider-v5, ALE/Qbert-v5, ALE/Breakout-v5, ALE/Seaquest-v5, ALE/Pong-v5",
)
parser.add_argument(
    "--record",
    action="store_true",
    default=False,
    help="Record videos",
)
parser.add_argument(
    "--output",
    type=str,
    default="output",
    help="Output directory",
)
args = parser.parse_args()

train.train(
    num_frames=args.frames,
    env_name=args.env,
    num_envs=args.num_envs,
    record=args.record,
    load_latest_ckpt=args.resume,
    log_interval=args.log_interval,
    save_interval=args.save_interval,
    warmup_frames=args.warmup,
    max_frames=args.max_frames,
    batch_size=args.batch,
    lr=args.lr,
    gamma=args.gamma,
    output_dir=args.output,
    eps_start=args.e_start,
    eps_end=args.e_end,
    eps_decay=args.e_decay,
    memory_size=args.memory,
)
