import argparse

# import globals as g
import train_refactor

# from env_manager import EnvManager
# from model import DQN
# from replay_memory import ReplayMemory
# from train import Trainer

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train DeepQbert")
parser.add_argument(
    "--load-checkpoint",
    choices=["best", "latest", "none"],
    default="none",
    help="Load from checkpoint: best model, latest model, or none",
)
parser.add_argument(
    "--num-episodes",
    type=int,
    default=5,
    help="Number of episodes to train for",
)
parser.add_argument(
    "--checkpoint-freq",
    type=int,
    default=2,
    help="Save checkpoint every N episodes",
)
parser.add_argument(
    "--max-frames",
    type=int,
    default=None,
    help="Maximum number of frames to train for (default: no limit)",
)
parser.add_argument(
    "--env-name",
    type=str,
    default="ALE/Pong-v5",
    help="Environment name. Can be: ALE/Pong-v5, ALE/BeamRider-v5, ALE/Qbert-v5, ALE/Breakout-v5, ALE/Seaquest-v5, ALE/Pong-v5",
)
parser.add_argument(
    "--no-recording",
    action="store_true",
    help="Disable recording",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="output",
    help="Output directory",
)
args = parser.parse_args()

# # clean up the env name for the checkpoint directory
# env_name = args.env_name.replace("/", "-")

# envManager = EnvManager(args.env_name)

# Make the network
# network = DQN(g.MEMORY_SIZE, envManager.env.action_space.n).to(g.DEVICE)

train_refactor.train(
    num_frames=1000000,
    env_name=args.env_name,
    num_envs=1,
    record=not args.no_recording,
    checkpoint_type=args.load_checkpoint,
    log_interval=1,
    save_interval=10,
    warmup_frames=10,
    max_frames=args.max_frames,
    batch_size=32,
    lr=1e-4,
    gamma=0.99,
    output_dir=args.output_dir,
    eps_start=1,
    eps_end=0.1,
    eps_decay=1000000,
    memory_size=1000000,
)


# if not args.no_recording:
#     envManager.setup_recording(args.checkpoint_freq)

# # Make the memory
# memory = ReplayMemory(1000000)

# # Create trainer instance
# trainer = Trainer(
#     env_manager=envManager,
#     network=network,
#     memory=memory,
#     output_dir=args.output_dir,
# )

# # Train the model
# episodes_rewards = trainer.train(
#     num_episodes=args.num_episodes,
#     checkpoint_freq=args.checkpoint_freq,
#     load_checkpoint_type=args.load_checkpoint,
#     max_frames=args.max_frames,
# )

# # Close the environment
# envManager.env.close()
