import ale_py
import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
)
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
)

import globals as g

gym.register_envs(ale_py)  # Tell the IDE that ale_py is used


def make_env(
    env_name: str,
    idx: int,
    record: bool = False,
    run_name: str = None,
    vid_dir: str = None,
):
    run_name = run_name.replace("/", "_")

    def maker_fn():
        # if record, make video of only the first env
        if record and idx == 0:
            env = gym.make(
                env_name, render_mode="rgb_array", obs_type="grayscale", frameskip=4
            )
            env = gym.wrappers.RecordVideo(env, f"{vid_dir}/{run_name}")
        else:
            env = gym.make(env_name, obs_type="grayscale", frameskip=4)

        env = RecordEpisodeStatistics(env)

        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = ClipRewardEnv(env)

        env = ResizeObservation(env, (110, 84))
        env = FrameStackObservation(env, 4)

        return env

    return maker_fn


def make_envs(
    env_name: str,
    num_envs: int,
    record: bool = False,
    run_name: str = None,
    vid_dir: str = None,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, i, record, run_name, vid_dir) for i in range(num_envs)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "Only discrete action spaces are supported"

    return envs
