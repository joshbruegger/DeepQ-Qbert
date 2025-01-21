import gymnasium as gym
import ale_py

def main():
    gym.register_envs(ale_py)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger=lambda num: num % 2 == 0,
        video_folder="videos",
        name_prefix="video-",
    )
    env.reset()
    for _ in range(100):

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
