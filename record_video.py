import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from handstand_rl.envs.ant_stand_env import AntStandEnv
from stable_baselines3 import PPO

# === Load the trained model ===
model = PPO.load("ppo_ant_stand")

env = AntStandEnv(render_mode="rgb_array")

obs, _ = env.reset()

frame = env.render()
print(type(frame), frame.shape)

env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)


obs, _ = env.reset()
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
print("✅ Video saved in 'videos/' folder")
