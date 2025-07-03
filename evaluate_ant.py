from stable_baselines3 import PPO
from handstand_rl.envs.ant_stand_env import AntStandEnv
import time

# Load Trained Model
model = PPO.load("ppo_ant_stand")

# Create Env with GUI
env = AntStandEnv(render_mode="human")
obs, _ = env.reset()

# Evaluate the Agent
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"[STEP {step}] Reward: {reward:.2f} | Terminated: {terminated} | Truncated: {truncated}")

    time.sleep(0.02)  # optional: slow down to view behavior clearly

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
