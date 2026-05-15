from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from handstand_rl.envs.ant_stand_env import AntStandEnv

import os
import csv

# CSV Logger Setup 
log_path = os.path.join(".", "ant_performance_log.csv")
log_file = open(log_path, mode="w+", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["Step", "Reward", "TorsoHeight", "Terminated", "Truncated"])

# === Vectorized Multi-Ant Training Environment ===
def make_env():
    return Monitor(AntStandEnv(render_mode="Human"), filename=None)

env = DummyVecEnv([make_env for _ in range(10)])  # Training with 10 Ants

# === PPO Agent ===
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_logs_tb/"
)

# === Train the PPO Model ===
model.learn(total_timesteps=1000000)

# === Save Trained Agent ===
model.save("ppo_ant_stand")

# === Evaluation (Only One Ant Rendered) ===
eval_env = AntStandEnv(render_mode="human")
obs, _ = eval_env.reset()

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

    torso_height = obs[0]  # single Ant, so obs is 1D
    log_writer.writerow([step, float(reward), torso_height, bool(terminated), bool(truncated)])

    eval_env.render()

    if terminated or truncated:
        obs, _ = eval_env.reset()

log_file.close()
eval_env.close()
