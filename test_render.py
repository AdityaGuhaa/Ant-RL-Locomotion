import gymnasium as gym

env = gym.make("Ant-v4")  # no render_mode
obs, _ = env.reset()

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step} | Reward: {reward:.2f}")
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
