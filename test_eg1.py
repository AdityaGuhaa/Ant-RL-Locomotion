import gymnasium as gym

env = gym.make("Ant-v4", render_mode="human")
obs, _= env.reset()

for _ in range(5):
    env.step(env.action_space.sample())

env.close()