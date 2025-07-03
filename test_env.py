import gymnasium as gym
from stable_baselines3 import PPO

# Create the Ant environment from Gymnasium with MuJoCo backend
env = gym.make("Ant-v4", render_mode="human")  # Use "human" to open GUI window

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model (adjust timesteps as needed)
model.learn(total_timesteps=100000)

# Test the trained model
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Test episode total reward: {total_reward}")

env.close()
