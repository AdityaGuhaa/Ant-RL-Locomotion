import gymnasium as gym
import numpy as np

class AntStandEnv(gym.Env):
    def __init__(self, render_mode=None, height_threshold=0.5, bonus=5.0):
        super().__init__()
        self.env = gym.make("Ant-v5", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.height_threshold = height_threshold
        self.bonus = bonus

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        torso_height = obs[0]
        upright_bonus = float(torso_height > self.height_threshold)*self.bonus
        fall_penalty = float(torso_height<0.30)*-10.0

        reward += upright_bonus + fall_penalty
        
        print(f"[STEP] Height: {torso_height:.3f} | Bonus: {upright_bonus:.2f} | Penalty: {fall_penalty:.2f} | Total Reward: {reward:.2f}")

        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
