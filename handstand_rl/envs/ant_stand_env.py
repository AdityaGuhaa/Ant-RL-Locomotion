import gymnasium as gym
import numpy as np

# Layout of Ant-v4 observation (with exclude_current_positions_from_observation=True
# and use_contact_forces=True):
#   qpos[2:]       -> 13 dims  (indices 0..12)
#   qvel           -> 14 dims  (indices 13..26)
#   cfrc_ext       -> 14 * 6 = 84 dims (indices 27..110)
#
# The saved ppo_ant_stand model was trained against the older 105-dim layout
# that excluded the world body's contact forces (13 * 6 = 78 instead of 84),
# i.e. obs[27:33] was not present. The world body's cfrc_ext is always zero,
# so we drop those 6 entries to keep the saved policy compatible.
_LEGACY_OBS_SIZE = 105
_WORLD_CFRC_SLICE = slice(27, 33)


class AntStandEnv(gym.Env):
    def __init__(self, render_mode=None, height_threshold=0.5, bonus=5.0):
        super().__init__()
        self.render_mode = render_mode
        # use_contact_forces=True restores the cfrc_ext block in the obs so we
        # can reshape it to the legacy 105-dim layout below.
        self.env = gym.make("Ant-v4", render_mode=render_mode, use_contact_forces=True)
        self.action_space = self.env.action_space

        inner_obs_space = self.env.observation_space
        if inner_obs_space.shape == (_LEGACY_OBS_SIZE,):
            # Already the legacy size, nothing to slice.
            self._needs_legacy_slice = False
            self.observation_space = inner_obs_space
        else:
            self._needs_legacy_slice = True
            low = self._to_legacy(inner_obs_space.low)
            high = self._to_legacy(inner_obs_space.high)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=inner_obs_space.dtype)

        self.height_threshold = height_threshold
        self.bonus = bonus

    @staticmethod
    def _to_legacy(obs):
        """Drop the world body's cfrc_ext (6 dims) to match the legacy 105-dim layout."""
        return np.concatenate([obs[: _WORLD_CFRC_SLICE.start], obs[_WORLD_CFRC_SLICE.stop :]])

    def _maybe_slice(self, obs):
        return self._to_legacy(obs) if self._needs_legacy_slice else obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._maybe_slice(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Upright Bonus based on torso height
        torso_height = obs[0]
        upright_bonus = max(0, torso_height - self.height_threshold) * self.bonus

        # 2. Fall penalty
        fall_penalty = float(torso_height < 0.30) * -10.0

        # 3. Forward velocity reward (x-direction)
        x_velocity = self.env.unwrapped.data.qvel[0]
        forward_reward = x_velocity * 1.0  # adjust coefficient as needed

        # 4. Action penalty to avoid jittery movement
        action_penalty = -0.001 * np.square(action).sum()

        # Total reward
        reward += upright_bonus + fall_penalty + forward_reward + action_penalty

        return self._maybe_slice(obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
