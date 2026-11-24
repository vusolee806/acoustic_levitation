# HemisAcoustic/envs/sac_wrapper.py

import gymnasium as gym
import numpy as np


class SACAcousticWrapper(gym.Wrapper):
    """
    1. Static observation scaling into ~[-1, 1] (100% stationary for off-policy SAC replay buffer).
    2. Rebalanced dense reward so progress + proximity dominate entropy and balance the -50 boundary cliff.
    """
    def __init__(self, env, train_reward_shaping=True):
        super().__init__(env)
        cfg = env.cfg
        r_mm = cfg["radius"] * 1000.0
        zr_mm = cfg["z_radius"] * 1000.0
        vmax_mm = cfg["max_velocity"] * 1000.0

        # Fixed divisors matching TrainEnvSingle._get_obs() units (mm and mm/s)
        self.obs_scale = np.array(
            [r_mm, r_mm, zr_mm, vmax_mm, vmax_mm, vmax_mm, 2.0 * r_mm, 2.0 * r_mm, 2.0 * zr_mm],
            dtype=np.float32,
        )
        self.train_reward_shaping = train_reward_shaping

    def _scale_obs(self, obs):
        return np.clip(obs / self.obs_scale, -2.0, 2.0).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._scale_obs(obs), info

    def step(self, action):
        # Record distance before step to compute clean scaled progress
        prev_dist = self.env.particle.last_timestep_dist
        obs, orig_reward, terminated, truncated, info = self.env.step(action)
        curr_dist = self.env.particle.last_timestep_dist

        if not self.train_reward_shaping:
            return self._scale_obs(obs), orig_reward, terminated, truncated, info

        # 1. Strong dense progress: +1.0 per mm moved closer (10x stronger than original +0.1/mm)
        progress = (prev_dist - curr_dist) * 1000.0
        # 2. Distance penalty: eliminates hovering / forward-back oscillation far from target
        dist_penalty = -0.05 * (curr_dist * 1000.0)
        # 3. Mild action smoothness penalty
        act_penalty = -0.02 * float(np.linalg.norm(action))

        reward = progress + dist_penalty + act_penalty

        # Keep terminal outcomes proportional to dense rewards
        if terminated:
            if info.get("is_success", False):
                reward += 100.0
            else:
                reward -= 30.0  # Out-of-bounds penalty (strong, but no longer 20x total path reward)

        return self._scale_obs(obs), reward, terminated, truncated, info
