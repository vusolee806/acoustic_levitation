import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class CustomCartPoleEnv(gym.Env):
    def __init__(self, xml_path='assets/cartpole.xml'):
        super().__init__()
        
        # 1. Find exactly where envs.py is located on your hard drive (.../src)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Go up one folder level to get the project root (.../mujoco_project)
        project_root = os.path.dirname(current_dir)
        
        # 3. Combine it with the provided relative path
        full_xml_path = os.path.join(project_root, xml_path)
        
        # 4. Load MuJoCo using the guaranteed absolute path
        self.model = mujoco.MjModel.from_xml_path(full_xml_path)
        self.data = mujoco.MjData(self.model)

        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        high = np.array([10.0, 10.0, np.pi, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _get_obs(self):
        cart_pos = self.data.qpos[0]
        pole_angle = self.data.qpos[1]
        cart_vel = self.data.qvel[0]
        pole_vel = self.data.qvel[1]
        return np.array([cart_pos, cart_vel, pole_angle, pole_vel], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = self.np_random.uniform(low=-0.05, high=0.05) 
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[0] = action[0]
        for _ in range(10): 
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        
        reward = 1.0 - (pole_angle ** 2) - 0.1 * (cart_pos ** 2) - 0.01 * (action[0] ** 2)
        terminated = bool(abs(pole_angle) > 0.2 or abs(cart_pos) > 2.0)
        
        return obs, reward, terminated, False, {}
