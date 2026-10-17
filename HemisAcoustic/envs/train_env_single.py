# HemisAcoustic/envs/train_env_single.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Gọi các component lõi của dự án
from HemisAcoustic.particles import particle_slim as Particle
from HemisAcoustic.particles import target_slim as Target
from HemisAcoustic.utils.create_points import create_points_hemisphere

class TrainEnvSingle(gym.Env):
    """Môi trường RL cho 1 hạt trong không gian bán cầu Hemisphere"""
    
    def __init__(self, config):
        super().__init__()
        self.cfg = config['env']
        
        # Action Space: Vận tốc [vX, vY, vZ]
        v_max = self.cfg['max_velocity']
        self.action_space = spaces.Box(low=-v_max, high=v_max, shape=(3,), dtype=np.float32)
        
        # Observation Space: 9 chiều [x, y, z, vx, vy, vz, dx, dy, dz]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.particle = None
        self.target = None
        self.current_step = 0

    def _get_obs(self):
        """Rút trích 9 giá trị State cho mạng Neural"""
        dx = self.target.x - self.particle.x
        dy = self.target.y - self.particle.y
        dz = self.target.z - self.particle.z
        
        obs = np.array([
            self.particle.x, self.particle.y, self.particle.z,
            self.particle.vX, self.particle.vY, self.particle.vZ,
            dx, dy, dz
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """Khởi tạo lại vạch xuất phát và đích đến"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # BỔ SUNG: fixed_center_start=True để hạt luôn sinh ra ở tâm (0,0,0)
        pts = create_points_hemisphere(
            N=2, 
            radius=self.cfg['radius'], 
            z_radius=self.cfg['z_radius'],
            z_center=self.cfg['z_center'],
            fixed_center_start=True
        )
        
        self.particle = Particle(x=pts[0][0], y=pts[0][1], z=pts[0][2]) # Luôn là tâm
        self.target = Target(x=pts[1][0], y=pts[1][1], z=pts[1][2])     # Đích ngẫu nhiên
        
        # Tính khoảng cách ban đầu
        dist = np.linalg.norm(np.array([self.particle.x, self.particle.y, self.particle.z]) - 
                              np.array([self.target.x, self.target.y, self.target.z]))
        self.particle.last_timestep_dist = dist
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Áp dụng Action làm Vận tốc
        self.particle.vX, self.particle.vY, self.particle.vZ = action[0], action[1], action[2]
        
        # 2. Cập nhật Vị trí (Kinematics)
        dt = self.cfg['delta_time']
        self.particle.x += self.particle.vX * dt
        self.particle.y += self.particle.vY * dt
        self.particle.z += self.particle.vZ * dt
        
        # 3. Tính khoảng cách mới
        current_pos = np.array([self.particle.x, self.particle.y, self.particle.z])
        target_pos = np.array([self.target.x, self.target.y, self.target.z])
        current_dist = np.linalg.norm(target_pos - current_pos)
        
        # 4. Xác định Reward & Trạng thái kết thúc
        reward = 0.0
        terminated = False
        truncated = False
        
        # Thưởng tịnh tiến: Hạt bay gần đích hơn so với bước trước thì được điểm
        reward += (self.particle.last_timestep_dist - current_dist) * 100.0
        
        # Điều kiện 1: Tới đích thành công
        if current_dist <= self.cfg['target_radius']:
            reward += 20.0  # Thưởng lớn
            terminated = True
            
        # Điều kiện 2: Bay văng ra khỏi giới hạn bát bán cầu (Rách bẫy)
        centered_pos = current_pos.copy()
        centered_pos[2] -= self.cfg['z_center'] # Đưa tâm về 0 để tính bán kính
        if np.linalg.norm(centered_pos) > self.cfg['radius']:
            reward -= 10.0  # Phạt nặng
            terminated = True
            
        # Điều kiện 3: Hết thời gian
        if self.current_step >= self.cfg['max_steps']:
            truncated = True
            
        # Lưu lại trạng thái cho bước sau
        self.particle.last_timestep_dist = current_dist
        
        return self._get_obs(), reward, terminated, truncated, {}
