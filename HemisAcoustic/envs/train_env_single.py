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
        
        # Action Space: Cho mạng RL thoải mái xuất giá trị từ -1.0 đến 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation Space: 9 chiều [x, y, z, vx, vy, vz, dx, dy, dz]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.particle = None
        self.target = None
        self.current_step = 0

    def _get_obs(self):
        """Rút trích 9 giá trị State cho mạng Neural, CHUẨN HÓA VỀ MILLIMETER (mm)"""
        dx = self.target.x - self.particle.x
        dy = self.target.y - self.particle.y
        dz = self.target.z - self.particle.z
        
        # Nhân 1000 để chuyển đổi m -> mm
        obs = np.array([
            self.particle.x * 1000.0, self.particle.y * 1000.0, self.particle.z * 1000.0,
            self.particle.vX * 1000.0, self.particle.vY * 1000.0, self.particle.vZ * 1000.0,
            dx * 1000.0, dy * 1000.0, dz * 1000.0
        ], dtype=np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        """Khởi tạo lại vạch xuất phát và đích đến"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # fixed_center_start=True để hạt luôn sinh ra ở tâm (0,0,0)
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
        
        # 1. Scale hành động của AI [-1, 1] ra vận tốc thực tế (VD: 0.1 m/s)
        self.particle.vX = action[0] * self.cfg['max_velocity']
        self.particle.vY = action[1] * self.cfg['max_velocity']
        self.particle.vZ = action[2] * self.cfg['max_velocity']
        
        # 2. Cập nhật Vị trí (Kinematics) với vận tốc thực tế
        dt = self.cfg['delta_time']
        self.particle.x += self.particle.vX * dt
        self.particle.y += self.particle.vY * dt
        self.particle.z += self.particle.vZ * dt
        
        # 3. Tính khoảng cách mới (SAU KHI hạt đã di chuyển xong)
        current_pos = np.array([self.particle.x, self.particle.y, self.particle.z])
        target_pos = np.array([self.target.x, self.target.y, self.target.z])
        current_dist = np.linalg.norm(target_pos - current_pos)

        # 4. Xác định Reward & Trạng thái kết thúc
        terminated = False
        truncated = False
        
        # Thưởng tịnh tiến (Chỉ thưởng khi tiến lại gần)
        progress_reward = (self.particle.last_timestep_dist - current_dist) * 100.0 
        # Phạt thời gian cực nhỏ để ép không được đứng im
        time_penalty = -0.01 
        # Phạt nhẹ nếu dùng vận tốc lớn giúp quỹ đạo mượt hơn
        action_penalty = -0.005 * np.linalg.norm(action)
        reward = progress_reward + time_penalty + action_penalty
            
        # Kiểm tra văng khỏi bát bán cầu (Hình Elip)
        centered_pos = current_pos.copy()
        centered_pos[2] -= self.cfg['z_center'] 
        is_out_of_bounds = (
            (centered_pos[0]**2 / self.cfg['radius']**2) + 
            (centered_pos[1]**2 / self.cfg['radius']**2) + 
            (centered_pos[2]**2 / self.cfg['z_radius']**2)
        ) > 1.0

        if is_out_of_bounds:
            reward -= 50.0  # Phạt cực nặng
            terminated = True
        
        # Thưởng tới đích
        elif current_dist <= self.cfg['target_radius']:
            reward += 100.0 # Thưởng siêu khổng lồ
            terminated = True
            
        # Kiểm tra hết giờ (Max steps)
        if self.current_step >= self.cfg['max_steps']:
            truncated = True
            
        # Lưu lại khoảng cách cho bước sau
        self.particle.last_timestep_dist = current_dist
        info = {
            "particle_pos": current_pos.copy(),
            "is_success": current_dist <= self.cfg['target_radius']
        }
        return self._get_obs(), reward, terminated, truncated, info
        
