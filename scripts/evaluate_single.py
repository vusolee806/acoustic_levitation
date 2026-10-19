import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from HemisAcoustic.envs.train_env_single import TrainEnvSingle

def main():
    print("=== 1. TẢI CẤU HÌNH & MÔI TRƯỜNG ===")
    config_path = os.path.join(os.path.dirname(__file__), "../configs/train_single.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Bọc môi trường gốc vào DummyVecEnv
    env = DummyVecEnv([lambda: TrainEnvSingle(config)])
    
    print("=== 2. TẢI MÔ HÌNH VÀ BỘ CHUẨN HÓA ===")
    # ĐIỀN ĐÚNG VERSION (Ví dụ: _v4)
    model_version_path = config['train']['save_path'] + "_v7" 
    
    # Thêm device="cpu" để tắt cảnh báo màu vàng của GPU
    model = PPO.load(model_version_path + ".zip", device="cpu")
    
    # Tải lại bộ chuẩn hóa (Kính râm của AI)
    env = VecNormalize.load(model_version_path + "_vecnormalize.pkl", env)
    
    # Tắt chế độ train và không chuẩn hóa reward lúc test
    env.training = False
    env.norm_reward = False
    
    print("=== 3. CHẠY MÔ PHỎNG ===")
    obs = env.reset() # VecEnv reset chỉ trả về obs, không có info
    
    # -------------------------------------------------------------
    # BƯỚC QUAN TRỌNG: Xuyên qua lớp vỏ VecNormalize và DummyVecEnv 
    # để lấy môi trường TrainEnvSingle gốc
    base_env = env.venv.envs[0]
    # -------------------------------------------------------------
    
    trajectory = []
    
    # Lấy tọa độ từ base_env
    target_pos = [base_env.target.x, base_env.target.y, base_env.target.z]
    trajectory.append([base_env.particle.x, base_env.particle.y, base_env.particle.z])
    
    step = 0
    
    while True:
        # RL đưa ra quyết định
        action, _states = model.predict(obs, deterministic=True)
        
        # Môi trường thực thi. VecEnv trả về 4 biến (done gộp chung cả terminated và truncated)
        obs, reward, done, info = env.step(action)
        if done[0]:
            trajectory.append(info[0]["particle_pos"])
            break
        else:
            trajectory.append([base_env.particle.x, base_env.particle.y, base_env.particle.z])
        # Ghi nhận tọa độ mới
        trajectory.append([base_env.particle.x, base_env.particle.y, base_env.particle.z])
        step += 1
        
        # Nếu VecEnv báo done[0] == True thì dừng
        if done[0]:
            break
            
    trajectory = np.array(trajectory)
    
    # Kiểm tra xem điểm cuối cùng của quỹ đạo có nằm trong bán kính đích không
    if np.linalg.norm(trajectory[-1] - target_pos) <= config['env']['target_radius']:
        print(f"✅ THÀNH CÔNG! Hạt đã đến đích sau {step} bước.")
    else:
        print(f"❌ THẤT BẠI! Hạt đã rơi khỏi bán cầu hoặc hết thời gian ({step} bước).")

    print("=== 4. VẼ QUỸ ĐẠO 3D ===")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ không gian an toàn (Bát bán cầu giới hạn)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 50)
    x_sphere = config['env']['radius'] * np.outer(np.cos(u), np.sin(v))
    y_sphere = config['env']['radius'] * np.outer(np.sin(u), np.sin(v))
    z_sphere = config['env']['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + config['env']['z_center']
    z_sphere = config['env']['z_center'] - (z_sphere - config['env']['z_center'])
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1)

    # Vẽ đường bay của hạt
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            label='Quỹ đạo hạt (RL)', color='blue', linewidth=2, marker='o', markersize=3)
    
    # Vẽ điểm xuất phát (Màu xanh lá)
    ax.scatter(*trajectory[0], color='green', s=100, label='Điểm xuất phát', zorder=5)
    
    # Vẽ điểm đích (Màu đỏ)
    ax.scatter(*target_pos, color='red', s=100, label='Đích đến (Target)', zorder=5)
    
    ax.set_title("Mô phỏng Quỹ đạo bay của hạt (PPO)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    main()
