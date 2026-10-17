import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO

# Import môi trường của bạn
from HemisAcoustic.envs.train_env_single import TrainEnvSingle

def main():
    print("=== 1. TẢI CẤU HÌNH & MÔI TRƯỜNG ===")
    config_path = os.path.join(os.path.dirname(__file__), "../configs/train_single.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    env = TrainEnvSingle(config)
    
    print("=== 2. TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN ===")
    model_path = config['train']['save_path'] + ".zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại {model_path}. Bạn đã train xong chưa?")
        
    # Tải lại bộ não
    model = PPO.load(model_path)
    
    print("=== 3. CHẠY MÔ PHỎNG ===")
    obs, info = env.reset()
    
    # Lưu trữ tọa độ để vẽ hình
    trajectory = []
    target_pos = [env.target.x, env.target.y, env.target.z]
    
    # Lưu tọa độ ban đầu
    trajectory.append([env.particle.x, env.particle.y, env.particle.z])
    
    terminated = False
    truncated = False
    step = 0
    
    while not (terminated or truncated):
        # Yêu cầu RL đưa ra quyết định (deterministic=True để model lấy hành động tốt nhất, không ngẫu nhiên)
        action, _states = model.predict(obs, deterministic=True)
        
        # Môi trường thực thi hành động
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Ghi nhận tọa độ mới
        trajectory.append([env.particle.x, env.particle.y, env.particle.z])
        step += 1
        
    trajectory = np.array(trajectory)
    
    if terminated and np.linalg.norm(trajectory[-1] - target_pos) <= config['env']['target_radius']:
        print(f"✅ THÀNH CÔNG! Hạt đã đến đích sau {step} bước.")
    else:
        print(f"❌ THẤT BẠI! Hạt đã rơi khỏi bán cầu hoặc hết thời gian ({step} bước).")

    print("=== 4. VẼ QUỸ ĐẠO 3D ===")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ không gian an toàn (Bát bán cầu giới hạn)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 50) # Chỉ vẽ nửa dưới/bán cầu
    x_sphere = config['env']['radius'] * np.outer(np.cos(u), np.sin(v))
    y_sphere = config['env']['radius'] * np.outer(np.sin(u), np.sin(v))
    z_sphere = config['env']['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + config['env']['z_center']
    # Lật ngược bán cầu để giống cái bát
    z_sphere = config['env']['z_center'] - (z_sphere - config['env']['z_center'])
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1)

    # Vẽ đường bay của hạt
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            label='Quỹ đạo hạt (RL)', color='blue', linewidth=2, marker='o', markersize=3)
    
    # Vẽ điểm xuất phát (Màu xanh lá)
    ax.scatter(*trajectory[0], color='green', s=100, label='Điểm xuất phát', zorder=5)
    
    # Vẽ điểm đích (Màu đỏ)
    ax.scatter(*target_pos, color='red', s=100, label='Đích đến (Target)', zorder=5)
    
    # Thiết lập góc nhìn và nhãn
    ax.set_title("Mô phỏng Quỹ đạo bay của hạt (PPO)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    main()
