import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from HemisAcoustic.envs.train_env_single import TrainEnvSingle

def get_auto_version_path(base_path):
    """Hàm tự động đếm số v1, v2, v3... nếu file đã tồn tại"""
    version = 1
    # Kiểm tra xem file ppo_single_particle_v1.zip, v2.zip... đã có chưa
    while os.path.exists(f"{base_path}_v{version}.zip"):
        version += 1
    return f"{base_path}_v{version}"

def main():
    print("=== 1. TẢI CẤU HÌNH ===")
    config_path = os.path.join(os.path.dirname(__file__), "../configs/train_single.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    print("=== 2. KHỞI TẠO MÔI TRƯỜNG SONG SONG ===")
    env = make_vec_env(lambda: TrainEnvSingle(config), n_envs=16)
    
    print("=== 3. BẮT ĐẦU HUẤN LUYỆN (PPO) ===")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/tensorboard/")
    
    timesteps = config['train']['total_timesteps']
    model.learn(total_timesteps=timesteps)
    
    print("=== 4. LƯU MÔ HÌNH TỰ ĐỘNG ===")
    # Lấy đường dẫn gốc từ file YAML
    base_save_path = config['train']['save_path']
    os.makedirs(os.path.dirname(base_save_path), exist_ok=True)
    
    # Gọi hàm tự động sinh version mới nhất
    final_save_path = get_auto_version_path(base_save_path)
    
    model.save(final_save_path)
    print(f"✅ Mô hình đã được lưu tự động tại: {final_save_path}.zip")

if __name__ == "__main__":
    main()
