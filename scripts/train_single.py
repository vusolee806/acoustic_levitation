import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv


import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback


# Import môi trường của bạn
from HemisAcoustic.envs.train_env_single import TrainEnvSingle


def get_auto_version_path(base_path):
    """Hàm tự động đếm số v1, v2, v3... để không ghi đè model cũ"""
    version = 1
    while os.path.exists(f"{base_path}_v{version}.zip"):
        version += 1
    return f"{base_path}_v{version}"

def main():
    print("=== 1. TẢI CẤU HÌNH ===")
    config_path = os.path.join(os.path.dirname(__file__), "../configs/train_single.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    print("=== 2. KHỞI TẠO 4 MÔI TRƯỜNG SONG SONG ===")
    # Dùng SubprocVecEnv để ép CPU phân bổ mỗi môi trường lên 1 luồng độc lập
    env = make_vec_env(
        env_id=lambda: TrainEnvSingle(config), 
        n_envs=4, 
        vec_env_cls=SubprocVecEnv
    )

    # Thiết lập Hyperparameters tối ưu cho bài toán điều khiển liên tục ngắn hạn
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4,
        n_steps=512,            # 512 * 4 envs = 2048 steps mỗi lần update (cập nhật thường xuyên hơn)
        batch_size=64,          # Kích thước mini-batch
        n_epochs=10,
        gamma=0.99,             # Hệ số chiết khấu
        ent_coef=0.005,         # Giảm nhẹ khám phá ngẫu nhiên để quỹ đạo đỡ nhiễu
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]), # Mạng 2 lớp 128 node
            activation_fn=nn.Tanh                        # Tanh mượt hơn cho bài toán điều khiển vị trí
        ),
        verbose=1, 
        tensorboard_log="./logs/tensorboard/",
        device="cpu"
    )
    # BỌC THÊM VECNORMALIZE: Tự động chuẩn hóa State và Reward trong quá trình chạy
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    print("=== 3. BẮT ĐẦU HUẤN LUYỆN (PPO) ===")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/tensorboard/",device="cpu")
    
    timesteps = config['train']['total_timesteps']
    model.learn(total_timesteps=timesteps)
    
    print("=== 4. LƯU MÔ HÌNH TỰ ĐỘNG ===")
    base_save_path = config['train']['save_path']
    os.makedirs(os.path.dirname(base_save_path), exist_ok=True)
    
    final_save_path = get_auto_version_path(base_save_path)
    model.save(final_save_path)
    
    # BỔ SUNG LỆNH NÀY: Lưu lại bộ chuẩn hóa State
    env.save(final_save_path + "_vecnormalize.pkl")
    print(f"✅ Mô hình đã lưu tại: {final_save_path}.zip")

if __name__ == "__main__":
    main()
