# scripts/evaluate_sac.py

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from HemisAcoustic.envs.train_env_single import TrainEnvSingle


def main():
    MODEL_VERSION = "_v1"
    N_TEST_EPISODES = 20

    config_path = os.path.join(os.path.dirname(__file__), "../configs/train_sac_single.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config["train"]["save_path"] + MODEL_VERSION

    env = DummyVecEnv([lambda: TrainEnvSingle(config)])
    model = SAC.load(model_path + ".zip", device="cpu")
    env = VecNormalize.load(model_path + "_vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False

    base_env = env.venv.envs[0]

    # 1. Measure overall success rate across N episodes
    successes = 0
    last_traj, last_target = None, None

    for ep in range(N_TEST_EPISODES):
        obs = env.reset()
        target_pos = np.array([base_env.target.x, base_env.target.y, base_env.target.z])
        traj = [np.array([base_env.particle.x, base_env.particle.y, base_env.particle.z])]

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done[0]:
                traj.append(info[0]["particle_pos"])
                if info[0].get("is_success", False):
                    successes += 1
                break
            else:
                traj.append(np.array([base_env.particle.x, base_env.particle.y, base_env.particle.z]))

        last_traj = np.array(traj)
        last_target = target_pos

    print(f"SAC Success Rate: {successes}/{N_TEST_EPISODES} ({successes / N_TEST_EPISODES * 100:.1f}%)")

    # 2. Plot the last episode in 3D inside the hemisphere boundary
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 50)
    x_sphere = config["env"]["radius"] * np.outer(np.cos(u), np.sin(v))
    y_sphere = config["env"]["radius"] * np.outer(np.sin(u), np.sin(v))
    z_sphere = config["env"]["z_center"] - config["env"]["radius"] * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.1)

    ax.plot(
        last_traj[:, 0], last_traj[:, 1], last_traj[:, 2],
        label="SAC Trajectory", color="blue", linewidth=2, marker="o", markersize=3,
    )
    ax.scatter(*last_traj[0], color="green", s=100, label="Start")
    ax.scatter(*last_target, color="red", s=100, label="Target")

    ax.set_title("3D Particle Path Planning (SB3 SAC)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
