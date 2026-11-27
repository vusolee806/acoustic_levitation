# scripts/evaluate_sac.py

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from HemisAcoustic.envs.train_env_single import TrainEnvSingle
from HemisAcoustic.envs.sac_wrapper import SACAcousticWrapper


def main():
    N_TEST_EPISODES = 20

    config_path = os.path.join(os.path.dirname(__file__), "../configs/train_sac_single.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_version = config["train"].get("model_version", "_v2")
    model_path = config["train"]["save_path"] + model_version
    print(f"Loading SAC model: {model_path}.zip")

    base_env = TrainEnvSingle(config)
    env = SACAcousticWrapper(base_env, train_reward_shaping=False)
    model = SAC.load(model_path + ".zip", device="cpu")

    # 1. Measure overall success rate across N episodes
    successes = 0
    last_traj, last_target = None, None

    for _ in range(N_TEST_EPISODES):
        obs, _ = env.reset()
        target_pos = np.array([base_env.target.x, base_env.target.y, base_env.target.z])
        traj = [np.array([base_env.particle.x, base_env.particle.y, base_env.particle.z])]

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                traj.append(info["particle_pos"])
                if info.get("is_success", False):
                    successes += 1
                break
            else:
                traj.append(np.array([base_env.particle.x, base_env.particle.y, base_env.particle.z]))

        last_traj = np.array(traj)
        last_target = target_pos

    print(f"SAC Success Rate: {successes}/{N_TEST_EPISODES} ({successes / N_TEST_EPISODES * 100:.1f}%)")

    # 2. Plot the last episode in 3D inside the ellipsoid workspace boundary
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_ell = config["env"]["radius"] * np.outer(np.cos(u), np.sin(v))
    y_ell = config["env"]["radius"] * np.outer(np.sin(u), np.sin(v))
    z_ell = config["env"]["z_center"] + config["env"]["z_radius"] * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_ell, y_ell, z_ell, color="gray", alpha=0.1)

    ax.plot(
        last_traj[:, 0], last_traj[:, 1], last_traj[:, 2],
        label="SAC Trajectory", color="blue", linewidth=2, marker="o", markersize=3,
    )
    ax.scatter(*last_traj[0], color="green", s=100, label="Start")
    ax.scatter(*last_target, color="red", s=100, label="Target")

    ax.set_title(f"3D Particle Path Planning (SB3 SAC {model_version})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
