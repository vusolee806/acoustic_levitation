# scripts/simulate_mujoco_ddpg.py

import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from scipy.interpolate import interp1d
from stable_baselines3 import DDPG

from HemisAcoustic.envs.train_env_single import TrainEnvSingle
from HemisAcoustic.envs.sac_wrapper import SACAcousticWrapper
from HemisAcoustic.physics.acoustic_force_hemi import HemisphereAcousticSolver


def generate_ddpg_trajectory(config, model_version="_v1", custom_target=None):
    """Roll out the trained SB3 DDPG model using SACAcousticWrapper."""
    base_env = TrainEnvSingle(config)
    env = SACAcousticWrapper(base_env, train_reward_shaping=False)

    model_path = config["train"]["save_path"] + model_version
    print(f"Loading DDPG model: {model_path}.zip")
    model = DDPG.load(model_path + ".zip", device="cpu")

    obs, _ = env.reset()

    if custom_target is not None:
        cx = float(custom_target[0])
        cy = float(custom_target[1])
        cz = float(custom_target[2]) - config["env"]["z_center"]
        ellipsoid_val = (
            (cx ** 2 / config["env"]["radius"] ** 2)
            + (cy ** 2 / config["env"]["radius"] ** 2)
            + (cz ** 2 / config["env"]["z_radius"] ** 2)
        )
        if ellipsoid_val >= 1.0:
            print(
                f"[WARNING] CUSTOM_TARGET {custom_target} is OUTSIDE workspace "
                f"(ellipsoid={ellipsoid_val:.3f} >= 1.0)!"
            )

        base_env.target.x = float(custom_target[0])
        base_env.target.y = float(custom_target[1])
        base_env.target.z = float(custom_target[2])
        base_env.particle.last_timestep_dist = np.linalg.norm(
            np.array([base_env.particle.x, base_env.particle.y, base_env.particle.z]) - np.array(custom_target)
        )
        obs = env._scale_obs(base_env._get_obs())

    target_pos = np.array([base_env.target.x, base_env.target.y, base_env.target.z])
    waypoints = [np.array([base_env.particle.x, base_env.particle.y, base_env.particle.z])]

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            waypoints.append(info["particle_pos"])
            break
        else:
            waypoints.append(np.array([base_env.particle.x, base_env.particle.y, base_env.particle.z]))

    return np.array(waypoints), target_pos


def smooth_and_interpolate_path(waypoints, dt_rl=0.02, dt_sim=0.005):
    """Upsample coarse DDPG waypoints to fine acoustic trap control steps."""
    diffs = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    valid_mask = np.insert(diffs > 1e-6, 0, True)
    waypoints = waypoints[valid_mask]

    n_pts = len(waypoints)
    if n_pts < 2:
        return waypoints

    t_coarse = np.linspace(0, (n_pts - 1) * dt_rl, n_pts)
    t_fine = np.arange(0, t_coarse[-1], dt_sim)

    kind = "cubic" if n_pts > 3 else "linear"
    interpolator = interp1d(t_coarse, waypoints, axis=0, kind=kind)
    return interpolator(t_fine)


def add_visual_sphere(scn, pos, radius, rgba):
    """Draw a visual marker in the MuJoCo viewer."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def plot_comparison(waypoints, fine_trap_path, actual_ball_path, target_pos):
    """Plot 3D trajectory comparison and tracking error over time."""
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(
        waypoints[:, 0] * 1000, waypoints[:, 1] * 1000, waypoints[:, 2] * 1000,
        "b--o", markersize=4, label="DDPG Planned Waypoints",
    )
    ax1.plot(
        actual_ball_path[:, 0] * 1000, actual_ball_path[:, 1] * 1000, actual_ball_path[:, 2] * 1000,
        "r-", linewidth=2, label="Actual MuJoCo Ball Path",
    )
    ax1.scatter(*(waypoints[0] * 1000), color="green", s=100, label="Start (0,0,0)")
    ax1.scatter(*(target_pos * 1000), color="red", s=120, marker="*", label="Target")

    ax1.set_title("3D Path Comparison: DDPG Plan vs. MuJoCo Physics")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    min_len = min(len(fine_trap_path), len(actual_ball_path))
    errors_mm = np.linalg.norm(fine_trap_path[:min_len] - actual_ball_path[:min_len], axis=1) * 1000.0
    dist_to_target_mm = np.linalg.norm(actual_ball_path - target_pos, axis=1) * 1000.0

    ax2.plot(errors_mm, label="Trap-to-Ball Lag Error (mm)", color="orange")
    ax2.plot(dist_to_target_mm, label="Ball Distance to Target (mm)", color="purple")
    ax2.set_title(f"DDPG Tracking Performance (Mean Error: {np.mean(errors_mm):.2f} mm)")
    ax2.set_xlabel("Simulation Control Step")
    ax2.set_ylabel("Error / Distance (mm)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Valid interior ellipsoid: (x/0.03)^2 + (y/0.03)^2 + (z/0.015)^2 < 1.0
    CUSTOM_TARGET = [0.018, -0.012, 0.010]
    PLAYBACK_DELAY = 0.015

    print("=== 1. GENERATE PATH FROM SB3 DDPG MODEL ===")
    config_path = os.path.join(os.path.dirname(__file__), "../configs/train_ddpg_single.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_version = config["train"].get("model_version", "_v1")

    waypoints, target_pos = generate_ddpg_trajectory(
        config, model_version=model_version, custom_target=CUSTOM_TARGET
    )
    print(f"DDPG generated {len(waypoints)} waypoints.")
    print(f"Start: {waypoints[0].round(4)} m -> Target: {target_pos.round(4)} m")

    fine_trap_path = smooth_and_interpolate_path(
        waypoints, dt_rl=config["env"]["delta_time"], dt_sim=0.005
    )

    print("=== 2. INITIALIZE MUJOCO & ACOUSTIC SOLVER ===")
    xml_path = os.path.join(os.path.dirname(__file__), "../HemisAcoustic/assets/mujoco_levitator.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    acoustic_solver = HemisphereAcousticSolver(mj_model, p0=40.0)

    ball_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "test_ball")
    ball_jnt_id = mj_model.body_jntadr[ball_body_id]
    qpos_adr = mj_model.jnt_qposadr[ball_jnt_id]
    qvel_adr = mj_model.jnt_dofadr[ball_jnt_id]

    mj_data.qpos[qpos_adr:qpos_adr + 3] = waypoints[0]
    mj_data.qvel[qvel_adr:qvel_adr + 6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    print("=== 3. LAUNCH MUJOCO 3D SIMULATION ===")
    print("  - RED Sphere    : Target Position")
    print("  - GREEN Sphere  : Moving Acoustic Trap (Gor'kov Minimum)")
    print("  - BLUE Dots     : DDPG Planned Trajectory")
    print("  - YELLOW Trail  : Actual MuJoCo Ball Path")

    actual_ball_path = []
    visual_trail = []

    hold_start_steps = 100
    hold_end_steps = 150
    total_steps = hold_start_steps + len(fine_trap_path) + hold_end_steps
    substeps_per_trap_update = 10

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, -0.01]
        viewer.cam.distance = 0.22
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 45

        for step_idx in range(total_steps):
            if not viewer.is_running():
                break

            step_start = time.time()

            if step_idx < hold_start_steps:
                trap_pos = fine_trap_path[0]
            elif step_idx < hold_start_steps + len(fine_trap_path):
                trap_pos = fine_trap_path[step_idx - hold_start_steps]
            else:
                trap_pos = fine_trap_path[-1]

            for _ in range(substeps_per_trap_update):
                ball_pos = mj_data.qpos[qpos_adr:qpos_adr + 3].copy()
                ball_vel = mj_data.qvel[qvel_adr:qvel_adr + 3].copy()

                F_rad = acoustic_solver.compute_radiation_force(ball_pos, trap_pos)
                F_drag = -0.00015 * ball_vel

                mj_data.xfrc_applied[ball_body_id, :3] = F_rad + F_drag
                mj_data.xfrc_applied[ball_body_id, 3:] = 0.0

                mujoco.mj_step(mj_model, mj_data)

            current_ball_pos = mj_data.qpos[qpos_adr:qpos_adr + 3].copy()
            if hold_start_steps <= step_idx < hold_start_steps + len(fine_trap_path):
                actual_ball_path.append(current_ball_pos)
                if step_idx % 5 == 0:
                    visual_trail.append(current_ball_pos)

            viewer.user_scn.ngeom = 0
            add_visual_sphere(viewer.user_scn, target_pos, 0.0025, [1, 0, 0, 0.8])
            add_visual_sphere(viewer.user_scn, trap_pos, 0.003, [0, 1, 0, 0.35])
            for wp in waypoints:
                add_visual_sphere(viewer.user_scn, wp, 0.0008, [0.1, 0.4, 1.0, 0.7])
            for tp in visual_trail:
                add_visual_sphere(viewer.user_scn, tp, 0.0007, [1.0, 0.9, 0.0, 0.9])

            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < PLAYBACK_DELAY:
                time.sleep(PLAYBACK_DELAY - elapsed)

    if len(actual_ball_path) > 0:
        print("=== 4. PLOTTING ACTUAL vs PLANNED PATH COMPARISON ===")
        plot_comparison(waypoints, fine_trap_path, np.array(actual_ball_path), target_pos)


if __name__ == "__main__":
    main()
