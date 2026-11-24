# scripts/visualize_heatmap_moving.py

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import mujoco

from simulate_mujoco_ppo import generate_ppo_trajectory, smooth_and_interpolate_path
from HemisAcoustic.physics.acoustic_force_hemi import HemisphereAcousticSolver


def compute_2d_slice(solver, x_activations, z_plane, x_range=(-0.04, 0.04), y_range=(-0.04, 0.04), res=50):
    """
    Computes a 2D heatmap grid of Acoustic Pressure |p| and Gor'kov Potential U
    on the X-Y plane at height z = z_plane (or X-Z plane).
    """
    xs = np.linspace(x_range[0], x_range[1], res)
    ys = np.linspace(y_range[0], y_range[1], res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, z_plane)

    grid_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    grid_tensor = torch.tensor(grid_pts, dtype=torch.float32, device=solver.device)

    # 1. Compute Acoustic Pressure |p| (Pa)
    H = solver.piston_model_hemi(grid_tensor)
    p = torch.abs(torch.matmul(H, x_activations)).reshape(res, res).numpy()

    # 2. Compute Gor'kov Potential U (Joules)
    U = solver.compute_gorkov_potential(grid_tensor, x_activations).reshape(res, res).numpy()

    return X * 1000, Y * 1000, p, U


def main():
    CUSTOM_TARGET = [0.025, -0.015, 0.005]
    MODEL_VERSION = "_v8"

    print("=== 1. GENERATING PPO TRAJECTORY ===")
    config_path = os.path.join(os.path.dirname(__file__), "../configs/train_single.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    waypoints, target_pos = generate_ppo_trajectory(config, model_version=MODEL_VERSION, custom_target=CUSTOM_TARGET)
    fine_trap_path = smooth_and_interpolate_path(waypoints, dt_rl=config['env']['delta_time'], dt_sim=0.005)

    print("=== 2. INITIALIZING MUJOCO & ACOUSTIC SOLVER ===")
    xml_path = os.path.join(os.path.dirname(__file__), "../HemisAcoustic/assets/mujoco_levitator.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    solver = HemisphereAcousticSolver(mj_model, p0=40.0)

    ball_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "test_ball")
    ball_jnt_id = mj_model.body_jntadr[ball_body_id]
    qpos_adr = mj_model.jnt_qposadr[ball_jnt_id]
    qvel_adr = mj_model.jnt_dofadr[ball_jnt_id]

    mj_data.qpos[qpos_adr:qpos_adr+3] = waypoints[0]
    mj_data.qvel[qvel_adr:qvel_adr+6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    print("=== 3. RUNNING LIVE HEATMAP & FORCE VECTOR VERIFICATION ===")
    plt.ion()
    fig, (ax_p, ax_u) = plt.subplots(1, 2, figsize=(13, 5.5))

    actual_history = []
    substeps = 10

    for step_idx, trap_pos in enumerate(fine_trap_path):
        # Step MuJoCo physics
        for _ in range(substeps):
            ball_pos = mj_data.qpos[qpos_adr:qpos_adr+3].copy()
            ball_vel = mj_data.qvel[qvel_adr:qvel_adr+3].copy()
            F_rad = solver.compute_radiation_force(ball_pos, trap_pos)
            F_drag = -0.00015 * ball_vel
            mj_data.xfrc_applied[ball_body_id, :3] = F_rad + F_drag
            mujoco.mj_step(mj_model, mj_data)

        ball_pos = mj_data.qpos[qpos_adr:qpos_adr+3].copy()
        actual_history.append(ball_pos)

        # Update heatmap every 3 steps for smooth rendering speed
        if step_idx % 3 == 0 or step_idx == len(fine_trap_path) - 1:
            x_act = solver.solve_phases_twin_trap(trap_pos)
            X_mm, Y_mm, P_grid, U_grid = compute_2d_slice(
                solver, x_act, z_plane=ball_pos[2], x_range=(-0.035, 0.035), y_range=(-0.035, 0.035), res=45
            )

            ax_p.clear()
            ax_u.clear()
            hist_arr = np.array(actual_history) * 1000.0

            # Left Panel: Acoustic Pressure |p| (Twin-Trap Tweezers)
            cp1 = ax_p.contourf(X_mm, Y_mm, P_grid, levels=25, cmap='inferno')
            ax_p.plot(waypoints[:, 0]*1000, waypoints[:, 1]*1000, 'w--', alpha=0.7, label='PPO Path')
            ax_p.plot(hist_arr[:, 0], hist_arr[:, 1], 'c-', linewidth=2, label='Ball Trail')
            ax_p.scatter(trap_pos[0]*1000, trap_pos[1]*1000, color='lime', marker='+', s=120, label='Trap Center')
            ax_p.scatter(ball_pos[0]*1000, ball_pos[1]*1000, color='cyan', edgecolors='black', s=90, label='Ball (MuJoCo)')
            ax_p.set_title(f"Acoustic Pressure |p| (Step {step_idx}/{len(fine_trap_path)})\nDark Node Between 2 Bright Tweezers")
            ax_p.set_xlabel("X (mm)")
            ax_p.set_ylabel("Y (mm)")
            ax_p.legend(loc='upper right', fontsize=8)

            # Right Panel: Gor'kov Potential U (Dark Blue = Potential Well Minimum)
            cp2 = ax_u.contourf(X_mm, Y_mm, U_grid, levels=25, cmap='viridis')
            ax_u.plot(waypoints[:, 0]*1000, waypoints[:, 1]*1000, 'w--', alpha=0.7)
            ax_u.scatter(trap_pos[0]*1000, trap_pos[1]*1000, color='lime', marker='+', s=120, label='Trap Minimum')
            ax_u.scatter(ball_pos[0]*1000, ball_pos[1]*1000, color='red', edgecolors='white', s=90, label='Ball Position')
            ax_u.set_title(f"Gor'kov Potential U (Z = {ball_pos[2]*1000:.1f} mm)\nDark Blue Well Pulls the Red Ball")
            ax_u.set_xlabel("X (mm)")
            ax_u.set_ylabel("Y (mm)")
            ax_u.legend(loc='upper right', fontsize=8)

            plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
