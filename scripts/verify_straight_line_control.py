# scripts/verify_straight_line_control.py

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import mujoco

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from HemisAcoustic.physics.acoustic_force_hemi import HemisphereAcousticSolver


def generate_straight_line_trajectory(start_pos, target_pos, speed=0.03, dt_ctrl=0.005, hold_start=0.15, hold_end=0.25):
    """
    Traditional straight-line planner:
    1. Hold trap at start_pos so the ball settles at vertical equilibrium.
    2. Move linearly at constant velocity along the exact straight line to target_pos.
    3. Hold trap at target_pos to confirm stable levitation.
    """
    start_pos = np.asarray(start_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    dist = np.linalg.norm(target_pos - start_pos)
    n_hold_start = int(round(hold_start / dt_ctrl))
    n_move = max(2, int(np.ceil((dist / speed) / dt_ctrl)))
    n_hold_end = int(round(hold_end / dt_ctrl))

    alphas = np.linspace(0.0, 1.0, n_move)
    move_pts = start_pos[None, :] + alphas[:, None] * (target_pos - start_pos)[None, :]

    path = np.vstack([
        np.repeat(start_pos[None, :], n_hold_start, axis=0),
        move_pts,
        np.repeat(target_pos[None, :], n_hold_end, axis=0),
    ])
    times = np.arange(len(path)) * dt_ctrl
    return path, times


def compute_moving_slices(solver, x_act, trap_pos, half_span=0.018, res=35):
    """
    Computes local moving windows centered on trap_pos:
      - XY slice of Acoustic Pressure |p| at z = trap_pos[2]
      - XZ slice of Gor'kov Potential U  at y = trap_pos[1]
    """
    xs = np.linspace(trap_pos[0] - half_span, trap_pos[0] + half_span, res)
    ys = np.linspace(trap_pos[1] - half_span, trap_pos[1] + half_span, res)
    zs = np.linspace(trap_pos[2] - half_span, trap_pos[2] + half_span, res)

    # XY grid for Pressure |p|
    X_xy, Y_xy = np.meshgrid(xs, ys)
    Z_xy = np.full_like(X_xy, trap_pos[2])
    pts_xy = torch.tensor(np.stack([X_xy.ravel(), Y_xy.ravel(), Z_xy.ravel()], axis=-1), dtype=torch.float32)
    P_xy = solver.compute_pressure(pts_xy, x_act).reshape(res, res).numpy()

    # XZ grid for Gor'kov U
    X_xz, Z_xz = np.meshgrid(xs, zs)
    Y_xz = np.full_like(X_xz, trap_pos[1])
    pts_xz = torch.tensor(np.stack([X_xz.ravel(), Y_xz.ravel(), Z_xz.ravel()], axis=-1), dtype=torch.float32)
    U_xz = solver.compute_gorkov_potential(pts_xz, x_act).reshape(res, res).numpy()

    return (X_xy * 1000, Y_xy * 1000, P_xy), (X_xz * 1000, Z_xz * 1000, U_xz)


def run_case(case_name, start_pos, target_pos, p0=2.8, speed=0.03, dt_ctrl=0.005, live_heatmap=True):
    """
    Simulates a straight-line move in MuJoCo using Naive Twin-Trap radiation forces,
    tracks setup vs. real trajectory, and renders a live XY + XZ moving heatmap.
    """
    xml_path = os.path.join(PROJECT_ROOT, "HemisAcoustic", "assets", "mujoco_levitator.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    solver = HemisphereAcousticSolver(mj_model, p0=p0)

    # Verify exact start and target lifting safety factors before running
    start_sf = solver.estimate_max_upward_force(np.asarray(start_pos)) / solver.mg
    target_sf = solver.estimate_max_upward_force(np.asarray(target_pos)) / solver.mg
    print(f"\n==================== {case_name} ====================")
    print(f"  Start : {np.array(start_pos)*1000} mm  (Fz_max / mg = {start_sf:.2f}x)")
    print(f"  Target: {np.array(target_pos)*1000} mm  (Fz_max / mg = {target_sf:.2f}x)")

    # Locate test_ball in MuJoCo
    ball_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "test_ball")
    ball_jnt_id = mj_model.body_jntadr[ball_body_id]
    qpos_adr = mj_model.jnt_qposadr[ball_jnt_id]
    qvel_adr = mj_model.jnt_dofadr[ball_jnt_id]

    mj_data.qpos[qpos_adr : qpos_adr + 3] = start_pos
    mj_data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # Build straight-line setup trajectory
    setup_path, times = generate_straight_line_trajectory(
        start_pos, target_pos, speed=speed, dt_ctrl=dt_ctrl
    )

    # FIX #3: Exact 1:1 physics time synchronization
    substeps = max(1, int(round(dt_ctrl / mj_model.opt.timestep)))
    # Light viscous damping so the ball stabilizes in the well without hiding gravity drops
    c_damp = 1.5e-5

    real_path = []
    fz_log = []
    dropped = False
    drop_idx = None

    if live_heatmap:
        plt.ion()
        fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(13.5, 5.5))
        fig.suptitle(f"{case_name} — Live Moving Acoustic Field (Naive Twin-Trap)", fontsize=12)

    for step_idx, trap_pos in enumerate(setup_path):
        # Solve Naive + Twin-Trap phases ONCE per control step
        x_act = solver.solve_phases_naive(trap_pos, twin_trap=True)

        # Advance MuJoCo by exact dt_ctrl
        last_F = np.zeros(3)
        for _ in range(substeps):
            ball_pos = mj_data.qpos[qpos_adr : qpos_adr + 3].copy()
            ball_vel = mj_data.qvel[qvel_adr : qvel_adr + 3].copy()

            F_rad = solver.compute_radiation_force_from_phases(ball_pos, x_act)
            F_damp = -c_damp * ball_vel

            mj_data.xfrc_applied[ball_body_id, :3] = F_rad + F_damp
            mj_data.xfrc_applied[ball_body_id, 3:] = 0.0
            mujoco.mj_step(mj_model, mj_data)
            last_F = F_rad

        ball_pos = mj_data.qpos[qpos_adr : qpos_adr + 3].copy()
        real_path.append(ball_pos)
        fz_log.append(last_F[2])

        # Detect drop (ball fell > 12 mm below trap or escaped > 15 mm away)
        if (not dropped) and (
            (trap_pos[2] - ball_pos[2] > 0.012) or (np.linalg.norm(ball_pos - trap_pos) > 0.015)
        ):
            dropped = True
            drop_idx = step_idx
            print(f"  [!] DROP DETECTED at step {step_idx} (t={times[step_idx]:.2f}s), "
                  f"trap={trap_pos*1000} mm, ball={ball_pos*1000} mm")

        # Update live moving heatmap every 5 control steps
        if live_heatmap and (step_idx % 5 == 0 or step_idx == len(setup_path) - 1 or step_idx == drop_idx):
            (X_xy, Y_xy, P_xy), (X_xz, Z_xz, U_xz) = compute_moving_slices(
                solver, x_act, trap_pos, half_span=0.018, res=33
            )
            hist = np.array(real_path) * 1000.0
            setup_mm = setup_path * 1000.0
            tp_mm = trap_pos * 1000.0
            bp_mm = ball_pos * 1000.0

            ax_xy.clear()
            ax_xz.clear()

            # Left: Moving XY Pressure Heatmap
            ax_xy.contourf(X_xy, Y_xy, P_xy, levels=22, cmap="inferno")
            ax_xy.plot(setup_mm[:, 0], setup_mm[:, 1], "w--", lw=1.4, label="Setup Straight Line")
            ax_xy.plot(hist[:, 0], hist[:, 1], "c-", lw=2.0, label="Real Ball Path")
            ax_xy.scatter(tp_mm[0], tp_mm[1], color="lime", marker="+", s=130, lw=2, label="Trap Center")
            ax_xy.scatter(bp_mm[0], bp_mm[1], color="cyan", edgecolors="black", s=85, label="Ball (MuJoCo)")
            ax_xy.set_xlim(tp_mm[0] - 18, tp_mm[0] + 18)
            ax_xy.set_ylim(tp_mm[1] - 18, tp_mm[1] + 18)
            ax_xy.set_title(f"XY Pressure |p| (t={times[step_idx]:.2f}s) | Status: {'DROPPED' if dropped else 'TRAPPED'}")
            ax_xy.set_xlabel("X (mm)")
            ax_xy.set_ylabel("Y (mm)")
            ax_xy.legend(loc="upper right", fontsize=8)

            # Right: Moving XZ Gor'kov Potential Heatmap (shows vertical drop directly!)
            ax_xz.contourf(X_xz, Z_xz, U_xz * 1e6, levels=22, cmap="viridis")
            ax_xz.plot(setup_mm[:, 0], setup_mm[:, 2], "w--", lw=1.4, label="Setup Straight Line")
            ax_xz.plot(hist[:, 0], hist[:, 2], "r-", lw=2.0, label="Real Ball Path")
            ax_xz.scatter(tp_mm[0], tp_mm[2], color="lime", marker="+", s=130, lw=2, label="Trap Center")
            ax_xz.scatter(bp_mm[0], bp_mm[2], color="red", edgecolors="white", s=85, label="Ball (MuJoCo)")
            ax_xz.set_xlim(tp_mm[0] - 18, tp_mm[0] + 18)
            ax_xz.set_ylim(tp_mm[2] - 18, tp_mm[2] + 18)
            ax_xz.set_title(f"XZ Gor'kov U (µJ) | Fz={last_F[2]*1e6:5.1f} µN vs mg={solver.mg*1e6:4.1f} µN")
            ax_xz.set_xlabel("X (mm)")
            ax_xz.set_ylabel("Z (mm)")
            ax_xz.legend(loc="upper right", fontsize=8)

            plt.pause(0.01)

        # Stop shortly after the ball hits the floor so the fail plot stays clean
        if dropped and ball_pos[2] < -0.10:
            break

    if live_heatmap:
        plt.ioff()
        plt.close(fig)

    real_path = np.array(real_path)
    n = len(real_path)
    setup_trim = setup_path[:n]
    times_trim = times[:n]
    fz_log = np.array(fz_log)

    final_err_mm = np.linalg.norm(real_path[-1] - target_pos) * 1000.0
    mean_track_mm = np.mean(np.linalg.norm(real_path - setup_trim, axis=1)) * 1000.0
    print(f"  Result: {'FAILED (DROPPED)' if dropped else 'SUCCESS'} | "
          f"Mean Track Err: {mean_track_mm:.2f} mm | Final Dist to Target: {final_err_mm:.2f} mm")

    return {
        "name": case_name,
        "setup": setup_trim,
        "real": real_path,
        "times": times_trim,
        "fz": fz_log,
        "mg": solver.mg,
        "target": np.asarray(target_pos),
        "dropped": dropped,
        "drop_idx": drop_idx,
    }


def plot_tracking_summary(res_good, res_fail):
    """Side-by-side tracking verification for both Good Case and Fail Case."""
    fig = plt.figure(figsize=(15, 9))

    for col, res in enumerate([res_good, res_fail]):
        setup_mm = res["setup"] * 1000.0
        real_mm = res["real"] * 1000.0
        tgt_mm = res["target"] * 1000.0
        t = res["times"]

        # Row 1: 3D Trajectory Comparison (Setup vs Real)
        ax3d = fig.add_subplot(3, 2, col + 1, projection="3d")
        ax3d.plot(setup_mm[:, 0], setup_mm[:, 1], setup_mm[:, 2], "b--", lw=2, label="Setup Straight Line")
        ax3d.plot(real_mm[:, 0], real_mm[:, 1], real_mm[:, 2], "r-", lw=2, label="Real Ball (MuJoCo)")
        ax3d.scatter(*setup_mm[0], color="green", s=60, label="Start")
        ax3d.scatter(*tgt_mm, color="black", marker="*", s=110, label="Target")
        if res["drop_idx"] is not None:
            di = res["drop_idx"]
            ax3d.scatter(*real_mm[di], color="orange", s=90, label="Drop Point")
        ax3d.set_title(f"{res['name']}: 3D Real vs. Setup Trajectory")
        ax3d.set_xlabel("X (mm)")
        ax3d.set_ylabel("Y (mm)")
        ax3d.set_zlabel("Z (mm)")
        ax3d.legend(fontsize=8)

        # Row 2: X(t) and Z(t) Tracking Over Time
        ax_xz = fig.add_subplot(3, 2, col + 3)
        ax_xz.plot(t, setup_mm[:, 0], "b--", label="Setup X")
        ax_xz.plot(t, real_mm[:, 0], "b-", lw=1.8, label="Real X")
        ax_xz.plot(t, setup_mm[:, 2], "r--", label="Setup Z")
        ax_xz.plot(t, real_mm[:, 2], "r-", lw=1.8, label="Real Z")
        if res["drop_idx"] is not None:
            ax_xz.axvline(t[res["drop_idx"]], color="orange", ls=":", lw=2, label="Drop Time")
        ax_xz.set_title("Position Tracking Over Time (X & Z)")
        ax_xz.set_xlabel("Time (s)")
        ax_xz.set_ylabel("Position (mm)")
        ax_xz.grid(True, alpha=0.3)
        ax_xz.legend(fontsize=8, ncol=2)

        # Row 3: Tracking Error & Vertical Force Fz vs Gravity mg
        ax_err = fig.add_subplot(3, 2, col + 5)
        err_mm = np.linalg.norm(real_mm - setup_mm, axis=1)
        ax_err.plot(t, err_mm, "purple", lw=2, label="3D Tracking Error (mm)")
        ax_err.set_xlabel("Time (s)")
        ax_err.set_ylabel("Tracking Error (mm)", color="purple")
        ax_err.grid(True, alpha=0.3)

        ax_f = ax_err.twinx()
        ax_f.plot(t, res["fz"] * 1e6, "g-", alpha=0.75, label="Acoustic Fz (µN)")
        ax_f.axhline(res["mg"] * 1e6, color="red", ls="--", lw=1.5, label=f"Gravity mg ({res['mg']*1e6:.1f} µN)")
        ax_f.set_ylabel("Vertical Force (µN)", color="green")
        ax_err.set_title("Tracking Error & Vertical Lifting Force vs. Gravity")

        h1, l1 = ax_err.get_legend_handles_labels()
        h2, l2 = ax_f.get_legend_handles_labels()
        ax_err.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()


def main():
    # Use identical solver power p0 for both cases so the only difference is the path zone!
    # Check verify_piston_and_field.py first: adjust P0 if needed so center Fz_max/mg is ~2.5x–3.5x.
    P0 = 2.8
    SPEED = 0.03  # 30 mm/s straight-line motion

    # GOOD CASE: Straight line inside the strong trapping zone
    res_good = run_case(
        case_name="GOOD CASE (Strong Acoustic Zone)",
        start_pos=[0.0, 0.0, 0.0],
        target_pos=[0.020, 0.0, 0.005],
        p0=P0,
        speed=SPEED,
        live_heatmap=True,
    )

    # FAIL CASE: Straight line pushing into the weak edge where Fz_max drops below mg
    res_fail = run_case(
        case_name="FAIL CASE (Moves Into Weak Zone & Drops)",
        start_pos=[0.0, 0.0, 0.0],
        target_pos=[0.060, 0.0, 0.045],
        p0=P0,
        speed=SPEED,
        live_heatmap=True,
    )

    plot_tracking_summary(res_good, res_fail)


if __name__ == "__main__":
    main()
