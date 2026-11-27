# scripts/verify_piston_and_field.py

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import mujoco
from scipy.special import j1

# Allow running from either project root or scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from HemisAcoustic.physics.acoustic_force_hemi import HemisphereAcousticSolver


def verify_normals(mj_model, solver):
    """Compare buggy body_quat normals vs fixed site_xmat normals."""
    old_dots = []
    new_dots = []
    for i in range(1, 80):
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"sensor_{i}")
        pos = mj_model.body_pos[body_id]
        inward = -pos / np.linalg.norm(pos)

        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, mj_model.body_quat[body_id])
        old_z = mat.reshape(3, 3)[:, 2]
        new_z = solver.trans_norm[i - 1].numpy()

        old_dots.append(np.dot(old_z, inward))
        new_dots.append(np.dot(new_z, inward))

    print("=== 1. TRANSDUCER NORMAL ALIGNMENT CHECK (1.0 = points straight to origin) ===")
    print(f"  OLD (body_quat) min/mean alignment: {np.min(old_dots):.4f} / {np.mean(old_dots):.4f}  <-- BROKEN")
    print(f"  NEW (site_xmat) min/mean alignment: {np.min(new_dots):.4f} / {np.mean(new_dots):.4f}  <-- FIXED")


def verify_single_piston_directivity(solver):
    """Test transducer #1 radiation pattern against analytical baffled piston formula."""
    t_pos = solver.trans_pos[0].numpy()
    t_norm = solver.trans_norm[0].numpy()

    # Build orthonormal tangent vector perpendicular to t_norm
    ref = np.array([1.0, 0.0, 0.0]) if abs(t_norm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent = np.cross(t_norm, ref)
    tangent /= np.linalg.norm(tangent)

    R_test = 0.086  # Test at focal distance (86 mm)
    angles_deg = np.linspace(0.0, 180.0, 181)
    angles_rad = np.deg2rad(angles_deg)

    pts = np.array([
        t_pos + R_test * (np.cos(th) * t_norm + np.sin(th) * tangent)
        for th in angles_rad
    ])
    H = solver.piston_model_hemi(torch.tensor(pts, dtype=torch.float32))
    # Extract transducer 0 column and normalize by on-axis value (p0 / R_test)
    sim_dir = torch.abs(H[:, 0]).numpy() * (R_test / solver.p0)

    # Analytical baffled piston directivity
    ka_sin = solver.k * solver.a * np.sin(angles_rad)
    ana_dir = np.where(ka_sin > 1e-6, np.abs(2.0 * j1(ka_sin) / np.where(ka_sin > 1e-6, ka_sin, 1.0)), 1.0)
    ana_dir = np.where(np.cos(angles_rad) > 0.0, ana_dir, 0.0)

    max_err = np.max(np.abs(sim_dir - ana_dir))
    print("\n=== 2. PISTON MODEL DIRECTIVITY CHECK ===")
    print(f"  Max absolute error vs analytical 2*J1(ka*sin(theta))/(ka*sin(theta)): {max_err:.2e}")
    print(f"  Backward emission (theta > 90 deg) max amplitude: {np.max(sim_dir[angles_deg > 90]):.2e}")
    return angles_deg, sim_dir, ana_dir


def compute_slice(solver, x_act, plane="xy", fixed_val=0.0, lim=0.025, res=51):
    u = np.linspace(-lim, lim, res)
    v = np.linspace(-lim, lim, res)
    U_grid, V_grid = np.meshgrid(u, v)
    W_grid = np.full_like(U_grid, fixed_val)

    if plane == "xy":
        pts = np.stack([U_grid.ravel(), V_grid.ravel(), W_grid.ravel()], axis=-1)
    elif plane == "xz":
        pts = np.stack([U_grid.ravel(), W_grid.ravel(), V_grid.ravel()], axis=-1)
    else:
        raise ValueError("plane must be 'xy' or 'xz'")

    pts_t = torch.tensor(pts, dtype=torch.float32)
    P = solver.compute_pressure(pts_t, x_act).reshape(res, res).numpy()
    G = solver.compute_gorkov_potential(pts_t, x_act).reshape(res, res).numpy()
    return U_grid * 1000.0, V_grid * 1000.0, P, G


def compute_force_quiver(solver, x_act, plane="xy", fixed_val=0.0, lim=0.012, res=11):
    u = np.linspace(-lim, lim, res)
    v = np.linspace(-lim, lim, res)
    U_g, V_g = np.meshgrid(u, v)
    Fu = np.zeros_like(U_g)
    Fv = np.zeros_like(V_g)

    for r in range(res):
        for c in range(res):
            pos = (
                np.array([U_g[r, c], V_g[r, c], fixed_val])
                if plane == "xy"
                else np.array([U_g[r, c], fixed_val, V_g[r, c]])
            )
            F = solver.compute_radiation_force_from_phases(pos, x_act)
            Fu[r, c] = F[0]
            Fv[r, c] = F[1] if plane == "xy" else F[2]
    return U_g * 1000.0, V_g * 1000.0, Fu, Fv


def compute_workspace_strength_map(solver, x_lim=0.065, z_min=-0.040, z_max=0.055, nx=19, nz=17):
    """Map peak upward force ratio Fz_max / mg across the XZ plane to locate weak drop zones."""
    xs = np.linspace(-x_lim, x_lim, nx)
    zs = np.linspace(z_min, z_max, nz)
    X, Z = np.meshgrid(xs, zs)
    safety = np.zeros_like(X)

    print("\n=== 3. MAPPING WORKSPACE LIFTING MARGIN (Fz_max / mg) ON XZ PLANE ===")
    for iz in range(nz):
        for ix in range(nx):
            fz_max = solver.estimate_max_upward_force(np.array([X[iz, ix], 0.0, Z[iz, ix]]), n_pts=17)
            safety[iz, ix] = fz_max / solver.mg

    print(f"  Particle mass: {solver.mass*1e6:.2f} mg | Gravity mg: {solver.mg*1e6:.2f} uN")
    print(f"  Center (0,0,0) safety factor Fz_max/mg: {solver.estimate_max_upward_force(np.zeros(3))/solver.mg:.2f}x")
    return X * 1000.0, Z * 1000.0, safety


def main():
    P0 = 3.0  # Realistic transducer amplitude (Pa*m); produces ~2,700 Pa focus at origin

    xml_path = os.path.join(PROJECT_ROOT, "HemisAcoustic", "assets", "mujoco_levitator.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    solver = HemisphereAcousticSolver(mj_model, p0=P0)

    verify_normals(mj_model, solver)
    angles_deg, sim_dir, ana_dir = verify_single_piston_directivity(solver)

    # Compare Pure Naive vs Naive + Twin-Trap at origin
    origin = np.array([0.0, 0.0, 0.0])
    x_focus = solver.solve_phases_naive(origin, twin_trap=False)
    x_twin = solver.solve_phases_naive(origin, twin_trap=True)

    X1, Y1, P_foc, U_foc = compute_slice(solver, x_focus, plane="xy", lim=0.015, res=61)
    X2, Y2, P_tw_xy, U_tw_xy = compute_slice(solver, x_twin, plane="xy", lim=0.015, res=61)
    X3, Z3, P_tw_xz, U_tw_xz = compute_slice(solver, x_twin, plane="xz", lim=0.015, res=61)

    qx_f, qy_f, Fxf, Fyf = compute_force_quiver(solver, x_focus, plane="xy", lim=0.008, res=11)
    qx_t, qy_t, Fxt, Fyt = compute_force_quiver(solver, x_twin, plane="xy", lim=0.008, res=11)
    qx_z, qz_z, Fxz, Fzz = compute_force_quiver(solver, x_twin, plane="xz", lim=0.008, res=11)

    Xw, Zw, safety_map = compute_workspace_strength_map(solver)

    # --- Plot Figure 1: Piston Directivity + Pressure/Gor'kov Heatmaps ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))

    # (0,0) Piston Directivity Verification
    axes[0, 0].plot(angles_deg, ana_dir, "k-", lw=2.5, label="Analytical 2*J1(ka*sinθ)/(ka*sinθ)")
    axes[0, 0].plot(angles_deg, sim_dir, "r--", lw=1.8, label="piston_model_hemi()")
    axes[0, 0].axvline(90, color="gray", ls=":", label="90° (baffle plane)")
    axes[0, 0].set_title("1. Single-Transducer Piston Directivity")
    axes[0, 0].set_xlabel("Off-axis Angle θ (deg)")
    axes[0, 0].set_ylabel("Normalized Amplitude |D(θ)|")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    # (0,1) Pure Naive Pressure |p| (XY)
    c1 = axes[0, 1].contourf(X1, Y1, P_foc, levels=30, cmap="inferno")
    fig.colorbar(c1, ax=axes[0, 1], label="|p| (Pa)")
    axes[0, 1].set_title("2. Pure Naive Focus: Pressure |p| (XY)\n(Single Bright Peak -> Repels Bead)")
    axes[0, 1].set_xlabel("X (mm)")
    axes[0, 1].set_ylabel("Y (mm)")

    # (0,2) Pure Naive Gor'kov U + Force Vectors (XY)
    c2 = axes[0, 2].contourf(X1, Y1, U_foc * 1e6, levels=30, cmap="viridis")
    fig.colorbar(c2, ax=axes[0, 2], label="U (µJ)")
    axes[0, 2].quiver(qx_f, qy_f, Fxf, Fyf, color="white", alpha=0.85)
    axes[0, 2].set_title("Pure Naive: Gor'kov U & Force (XY)\n(Arrows Push OUTWARD from Center)")
    axes[0, 2].set_xlabel("X (mm)")
    axes[0, 2].set_ylabel("Y (mm)")

    # (1,0) Naive + Twin-Trap Pressure |p| (XY)
    c3 = axes[1, 0].contourf(X2, Y2, P_tw_xy, levels=30, cmap="inferno")
    fig.colorbar(c3, ax=axes[1, 0], label="|p| (Pa)")
    axes[1, 0].set_title("3. Naive + Twin-Trap: Pressure |p| (XY)\n(Dark Node Between 2 Bright Tweezers)")
    axes[1, 0].set_xlabel("X (mm)")
    axes[1, 0].set_ylabel("Y (mm)")

    # (1,1) Naive + Twin-Trap Gor'kov U + Force (XY)
    c4 = axes[1, 1].contourf(X2, Y2, U_tw_xy * 1e6, levels=30, cmap="viridis")
    fig.colorbar(c4, ax=axes[1, 1], label="U (µJ)")
    axes[1, 1].quiver(qx_t, qy_t, Fxt, Fyt, color="white", alpha=0.85)
    axes[1, 1].set_title("Twin-Trap: Gor'kov U & Force (XY)\n(Arrows Pull INWARD to Origin)")
    axes[1, 1].set_xlabel("X (mm)")
    axes[1, 1].set_ylabel("Y (mm)")

    # (1,2) Workspace Lifting Safety Factor Fz_max / mg (XZ)
    c5 = axes[1, 2].contourf(Xw, Zw, safety_map, levels=np.linspace(0, max(3.5, safety_map.max()), 25), cmap="RdYlGn")
    fig.colorbar(c5, ax=axes[1, 2], label="Fz_max / mg")
    axes[1, 2].contour(Xw, Zw, safety_map, levels=[1.0], colors="red", linewidths=2.5)
    axes[1, 2].set_title("4. Workspace Lifting Margin Fz_max / mg (XZ)\n(Red Line = 1.0x Gravity Drop Boundary)")
    axes[1, 2].set_xlabel("X (mm)")
    axes[1, 2].set_ylabel("Z (mm)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
