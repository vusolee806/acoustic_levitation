# scripts/check_transducer_vectors.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mujoco

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from HemisAcoustic.physics.acoustic_force_hemi import HemisphereAcousticSolver


def extract_old_body_quat(mj_model):
    """Exact extraction logic from original acoustic_force_hemi.py (body_pos + body_quat)."""
    positions, normals = [], []
    for i in range(1, 80):
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"sensor_{i}")
        pos = mj_model.body_pos[body_id].copy()
        quat = mj_model.body_quat[body_id]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        zaxis = mat.reshape(3, 3)[:, 2].copy()
        positions.append(pos)
        normals.append(zaxis)
    return np.array(positions), np.array(normals)


def extract_new_site_xmat(mj_model):
    """Modified extraction logic (site_xpos + site_xmat from site_sensor_i)."""
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    positions, normals = [], []
    for i in range(1, 80):
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f"site_sensor_{i}")
        pos = mj_data.site_xpos[site_id].copy()
        zaxis = mj_data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()
        positions.append(pos)
        normals.append(zaxis)
    return np.array(positions), np.array(normals)


def angle_to_origin_deg(positions, normals):
    """Angle (deg) between each transducer normal and the ideal inward vector pointing to (0,0,0)."""
    ideal_inward = -positions / (np.linalg.norm(positions, axis=1, keepdims=True) + 1e-12)
    unit_normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    dots = np.clip(np.sum(unit_normals * ideal_inward, axis=1), -1.0, 1.0)
    return np.rad2deg(np.arccos(dots))


def main():
    xml_path = os.path.join(PROJECT_ROOT, "HemisAcoustic", "assets", "mujoco_levitator.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    old_pos, old_norm = extract_old_body_quat(mj_model)
    new_pos, new_norm = extract_new_site_xmat(mj_model)

    # Also inspect whatever your CURRENT acoustic_force_hemi.py actually loads right now
    solver = HemisphereAcousticSolver(mj_model)
    live_pos = solver.trans_pos.detach().cpu().numpy()
    live_norm = solver.trans_norm.detach().cpu().numpy()

    old_err = angle_to_origin_deg(old_pos, old_norm)
    new_err = angle_to_origin_deg(new_pos, new_norm)
    live_err = angle_to_origin_deg(live_pos, live_norm)

    # 1. Print full 79-transducer table (in meters to match mujoco_levitator.xml directly)
    header = (
        f"{'ID':>3} | {'Position (x, y, z) [m]':^29} | "
        f"{'OLD Normal (body_quat)':^26} | {'Err°':>5} | "
        f"{'NEW Normal (site_xmat)':^26} | {'Err°':>5}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i in range(79):
        p = new_pos[i]
        no = old_norm[i]
        nn = new_norm[i]
        print(
            f"{i+1:3d} | ({p[0]:+8.5f}, {p[1]:+8.5f}, {p[2]:+8.5f}) | "
            f"({no[0]:+7.4f}, {no[1]:+7.4f}, {no[2]:+7.4f}) | {old_err[i]:5.2f} | "
            f"({nn[0]:+7.4f}, {nn[1]:+7.4f}, {nn[2]:+7.4f}) | {new_err[i]:5.2f}"
        )
    print("=" * len(header))

    # 2. Print summary & check active solver state
    print("\n=== SUMMARY ===")
    print(f"Position max diff (OLD vs NEW) : {np.max(np.abs(old_pos - new_pos)):.2e} m")
    print(f"OLD (body_quat) angle to (0,0,0): min={old_err.min():5.2f}°, mean={old_err.mean():5.2f}°, max={old_err.max():5.2f}°")
    print(f"NEW (site_xmat) angle to (0,0,0): min={new_err.min():5.2f}°, mean={new_err.mean():5.2f}°, max={new_err.max():5.2f}°")
    print(f"LIVE solver     angle to (0,0,0): min={live_err.min():5.2f}°, mean={live_err.mean():5.2f}°, max={live_err.max():5.2f}°")

    diff_live_old = np.max(np.abs(live_norm - old_norm))
    diff_live_new = np.max(np.abs(live_norm - new_norm))
    if diff_live_old < 1e-5:
        print("-> Your live HemisphereAcousticSolver is currently using the OLD (body_quat) normals.")
    elif diff_live_new < 1e-5:
        print("-> Your live HemisphereAcousticSolver is currently using the NEW (site_xmat) normals.")
    else:
        print(f"-> Live solver differs from both (max diff vs old={diff_live_old:.4f}, vs new={diff_live_new:.4f}).")

    # 3. Side-by-side 3D quiver visualization
    fig = plt.figure(figsize=(14, 6.5))
    panels = [
        ("OLD Code: body_quat Normals", old_pos * 1000.0, old_norm, "red"),
        ("MODIFIED Code: site_xmat Normals", new_pos * 1000.0, new_norm, "dodgerblue"),
    ]

    for idx, (title, pos_mm, norm, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.scatter(pos_mm[:, 0], pos_mm[:, 1], pos_mm[:, 2], c="black", s=18, label="79 Transducers")
        ax.quiver(
            pos_mm[:, 0], pos_mm[:, 1], pos_mm[:, 2],
            norm[:, 0], norm[:, 1], norm[:, 2],
            length=18.0, normalize=True, color=color, arrow_length_ratio=0.25, linewidth=1.2,
            label="Emission Normal (+Z)"
        )
        ax.scatter([0], [0], [0], c="lime", edgecolors="black", s=90, marker="*", label="Focal Origin (0,0,0)")
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_xlim(-95, 95)
        ax.set_ylim(-95, 95)
        ax.set_zlim(-95, 45)
        ax.set_box_aspect([190, 190, 140])
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
