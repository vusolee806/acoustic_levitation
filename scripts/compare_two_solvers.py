# scripts/compare_two_solvers.py

import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco

import HemisAcoustic
from HemisAcoustic.physics import acoustic_force_hemi as old_mod
from HemisAcoustic.physics import acoustic_temp as new_mod


def angle_to_origin_deg(positions, normals):
    """Angle (deg) between each transducer normal and the inward vector to (0, 0, 0)."""
    ideal_inward = -positions / (np.linalg.norm(positions, axis=1, keepdims=True) + 1e-12)
    unit_normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    dots = np.clip(np.sum(unit_normals * ideal_inward, axis=1), -1.0, 1.0)
    return np.rad2deg(np.arccos(dots))


def main():
    # Locate XML directly through the editable-installed HemisAcoustic package
    pkg_dir = os.path.dirname(HemisAcoustic.__file__)
    xml_path = os.path.join(pkg_dir, "assets", "mujoco_levitator.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # Instantiate directly from your two actual files
    old_solver = old_mod.HemisphereAcousticSolver(mj_model)
    new_solver = new_mod.HemisphereAcousticSolver(mj_model)

    print("=== LOADED FILES ===")
    print(f"  OLD file : {old_mod.__file__}  (default p0={old_solver.p0})")
    print(f"  NEW file : {new_mod.__file__}  (default p0={new_solver.p0})")
    print(f"  XML file : {xml_path}\n")

    # Pull tensors directly from each solver instance
    old_pos = old_solver.trans_pos.detach().cpu().numpy()
    old_norm = old_solver.trans_norm.detach().cpu().numpy()

    new_pos = new_solver.trans_pos.detach().cpu().numpy()
    new_norm = new_solver.trans_norm.detach().cpu().numpy()

    old_err = angle_to_origin_deg(old_pos, old_norm)
    new_err = angle_to_origin_deg(new_pos, new_norm)

    # Print all 79 transducer coordinates and normals from both files
    header = (
        f"{'ID':>3} | {'Position (x, y, z) [m]':^29} | "
        f"{'acoustic_force_hemi.py':^26} | {'Err°':>5} | "
        f"{'acoustic_temp.py':^26} | {'Err°':>5}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i in range(len(old_pos)):
        p = old_pos[i]
        no = old_norm[i]
        nn = new_norm[i]
        print(
            f"{i+1:3d} | ({p[0]:+8.5f}, {p[1]:+8.5f}, {p[2]:+8.5f}) | "
            f"({no[0]:+7.4f}, {no[1]:+7.4f}, {no[2]:+7.4f}) | {old_err[i]:5.2f} | "
            f"({nn[0]:+7.4f}, {nn[1]:+7.4f}, {nn[2]:+7.4f}) | {new_err[i]:5.2f}"
        )
    print("=" * len(header))

    # Numerical summary
    print("\n=== DIRECT COMPARISON SUMMARY ===")
    print(f"  Max position diff (old vs temp)  : {np.max(np.abs(old_pos - new_pos)):.2e} m")
    print(f"  Max normal diff   (old vs temp)  : {np.max(np.abs(old_norm - new_norm)):.4f}")
    print(f"  acoustic_force_hemi angle err    : min={old_err.min():5.2f}°, mean={old_err.mean():5.2f}°, max={old_err.max():5.2f}°")
    print(f"  acoustic_temp       angle err    : min={new_err.min():5.2f}°, mean={new_err.mean():5.2f}°, max={new_err.max():5.2f}°")

    # Side-by-side 3D plot directly from the two solver instances
    fig = plt.figure(figsize=(14, 6.5))
    panels = [
        ("acoustic_force_hemi.py (trans_norm)", old_pos * 1000.0, old_norm, "red"),
        ("acoustic_temp.py (trans_norm)", new_pos * 1000.0, new_norm, "dodgerblue"),
    ]

    for idx, (title, pos_mm, norm, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.scatter(pos_mm[:, 0], pos_mm[:, 1], pos_mm[:, 2], c="black", s=18, label="79 Transducers")
        ax.quiver(
            pos_mm[:, 0], pos_mm[:, 1], pos_mm[:, 2],
            norm[:, 0], norm[:, 1], norm[:, 2],
            length=18.0, normalize=True, color=color, arrow_length_ratio=0.25, linewidth=1.2,
            label="solver.trans_norm"
        )
        ax.scatter([0], [0], [0], c="lime", edgecolors="black", s=90, marker="*", label="Origin (0,0,0)")
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
