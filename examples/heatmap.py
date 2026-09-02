import os
import sys
import numpy as np
import torch
import mujoco
import matplotlib.pyplot as plt

# Set up project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.acoustic_physics import calculate_complex_pressure
from src.acoustic_physics_pytorch import generate_twin_trap_phases


def load_array_geometry(xml_filename="mujoco_levitator.xml", num_transducers=79):
    """Loads the MuJoCo model and extracts transducer positions and Z-axis vectors."""
    xml_path = os.path.join(PROJECT_ROOT, "assets", xml_filename)
    
    # Fallback if levitator file is not found
    if not os.path.exists(xml_path):
        xml_path = os.path.join(PROJECT_ROOT, "assets", "mujoco_levitator.xml")
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    transducer_positions = np.zeros((num_transducers, 3))
    transducer_zaxis = np.zeros((num_transducers, 3))
    
    for i in range(num_transducers):
        site_id = model.site(f"site_sensor_{i+1}").id
        transducer_positions[i] = model.site_pos[site_id]
        
        quat = model.site_quat[site_id]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        transducer_zaxis[i] = mat.reshape(3, 3)[:, 2]
        
    return transducer_positions, transducer_zaxis


def compute_pressure_slice(points_grid, trans_pos, trans_zaxis, phases):
    """Calculates absolute pressure amplitude for a 2D meshgrid of 3D points."""
    h, w, _ = points_grid.shape
    p_mag = np.zeros((h, w))
    
    for i in range(h):
        for j in range(w):
            point = points_grid[i, j]
            p_complex = calculate_complex_pressure(point, trans_pos, trans_zaxis, phases)
            p_mag[i, j] = np.abs(p_complex)
            
    return p_mag


def plot_all_pressure_planes(trans_pos, trans_zaxis, phases, target_focal_point, span=0.030, resolution=0.0005):
    """Generates and plots the XY, XZ, and YZ pressure planes around the target point."""
    cx, cy, cz = target_focal_point
    vec = np.arange(-span, span, resolution)
    n_pts = len(vec)
    
    print(f"Calculating pressure grid ({n_pts}x{n_pts}) across all 3 planes...")
    
    # 1. XY Plane (Fixed Z = cz)
    X_xy, Y_xy = np.meshgrid(vec + cx, vec + cy)
    grid_xy = np.stack([X_xy, Y_xy, np.full_like(X_xy, cz)], axis=-1)
    P_XY = compute_pressure_slice(grid_xy, trans_pos, trans_zaxis, phases)
    
    # 2. XZ Plane (Fixed Y = cy)
    X_xz, Z_xz = np.meshgrid(vec + cx, vec + cz)
    grid_xz = np.stack([X_xz, np.full_like(X_xz, cy), Z_xz], axis=-1)
    P_XZ = compute_pressure_slice(grid_xz, trans_pos, trans_zaxis, phases)
    
    # 3. YZ Plane (Fixed X = cx)
    Y_yz, Z_yz = np.meshgrid(vec + cy, vec + cz)
    grid_yz = np.stack([np.full_like(Y_yz, cx), Y_yz, Z_yz], axis=-1)
    P_YZ = compute_pressure_slice(grid_yz, trans_pos, trans_zaxis, phases)
    
    print("Computation complete. Generating figure...")
    
    # Setup Figure and Subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
    
    span_mm = span * 1000
    extent_xy = [(cx - span) * 1000, (cx + span) * 1000, (cy - span) * 1000, (cy + span) * 1000]
    extent_xz = [(cx - span) * 1000, (cx + span) * 1000, (cz - span) * 1000, (cz + span) * 1000]
    extent_yz = [(cy - span) * 1000, (cy + span) * 1000, (cz - span) * 1000, (cz + span) * 1000]
    
    # Subplot 1: XY Plane
    im0 = axs[0].imshow(P_XY, extent=extent_xy, origin='lower', cmap='hot', aspect='equal')
    axs[0].plot(cx * 1000, cy * 1000, 'c+', markersize=10, markeredgewidth=2, label='Target Node')
    axs[0].set_title(f'XY Plane (Z = {cz*1000:.1f} mm)')
    axs[0].set_xlabel('X (mm)')
    axs[0].set_ylabel('Y (mm)')
    axs[0].legend(loc='upper right')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label='Pressure Magnitude')

    # Subplot 2: XZ Plane
    im1 = axs[1].imshow(P_XZ, extent=extent_xz, origin='lower', cmap='hot', aspect='equal')
    axs[1].plot(cx * 1000, cz * 1000, 'c+', markersize=10, markeredgewidth=2)
    axs[1].set_title(f'XZ Plane (Y = {cy*1000:.1f} mm)')
    axs[1].set_xlabel('X (mm)')
    axs[1].set_ylabel('Z (mm)')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label='Pressure Magnitude')

    # Subplot 3: YZ Plane
    im2 = axs[2].imshow(P_YZ, extent=extent_yz, origin='lower', cmap='hot', aspect='equal')
    axs[2].plot(cy * 1000, cz * 1000, 'c+', markersize=10, markeredgewidth=2)
    axs[2].set_title(f'YZ Plane (X = {cx*1000:.1f} mm)')
    axs[2].set_xlabel('Y (mm)')
    axs[2].set_ylabel('Z (mm)')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, label='Pressure Magnitude')

    plt.suptitle(f'Acoustic Field Visualization (Focal Center: [{cx*1000:.1f}, {cy*1000:.1f}, {cz*1000:.1f}] mm)', fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    # 1. Define Focal Target Coordinate (in meters)
    focal_target = np.array([0.0, 0.0, 0.04]) 
    
    # 2. Extract Geometry from MuJoCo XML
    trans_pos, trans_zaxis = load_array_geometry()
    
    # 3. Calculate Phases via PyTorch Twin Trap Generator
    trans_pos_tensor = torch.tensor(trans_pos, dtype=torch.float32)
    target_tensor = torch.tensor([focal_target], dtype=torch.float32)
    
    phases_tensor = generate_twin_trap_phases(target_tensor, trans_pos_tensor)
    phases = phases_tensor.detach().cpu().numpy().squeeze()
    
    
    # 4. Render Heatmap Across All Planes
    plot_all_pressure_planes(
        trans_pos=trans_pos,
        trans_zaxis=trans_zaxis,
        phases=phases,
        target_focal_point=focal_target,
        span=0.030,          # Field span: 30 mm in each direction
        resolution=0.0005    # Resolution: 0.5 mm grid step
    )


if __name__ == "__main__":
    main()
