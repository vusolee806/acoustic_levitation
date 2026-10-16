import sys
import os
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from scripts.plot_utils import plot_simulation_results
import HemisAcoustic.assets.constants as const

# === AcousTools Imports ===
from acoustools.Utilities import create_points, device, DTYPE, propagate_abs, add_lev_sig
from acoustools.Solvers import naive
from acoustools.Force import compute_force
from acoustools.Visualiser import Visualise, ABC
import acoustools.Constants as c

def main():
    xml_path = os.path.join(PROJECT_ROOT, "assets", "mujoco_levitator.xml")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except ValueError as e:
        print(f"Error loading XML: {e}")
        return

    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    num_transducers = 79
    transducer_positions = np.zeros((num_transducers, 3))
    transducer_zaxis = np.zeros((num_transducers, 3))

    for i in range(num_transducers):
        site_id = model.site(f"site_sensor_{i+1}").id
        
        # 1. Use GLOBAL position from data
        transducer_positions[i] = data.site_xpos[site_id]

        # 2. Use GLOBAL rotation matrix from data
        mat = data.site_xmat[site_id].reshape(3, 3)
        transducer_zaxis[i] = mat[:, 2]  # The Z-axis vector

    ball_id = model.body("test_ball").id

    # ==========================================
    # 2. SETUP ACOUSTOOLS TENSORS
    # ==========================================
    
    # Convert numpy arrays to PyTorch tensors and explicitly cast to AcousTools' DTYPE (complex64)
    board_pos = torch.tensor(transducer_positions, device=device).to(DTYPE)
    board_norms = torch.tensor(transducer_zaxis, device=device).to(DTYPE)
    
    # Map physical constants (fallback to AcousTools defaults if not in const)
    radius = getattr(const, 'TRANSDUCER_SIZE', c.radius)
    rho_1 = getattr(const, 'RHO_1', c.p_p)
    c_1 = getattr(const, 'C_1', c.c_p)
    particle_vol = (4/3) * np.pi * (getattr(const, 'PARTICLE_RADIUS', c.R))**3

    target_focal_point = [0.0, 0.0, 0.04]
    
    # AcousTools points require shape [Batch, 3, N]. create_points builds this natively.
    p = create_points(N=1, B=1, x=target_focal_point[0], y=target_focal_point[1], z=target_focal_point[2])

    # 1. Compute focal phases using the naive backpropagation solver
    hologram = naive(p, board=board_pos, norms=board_norms, transducer_radius=radius)
    
    # 2. Add the Twin Trap phase signature (Explicitly pass board_size for custom arrays)
    hologram = add_lev_sig(hologram, board=board_pos, mode='Twin', board_size=num_transducers)

    # 3. Apply the amplitude weight
    complex_weights = 0.4 * hologram

    # ==========================================
    # 3. VISUALIZATION
    # ==========================================
    print("Rendering Acoustic Field...")
    
    # Define a 4cm XZ viewing plane centered on the trap
    A, B, C = ABC(size=0.04, plane='xz', origin=target_focal_point)
    
    # Render non-blockingly so MuJoCo can launch
    Visualise(A, B, C, complex_weights, 
              colour_functions=[propagate_abs],
              colour_function_args=[{'board': board_pos, 'norms': board_norms, 'transducer_radius': radius}],
              res=(150, 150), show=True, block=False, 
              titles=['Twin Trap Pressure (XZ Plane)'])

    # --- LOGGING SETUP ---
    time_log = []
    fx_log, fy_log, fz_log = [], [], []
    x_pos_log, y_pos_log, z_pos_log = [], [], []

    print("Đang chạy mô phỏng... Tắt cửa sổ MuJoCo để xem đồ thị!")

    render_interval = 15  
    step_counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_counter += 1
            
            # ==========================================
            # 4. EVERY STEP: PHYSICS & LOGGING 
            # ==========================================
            data.qfrc_applied[:] = 0.0 
            ball_pos = data.xpos[ball_id]
            
            # Convert particle position to AcousTools [1, 3, 1] tensor shape
            ball_pos_torch = torch.tensor(ball_pos, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)
            
            # Evaluate Acoustic Radiation Force
            forces_tensor = compute_force(
                activations=complex_weights,
                points=ball_pos_torch,
                board=board_pos,
                transducer_norms=board_norms,
                transducer_radius=radius,
                particle_density=rho_1,
                particle_speed=c_1,
                V=particle_vol
            )
            
            # Extract the [Fx, Fy, Fz] force vector and format for MuJoCo
            arf_force = forces_tensor[0, 0, :].cpu().numpy()
            
            torque = np.array([0.0, 0.0, 0.0]) 
            mujoco.mj_applyFT(model, data, arf_force, torque, ball_pos, ball_id, data.qfrc_applied)
            
            # Log data
            time_log.append(data.time)
            fx_log.append(arf_force[0])
            fy_log.append(arf_force[1])
            fz_log.append(arf_force[2])
            x_pos_log.append(ball_pos[0])  
            y_pos_log.append(ball_pos[1])  
            z_pos_log.append(ball_pos[2])

            mujoco.mj_step(model, data)
            
            # ==========================================
            # 5. EVERY N STEPS: VISUALIZATION
            # ==========================================
            if step_counter % render_interval == 0:
                visual_scale = 1000.0  
                force_end = ball_pos + (arf_force * visual_scale)
                
                with viewer.lock():
                    viewer.user_scn.ngeom = num_transducers + 1
                    
                    for i in range(num_transducers):
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[i],
                            type=mujoco.mjtGeom.mjGEOM_LINE,
                            size=np.array([0.0005, 0, 0]),
                            pos=np.zeros(3),
                            mat=np.eye(3).flatten(),
                            rgba=np.array([0.2, 0.5, 1.0, 0.3])
                        )
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[i], 
                            mujoco.mjtGeom.mjGEOM_LINE, 
                            0.0005, 
                            transducer_positions[i],
                            target_focal_point   
                        )
                    
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[num_transducers],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=np.array([0.001, 0, 0]),  
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1.0, 0.0, 0.0, 1.0])
                    )
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[num_transducers], 
                        mujoco.mjtGeom.mjGEOM_CAPSULE, 
                        0.001,  
                        ball_pos,      
                        force_end      
                    )
                
                viewer.sync()

    # ==========================================
    # 6. PLOT RESULTS
    # ==========================================
    print("Đã tắt mô phỏng. Đang tạo đồ thị...")
    
    plot_simulation_results(
        time_log, 
        fx_log, fy_log, fz_log, 
        x_pos_log, y_pos_log, z_pos_log
    )

if __name__ == "__main__":
    main()
