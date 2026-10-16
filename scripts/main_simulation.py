import os
import sys
import torch
import mujoco
import mujoco.viewer
import numpy as np

# Adjust imports based on your project structure
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import HemisAcoustic.assets.constants as const #[cite: 65]
from scripts.plot_utils import plot_simulation_results

# AcousTools Imports
from acoustools.Utilities import create_points, device, DTYPE, add_lev_sig
from acoustools.Solvers import naive
from acoustools.Force import compute_force

def main():
    # ==========================================
    # 1. INITIALIZE MUJOCO & HARDWARE GEOMETRY
    # ==========================================
    model = mujoco.MjModel.from_xml_path(const.XML_PATH) #[cite: 65]
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    num_transducers = 79
    transducer_positions = np.zeros((num_transducers, 3))
    transducer_zaxis = np.zeros((num_transducers, 3))

    for i in range(num_transducers):
        site_id = model.site(f"site_sensor_{i+1}").id
        transducer_positions[i] = data.site_xpos[site_id]
        mat = data.site_xmat[site_id].reshape(3, 3)
        transducer_zaxis[i] = mat[:, 2] 

    ball_id = model.body("test_ball").id

    # ==========================================
    # 2. CONFIGURE ACOUSTOOLS TENSORS
    # ==========================================
    # Keep positions as real numbers (float32) for signature geometry calculations
    board_pos = torch.tensor(transducer_positions, dtype=torch.float32, device=device)
    
    # Cast only the normal vectors to AcousTools' internal complex datatype
    board_norms = torch.tensor(transducer_zaxis, device=device).to(DTYPE)

    # Set the trap height to exactly match the spawn height of test_ball in the XML
    target_focal_point = [0.0, 0.0, 0.04]
    p = create_points(N=1, B=1, x=target_focal_point[0], y=target_focal_point[1], z=target_focal_point[2])

    # ==========================================
    # 3. COMPUTE THE HOLOGRAM
    # ==========================================
    # Calculate focal phases using the naive solver
    hologram = naive(p, board=board_pos, norms=board_norms, transducer_radius=const.TRANSDUCER_RADIUS) #[cite: 65]
    
    # Apply a Vortex signature for inward-focusing bowls to create a stable central trap
    hologram = add_lev_sig(hologram, board=board_pos, mode='Vortex', board_size=num_transducers)

    # Apply the global amplitude from constants.py. 
    # Note: Divided by 10 to ensure MuJoCo stability (prevents the particle from exploding out of the trap).
    safe_multiplier = const.COMPLEX_WEIGHT_MULTIPLIER / 10 #[cite: 65]
    complex_weights = safe_multiplier * hologram 

    # ==========================================
    # 4. RUN MUJOCO PHYSICS LOOP
    # ==========================================
    time_log, fx_log, fy_log, fz_log = [], [], [], []
    x_pos_log, y_pos_log, z_pos_log = [], [], []

    print("Running simulation... Close the MuJoCo window to view plots.")

    render_interval = 15  
    step_counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_counter += 1
            
            data.qfrc_applied[:] = 0.0 
            ball_pos = data.xpos[ball_id]
            
            # Format particle position for AcousTools [Batch, 3, Points]
            ball_pos_torch = torch.tensor(ball_pos, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)
            
            # Compute Acoustic Radiation Force using mapped constants
            forces_tensor = compute_force(
                activations=complex_weights,
                points=ball_pos_torch,
                board=board_pos,
                transducer_norms=board_norms,
                transducer_radius=const.TRANSDUCER_RADIUS, #[cite: 65]
                medium_density=const.RHO_0,                #[cite: 65]
                medium_speed=const.C_0,                    #[cite: 65]
                particle_density=const.RHO_1,              #[cite: 65]
                particle_speed=const.C_1,                  #[cite: 65]
                V=const.PARTICLE_VOL                       #[cite: 65]
            )
            
            # Extract [Fx, Fy, Fz]
            arf_force = forces_tensor[0, 0, :].cpu().numpy()
            
            # Apply force to the particle in MuJoCo
            mujoco.mj_applyFT(model, data, arf_force, np.zeros(3), ball_pos, ball_id, data.qfrc_applied)
            
            # Logging
            time_log.append(data.time)
            fx_log.append(arf_force[0])
            fy_log.append(arf_force[1])
            fz_log.append(arf_force[2])
            x_pos_log.append(ball_pos[0])  
            y_pos_log.append(ball_pos[1])  
            z_pos_log.append(ball_pos[2])

            mujoco.mj_step(model, data)
            
            # Rendering (Rays and Force Vectors)
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
    # 5. GENERATE PLOTS
    # ==========================================
    plot_simulation_results(time_log, fx_log, fy_log, fz_log, x_pos_log, y_pos_log, z_pos_log)

if __name__ == "__main__":
    main()
