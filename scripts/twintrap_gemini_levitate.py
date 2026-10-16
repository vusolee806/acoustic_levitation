import sys
import os
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import levitate
from scripts.plot_utils import plot_simulation_results
import HemisAcoustic.assets.constants as const
# # ==========================================
# # 1. ACOUSTIC & MATERIAL CONSTANTS
# # ==========================================
# FREQ = 40000.0         # Operating frequency (Hz)
# C_0 = 343.0            # Speed of sound in air (m/s)
# RHO_0 = 1.225          # Density of air (kg/m^3)
# WAVELENGTH = C_0 / FREQ 

# C_1 = 1200.0           # Speed of sound in EPS foam (m/s)
# RHO_1 = 40.0           # Density of EPS foam (kg/m^3)
# PARTICLE_RADIUS = 0.003 # 2mm

def main():
    xml_path = os.path.join(PROJECT_ROOT, "assets", "mujoco_levitator.xml")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except ValueError as e:
        print(f"Error loading XML: {e}")
        return

    data = mujoco.MjData(model)

    # Initialize global coordinates before reading them
    mujoco.mj_kinematics(model, data)

    num_transducers = 79
    transducer_positions = np.zeros((num_transducers, 3))
    transducer_zaxis = np.zeros((num_transducers, 3))

    for i in range(num_transducers):
        site_id = model.site(f"site_sensor_{i+1}").id
        
        # 1. Use GLOBAL position from data, not local from model
        transducer_positions[i] = data.site_xpos[site_id]

        # 2. Use GLOBAL rotation matrix from data
        mat = data.site_xmat[site_id].reshape(3, 3)
        transducer_zaxis[i] = mat[:, 2]  # The Z-axis vector

    ball_id = model.body("test_ball").id

    # ==========================================
    # 2. SETUP LEVITATE ARRAY AND MATERIALS
    # ==========================================
    levitate.frequency = const.FREQ
    ball_material = levitate.materials.Material(rho=const.RHO_1, c=const.C_1)
    
    array = levitate.arrays.NormalTransducerArray(
        positions=transducer_positions.T, 
        normals=transducer_zaxis.T,
        transducer_size=const.TRANSDUCER_SIZE  
    )
    
    force_evaluator = levitate.fields.RadiationForce(
        array, radius=const.PARTICLE_RADIUS, material=ball_material
    )
    # --- LƯU TRỮ DỮ LIỆU ---
    time_log = []
    fx_log, fy_log, fz_log = [], [], []
    x_pos_log, y_pos_log, z_pos_log = [], [], []  # Added X and Y

    print("Đang chạy mô phỏng... Tắt cửa sổ MuJoCo để xem đồ thị!")

    # Apply the frequency directly to the array
    array.freq = const.FREQ

    # Define the target
    target_focal_point = np.array([0.0, 0.0, 0.03])

    # 1. Use Levitate to get focus phases
    focus_phases = array.focus_phases(target_focal_point)

    # 2. Use Levitate to apply the Twin Trap signature 
    twin_signature = array.signature(position=target_focal_point, stype='twin')

    # 3. Combine and calculate complex weights exactly ONCE
    array.phases = focus_phases + twin_signature
    complex_weights = 0.4 * np.exp(1j * array.phases)

    #visulize by library levitate
    # Create a visualizer tied to your array
    viz = array.visualize

    # 1. XZ Plane (Front View) - Normal points along Y
    viz.append(levitate.visualizers.PressureSlice(
        array, normal=(0, 1, 0), intersect=target_focal_point
    ))

    # 2. YZ Plane (Side View) - Normal points along X
    viz.append(levitate.visualizers.PressureSlice(
        array, normal=(1, 0, 0), intersect=target_focal_point
    ))

    # 3. XY Plane (Top-Down View) - Normal points along Z
    viz.append(levitate.visualizers.PressureSlice(
        array, normal=(0, 0, 1), intersect=target_focal_point
    ))

    # Render the interactive Plotly graph using your complex weights
    fig = viz(complex_weights)
    fig.show()

    # Set how often to update the screen (e.g., every 15 physics steps)
    render_interval = 15  
    step_counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_counter += 1
            
            # ==========================================
            # 1. EVERY STEP: PHYSICS & LOGGING 
            # (This must run as fast as possible)
            # ==========================================
            data.qfrc_applied[:] = 0.0 
            ball_pos = data.xpos[ball_id]
            
            forces_array = force_evaluator(complex_weights, ball_pos)
            arf_force = forces_array
            
            torque = np.array([0.0, 0.0, 0.0]) 
            mujoco.mj_applyFT(model, data, arf_force, torque, ball_pos, ball_id, data.qfrc_applied)
            
            # Log data for your plot
            time_log.append(data.time)
            fx_log.append(arf_force[0])
            fy_log.append(arf_force[1])
            fz_log.append(arf_force[2])
            x_pos_log.append(ball_pos[0])  
            y_pos_log.append(ball_pos[1])  
            z_pos_log.append(ball_pos[2])

            # Advance the physics engine
            mujoco.mj_step(model, data)
            
            # ==========================================
            # 2. EVERY N STEPS: VISUALIZATION
            # ==========================================
            if step_counter % render_interval == 0:
                visual_scale = 1000.0  
                force_end = ball_pos + (arf_force * visual_scale)
                
                # # The exact center of the acoustic trap
                # target_focal_point = np.array([0.0, 0.0, 0.04])
                
                with viewer.lock():
                    viewer.user_scn.ngeom = num_transducers + 1
                    
                    # --- A. Draw the 79 blue rays pointing to the TRAP ---
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
                            target_focal_point   # Correctly mapped to the trap
                        )
                    
                    # --- B. Draw the 1 red force vector pushing the BALL ---
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
                        ball_pos,      # Correctly mapped to start at the ball
                        force_end      # Correctly mapped to end at the scaled force
                    )
                
                # Sync the viewer ONLY when we redraw the shapes
                viewer.sync()

    # ==========================================
    # 3. VẼ ĐỒ THỊ
    # ==========================================
    print("Đã tắt mô phỏng. Đang tạo đồ thị...")
    
    plot_simulation_results(
        time_log, 
        fx_log, fy_log, fz_log, 
        x_pos_log, y_pos_log, z_pos_log
    )

if __name__ == "__main__":
    main()
