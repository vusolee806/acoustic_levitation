import sys
import os
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import levitate

# ==========================================
# 1. ACOUSTIC & MATERIAL CONSTANTS
# ==========================================
FREQ = 40000.0         # Operating frequency (Hz)
C_0 = 343.0            # Speed of sound in air (m/s)
RHO_0 = 1.225          # Density of air (kg/m^3)
WAVELENGTH = C_0 / FREQ 

C_1 = 1200.0           # Speed of sound in EPS foam (m/s)
RHO_1 = 40.0           # Density of EPS foam (kg/m^3)
PARTICLE_RADIUS = 0.002 # 2mm

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
    levitate.frequency = FREQ

    # Define only the levitated particle's material. Levitate defaults to air natively.
    ball_material = levitate.materials.Material(rho=RHO_1, c=C_1)
    
    # Transpose to (3, N) shape for Levitate
    #define your array and change it from TransducerArray to NormalTransducerArray
    #Because NormalTransducerArray overrides the .signature() method, it will now correctly recognize stype='twin'
    # and automatically calculate the geometric splitting without requiring you to pass phases
    
    array = levitate.arrays.NormalTransducerArray(
        positions=transducer_positions.T, 
        normals=transducer_zaxis.T,
        transducer_size=0.02  # Set to the 20mm diameter from your XML
    )
    
    # Remove the 'medium=air' argument entirely
    force_evaluator = levitate.fields.RadiationForce(
        array, radius=PARTICLE_RADIUS, material=ball_material
    )

    # --- LƯU TRỮ DỮ LIỆU ---
    time_log = []
    fx_log, fy_log, fz_log = [], [], []
    z_pos_log = []

    print("Đang chạy mô phỏng... Tắt cửa sổ MuJoCo để xem đồ thị!")

    # Apply the frequency directly to the array
    array.freq = FREQ

    # Define the target
    target_focal_point = np.array([0.0, 0.0, 0.04])

    # 1. Use Levitate to get focus phases
    focus_phases = array.focus_phases(target_focal_point)

    # 2. Use Levitate to apply the Twin Trap signature 
    twin_signature = array.signature(position=target_focal_point, stype='twin')

    # 3. Combine and calculate complex weights exactly ONCE
    array.phases = focus_phases + twin_signature
    complex_weights = np.exp(1j * array.phases)

    #visulize by library levitate
    # Create a visualizer tied to your array
    viz = array.visualize

    # Add a 3D slice showing the sound pressure field (SPL)
    viz.append(levitate.visualizers.PressureSlice(array))

    # Render the interactive Plotly graph using your complex weights
    fig = viz(complex_weights)
    fig.show()



    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            ball_pos = data.xpos[ball_id]
            
            # 1. Evaluate the acoustic force
            forces_array = force_evaluator(complex_weights, ball_pos)
            
            # 2. MISSING STEP: Extract and optionally clip the force vector
            # (If ball_pos is shape (3,), forces_array is also (3,))
            arf_force = np.clip(forces_array, -1.0, 1.0)
            
            # Apply to MuJoCo
            dof_start = model.body_dofadr[ball_id]
            data.qfrc_applied[dof_start : dof_start + 3] = arf_force
            
            # --- LƯU DỮ LIỆU ---
            time_log.append(data.time)
            fx_log.append(arf_force[0])
            fy_log.append(arf_force[1])
            fz_log.append(arf_force[2])
            z_pos_log.append(ball_pos[2])
            
            mujoco.mj_step(model, data)
            viewer.sync()

    # ==========================================
    # 3. VẼ ĐỒ THỊ
    # ==========================================
    print("Đã tắt mô phỏng. Đang tạo đồ thị...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time_log, fz_log, label='F_z (Vertical Force)', color='blue')
    ax1.plot(time_log, fx_log, label='F_x (Lateral Force)', color='red', alpha=0.6)
    ax1.plot(time_log, fy_log, label='F_y (Lateral Force)', color='green', alpha=0.6)
    ax1.set_ylabel("Force (Newtons)")
    ax1.set_title("Acoustic Radiation Force over Time (Levitate Engine)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_log, z_pos_log, label='Z Position (Height)', color='purple')
    ax2.set_xlabel("Simulation Time (Seconds)")
    ax2.set_ylabel("Height (Meters)")
    ax2.set_title("Particle Trajectory (Settling into Trap)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
