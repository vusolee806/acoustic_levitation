import sys
import os
import numpy as np
import mujoco


# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------
# This gets the absolute path of the directory containing this script, 
# then goes up one level to the main project folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to sys.path so Python can import your physics file
sys.path.append(PROJECT_ROOT)

# IMPORT YOUR PHYSICS FUNCTIONS HERE
from src.acoustic_physics import calculate_arf 

# ---------------------------------------------------------
# Simulation Loop
# ---------------------------------------------------------
def main():
    # Build the absolute path to your XML file in the 'assets' folder
    xml_path = os.path.join(PROJECT_ROOT, "assets", "mujoco_array.xml")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except ValueError as e:
        print(f"Error loading XML: {e}")
        print(f"Make sure 'mujoco_array.xml' is located at: {xml_path}")
        return

    data = mujoco.MjData(model)

    # Extract static transducer data once
    # NOTE: If you changed your XML to match the 30-sensor AcoMan paper, change 79 to 30!
    num_transducers = 79
    transducer_positions = np.zeros((num_transducers, 3))
    transducer_zaxis = np.zeros((num_transducers, 3))
    
    # Split the phase distribution to create the twin trap!
    phases = np.zeros(num_transducers)
    phases[:num_transducers//2] = np.pi 

    for i in range(num_transducers):
        site_id = model.site(f"site_sensor_{i+1}").id
        transducer_positions[i] = model.site_pos[site_id]
        
        # Extract the Z-axis from the orientation quaternion
        quat = model.site_quat[site_id]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        transducer_zaxis[i] = mat.reshape(3, 3)[:, 2]

    ball_id = model.body("test_ball").id

    # Launch the official passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 1. Get the current position of the ball
            ball_pos = data.xpos[ball_id]
            
            # 2. Calculate the exact Acoustic Radiation Force
            force_vector = calculate_arf(ball_pos, transducer_positions, transducer_zaxis, phases)
            
            # 3. Apply the force to the ball
            data.xfrc_applied[ball_id] = 0.0 # Reset
            data.xfrc_applied[ball_id, :3] = force_vector
            
            # 4. Step physics and sync viewer
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
