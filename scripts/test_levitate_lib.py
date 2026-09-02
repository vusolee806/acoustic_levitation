import os
import sys
import time
import numpy as np
import gymnasium as gym

# Directly import mujoco and the viewer
import mujoco
import mujoco.viewer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import src  

def validate_direct_mujoco():
    # Initialize the environment normally (no render_mode needed here)
    env = gym.make("AcousticLevitator-v0")
    obs, info = env.reset(seed=42)
    
    # Extract the raw MuJoCo model and data from the Gymnasium wrapper
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    static_focal_point = np.array([0.0, 0.0, 0.05], dtype=np.float32)
    print("Opening direct MuJoCo viewer...")
    
    # Explicitly launch the MuJoCo viewer and take control of the loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Step the physics
            obs, reward, terminated, truncated, info = env.step(static_focal_point)
            
            # Manually sync the viewer to update the 3D graphics
            viewer.sync()
            
            # Slow down the physics loop so you can watch
            time.sleep(0.05) 
            
            # If the ball escapes, reset it instantly so the window stays open
            if terminated or truncated:
                print("Ball escaped! Resetting...")
                env.reset()

    env.close()

if __name__ == "__main__":
    validate_direct_mujoco()
