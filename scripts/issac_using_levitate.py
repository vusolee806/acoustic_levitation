import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. INITIALIZE ISAAC SIM APP FIRST
from isaacsim import SimulationApp
# Pass optimization flags directly into the app configuration
config = {
    "headless": False,           # Set to True if you only need the data/graphs, not the visuals
    "width": 800,               # Lower window resolution
    "height": 600,
    "renderer": "RayTracedLighting",  # Forces RTX Real-Time instead of heavy Path Tracing
    "anti_aliasing": 0,         # Disable AA
}
simulation_app = SimulationApp(config)

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
import omni.physx as _physx

# Add your project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import levitate
from src.plot_utils import plot_simulation_results

# Acoustic Constants
FREQ = 40000.0         
C_0 = 343.0            
RHO_1 = 40.0           
C_1 = 1200.0           
PARTICLE_RADIUS = 0.002 

def main():
    # 2. SETUP ISAAC SIM WORLD
    world = World(physics_dt=1.0/10000.0, rendering_dt=1.0/60.0) # Match MuJoCo's 0.0001s step
    world.scene.add_default_ground_plane()

    # NOTE: In a full pipeline, you load your pre-configured Soft Body USD here.
    # For this script, we assume you have a USD file with a PhysxDeformableBodyAPI applied.
    soft_ball_path = os.path.join(PROJECT_ROOT, "assets", "soft_ball.usd")
    stage_utils.add_reference_to_stage(usd_path=soft_ball_path, prim_path="/World/SoftBall")
    
    # Wrap the prim to track its position
    ball_prim = XFormPrim(prim_path="/World/SoftBall/Mesh", name="soft_ball_prim")
    world.scene.add(ball_prim)
    
    world.reset()

    # 3. SETUP LEVITATE ARRAY
    # (Assuming you extracted these positions from your bowl CAD/USD)
    num_transducers = 79
    transducer_positions = np.zeros((num_transducers, 3)) # Replace with your CAD coordinates
    transducer_zaxis = np.zeros((num_transducers, 3))     # Replace with your CAD normals
    
    # Force Transducers to point upward for this placeholder
    transducer_zaxis[:, 2] = 1.0 

    levitate.frequency = FREQ
    ball_material = levitate.materials.Material(rho=RHO_1, c=C_1)
    
    array = levitate.arrays.NormalTransducerArray(
        positions=transducer_positions.T, 
        normals=transducer_zaxis.T,
        transducer_size=0.02  
    )
    
    force_evaluator = levitate.fields.RadiationForce(
        array, radius=PARTICLE_RADIUS, material=ball_material
    )

    # 4. ACOUSTIC MATH (Twin Trap)
    target_focal_point = np.array([0.0, 0.0, 0.04])
    focus_phases = array.focus_phases(target_focal_point)
    twin_signature = array.signature(position=target_focal_point, stype='twin')
    
    array.phases = focus_phases + twin_signature
    complex_weights = 0.3 * np.exp(1j * array.phases)

    # Data Logging
    time_log = []
    fx_log, fy_log, fz_log = [], [], []
    x_pos_log, y_pos_log, z_pos_log = [], [], []

    print("Running Isaac Sim Soft Body Simulation...")

    physx_interface = _physx.get_physx_interface()

    # 5. PHYSICS LOOP
    while simulation_app.is_running():
        world.step(render=True)
        current_time = world.current_time
        
        if world.is_playing():
            # Get current position of the soft body's center
            ball_pos, _ = ball_prim.get_world_pose()
            
            # Evaluate the acoustic force
            arf_force = force_evaluator(complex_weights, ball_pos)
            
            # Apply force to the soft body using PhysX
            # (Requires applying spatial force to the rigid/deformable body root)
            physx_interface.apply_force_at_pos(
                ball_prim.prim_path, 
                force=arf_force.tolist(), 
                pos=ball_pos.tolist()
            )
            
            # Log Data
            time_log.append(current_time)
            fx_log.append(arf_force[0])
            fy_log.append(arf_force[1])
            fz_log.append(arf_force[2])
            x_pos_log.append(ball_pos[0])  
            y_pos_log.append(ball_pos[1])  
            z_pos_log.append(ball_pos[2])

            # Stop condition for the graph
            if current_time > 2.5:
                break

    # 6. PLOT RESULTS
    simulation_app.close()
    
    plot_simulation_results(
        time_log, 
        fx_log, fy_log, fz_log, 
        x_pos_log, y_pos_log, z_pos_log
    )

if __name__ == "__main__":
    main()
