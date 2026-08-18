import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch

# 1. Dynamically add the project root to Python's path so it can find 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# 2. Import the physics logic from your src folder
from src.acoustic_physics_pytorch import (
    generate_twin_trap_phases, 
    get_physics_outputs, 
    PARTICLE_VOL, 
    RHO_1, 
    PARTICLE_RADIUS
)

# 3. Initialize SimulationApp FIRST
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "use_fabric": False
})

# 4. Import Omniverse and USD (pxr) modules AFTER SimulationApp is running
from pxr import Gf  # <--- MOVED HERE!
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicSphere, FixedCylinder
from isaacsim.core.prims import RigidPrim

def zaxis_to_quat(zaxis):
    zaxis = np.array(zaxis, dtype=np.float64)
    zaxis = zaxis / np.linalg.norm(zaxis)
    default_z = np.array([0.0, 0.0, 1.0])
    
    if np.allclose(zaxis, default_z):
        return np.array([1.0, 0.0, 0.0, 0.0])
    elif np.allclose(zaxis, -default_z):
        return np.array([0.0, 1.0, 0.0, 0.0])
        
    axis = np.cross(default_z, zaxis)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(default_z, zaxis), -1.0, 1.0))
    quat = Gf.Rotation(Gf.Vec3d(*axis), np.rad2deg(angle)).GetQuat()
    return np.array([quat.real, quat.imaginary[0], quat.imaginary[1], quat.imaginary[2]])

def main():
    # 60 Hz rendering, 120 Hz physics for smooth stability
    world = World(physics_dt=1.0/120.0, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane(z_position=-0.15)

    xml_path = os.path.join(project_root, "assets", "mujoco_levitator.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    pos_list = []
    zaxis_list = []

    print("Parsing 79 Transducers from XML...")
    for body in root.findall(".//body"):
        name = body.get("name")
        if name and name.startswith("sensor_"):
            p = [float(v) for v in body.get("pos").split()]
            geom = body.find("geom")
            z = [float(v) for v in geom.get("zaxis").split()]
            
            pos_list.append(p)
            zaxis_list.append(z)
            
            FixedCylinder(
                prim_path=f"/World/Levitator/{name}",
                name=name,
                position=np.array(p),
                orientation=zaxis_to_quat(z),
                radius=0.01,
                height=0.01,
                color=np.array([0.2, 0.6, 1.0])
            )

    trans_pos = torch.tensor(pos_list, dtype=torch.float32)
    trans_zaxis = torch.tensor(zaxis_list, dtype=torch.float32)

    # Define the acoustic trap focal point (height = 5 cm)
    focal_target = torch.tensor([0.0, 0.0, 0.05], dtype=torch.float32)
    phases = generate_twin_trap_phases(focal_target, trans_pos)

    # Spawn test ball
    ball_mass = float(PARTICLE_VOL * RHO_1)
    test_ball = world.scene.add(
        DynamicSphere(
            prim_path="/World/test_ball",
            name="test_ball",
            position=np.array([0.0, 0.0, 0.052]),  # Drop slightly above the trap node
            radius=PARTICLE_RADIUS,
            mass=ball_mass,
            color=np.array([1.0, 0.8, 0.0])
        )
    )

    # --- ADD THESE TWO LINES ---
    ball_view = RigidPrim(prim_paths_expr="/World/test_ball", name="ball_view")
    world.scene.add(ball_view)
    # ---------------------------
    # ---------------------------

    world.reset()
    print("Acoustic Levitation Simulation Running. Calculating ARF on ball...")

    step_count = 0
    while simulation_app.is_running():
        # 1. Query the current ball position
        ball_pos, _ = test_ball.get_world_pose()

        # 2. Reshape the (3,) position array into (1, 3) for PyTorch batching
        ball_pos_batch = ball_pos.reshape(1, 3)

        # 3. Call your pure physics function
        _, forces_array = get_physics_outputs(ball_pos_batch, trans_pos, trans_zaxis, phases)

        # 4. Flatten the output back to (3,) and apply to the ball
        force_vec = forces_array.flatten()
        ball_view.apply_forces(forces=force_vec.reshape(1, 3))

        world.step(render=True)
        
        step_count += 1
        if step_count % 120 == 0:
            print(f"[Pos: ({ball_pos[0]:.4f}, {ball_pos[1]:.4f}, {ball_pos[2]:.4f})] "
                  f"[Force: Fz={force_vec[2]:.2e} N | Grav={-ball_mass*9.81:.2e} N]")

    simulation_app.close()

if __name__ == "__main__":
    main()
