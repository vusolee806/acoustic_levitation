import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch

# 1. Dynamically add the project root to Python's path so it can find 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# 2. Initialize SimulationApp FIRST with memory/headless safeguards
from isaacsim import SimulationApp

CONFIG = {
    "headless": False,
    "renderer": "RayTracedLighting",
    "width": 1024,
    "height": 576,
    "anti_aliasing": 0,
    "fast_shutdown": True,
}
simulation_app = SimulationApp(CONFIG)

# 3. Import Omniverse and USD modules AFTER SimulationApp is up
from pxr import Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicSphere, FixedCylinder

# 4. Import the acoustic physics module
from src.acoustic_physics_pytorch import (
    generate_twin_trap_phases,
    get_physics_outputs,
    PARTICLE_VOL,
    RHO_1,
    PARTICLE_RADIUS,
)


def zaxis_to_quat(zaxis):
    zaxis = np.array(zaxis, dtype=np.float64)
    norm = np.linalg.norm(zaxis)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    zaxis = zaxis / norm
    default_z = np.array([0.0, 0.0, 1.0])

    if np.allclose(zaxis, default_z):
        return np.array([1.0, 0.0, 0.0, 0.0])
    elif np.allclose(zaxis, -default_z):
        # 180-deg rotation about Y axis
        return np.array([0.0, 0.0, 1.0, 0.0])

    axis = np.cross(default_z, zaxis)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(default_z, zaxis), -1.0, 1.0))
    quat = Gf.Rotation(Gf.Vec3d(*axis), np.rad2deg(angle)).GetQuat()
    return np.array([quat.real, quat.imaginary[0], quat.imaginary[1], quat.imaginary[2]])


def main():
    # 60 Hz rendering, 120 Hz physics for smooth stability
    world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane(z_position=-0.15)

    xml_path = os.path.join(project_root, "assets", "mujoco_levitator.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found at: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    pos_list = []
    zaxis_list = []

    print("Parsing transducers from XML...")
    for body in root.findall(".//body"):
        name = body.get("name")
        if name and name.startswith("sensor_"):
            p = [float(v) for v in body.get("pos").split()]
            geom = body.find("geom")
            z = [float(v) for v in geom.get("zaxis").split()]

            pos_list.append(p)
            zaxis_list.append(z)

            world.scene.add(
                FixedCylinder(
                    prim_path=f"/World/Levitator/{name}",
                    name=name,
                    position=np.array(p),
                    orientation=zaxis_to_quat(z),
                    radius=0.01,
                    height=0.01,
                    color=np.array([0.2, 0.6, 1.0]),
                )
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
            color=np.array([1.0, 0.8, 0.0]),
        )
    )

    # Reset initializes all registered prim views properly
    world.reset()
    print("Acoustic Levitation Simulation Running. Calculating ARF on ball...")

    step_count = 0
    gravity_force = -ball_mass * 9.81

    while simulation_app.is_running():
        # 1. Query current world position of the ball
        ball_pos, _ = test_ball.get_world_pose()

        # 2. Reshape into (1, 3) for the PyTorch forward pass
        ball_pos_batch = torch.tensor(ball_pos, dtype=torch.float32).unsqueeze(0)

        # 3. Compute Acoustic Radiation Force (ARF)
        _, forces_array = get_physics_outputs(ball_pos_batch, trans_pos, trans_zaxis, phases)

        # Convert back to numpy array of shape (1, 3)
        if isinstance(forces_array, torch.Tensor):
            force_np = forces_array.detach().cpu().numpy().reshape(1, 3)
        else:
            force_np = np.array(forces_array, dtype=np.float32).reshape(1, 3)

        # 4. Apply force directly via the DynamicSphere's native RigidPrim interface
        test_ball.apply_forces(forces=force_np)

        # Step physics & rendering
        world.step(render=True)

        step_count += 1
        if step_count % 120 == 0:
            print(
                f"[Step {step_count:04d}] "
                f"Pos: ({ball_pos[0]:.4f}, {ball_pos[1]:.4f}, {ball_pos[2]:.4f}) | "
                f"Fz: {force_np[0, 2]:.2e} N | Grav: {gravity_force:.2e} N"
            )

    simulation_app.close()


if __name__ == "__main__":
    main()
