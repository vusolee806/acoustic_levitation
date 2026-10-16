import sys
import os
import numpy as np
import mujoco

# Add project root to path so we can import configs
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import levitate
import configs.constants as const

def main():
    print("==================================================")
    print("       LEVITATE STATIC CHECKS (ACOUSTIC MATH)     ")
    print("==================================================\n")
    
    # 1. Load MuJoCo purely to extract the exact physical coordinates
    xml_path = os.path.join(PROJECT_ROOT, "assets", "mujoco_levitator.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
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

    # 2. Initialize Levitate Array
    array = levitate.arrays.NormalTransducerArray(
        positions=transducer_positions.T, 
        normals=transducer_zaxis.T,
        transducer_size=const.TRANSDUCER_SIZE  
    )
    array.freq = const.FREQ
    
    # 3. Setup the Radiation Force Evaluator
    ball_material = levitate.materials.Material(rho=const.RHO_1, c=const.C_1)
    force_evaluator = levitate.fields.RadiationForce(
        array, radius=const.PARTICLE_RADIUS, material=ball_material
    )

    # 4. Configure Trap Parameters (30% power, Vortex trap, 4cm height)
    target_focal_point = np.array([0.0, 0.0, 0.04])
    focus_phases = array.focus_phases(target_focal_point)
    vortex_signature = array.signature(position=target_focal_point, stype='vortex')
    
    array.phases = focus_phases + vortex_signature
    complex_weights = 0.3 * np.exp(1j * array.phases)

    # 5. Calculate physical constraints
    volume = (4/3) * np.pi * (const.PARTICLE_RADIUS ** 3)
    mass = volume * const.RHO_1
    gravity_force = mass * 9.81
    
    print(f"System Configuration:")
    print(f"  - Trap Type: Vortex")
    print(f"  - Amplitude Power: 30%")
    print(f"  - Particle Weight: {gravity_force:.3e} N\n")

    # ---------------------------------------------------------
    # TEST 1: The Dead-Center Balance
    # ---------------------------------------------------------
    print("[TEST 1] Dead-Center Balance (Pos: 0, 0, 0.04)")
    forces_center = force_evaluator(complex_weights, target_focal_point)
    print(f"  Fx: {forces_center[0]:.3e} N")
    print(f"  Fy: {forces_center[1]:.3e} N")
    print(f"  Fz: {forces_center[2]:.3e} N")
    
    if abs(forces_center[0]) < 1e-6 and abs(forces_center[1]) < 1e-6:
        print("  -> PASS: Lateral forces perfectly cancel out at the node.")
    else:
        print("  -> FAIL: Trap is asymmetric. Lateral forces are pushing the particle.")

    # ---------------------------------------------------------
    # TEST 2: Lateral Spring Test (X-Axis)
    # ---------------------------------------------------------
    off_center_x = np.array([0.001, 0.0, 0.04]) # 1mm right
    print("\n[TEST 2] Lateral Restoring Force (Pos: 0.001, 0, 0.04)")
    forces_x = force_evaluator(complex_weights, off_center_x)
    print(f"  Fx: {forces_x[0]:.3e} N")
    
    if forces_x[0] < 0:
        print("  -> PASS: Trap acts as a stable spring, pushing particle back left.")
    else:
        print("  -> FAIL: Trap is unstable and pushes particle outward.")

    # ---------------------------------------------------------
    # TEST 3: Vertical Spring Test (Z-Axis)
    # ---------------------------------------------------------
    too_high_z = np.array([0.0, 0.0, 0.041]) # 1mm high
    print("\n[TEST 3] Vertical Restoring Force (Pos: 0, 0, 0.041)")
    forces_z = force_evaluator(complex_weights, too_high_z)
    print(f"  Fz: {forces_z[2]:.3e} N")
    
    if forces_z[2] < gravity_force:
        print("  -> PASS: Acoustic lift drops below gravity. Particle will safely fall back to center.")
    else:
        print("  -> FAIL: Acoustic lift exceeds gravity even 1mm too high. Particle will shoot upwards.")
        
    print("\n==================================================")

if __name__ == "__main__":
    main()
