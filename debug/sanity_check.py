import sys
import os
import numpy as np
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import levitate
import configs.constants as const

def main():
    print("--- STARTING ACOUSTIC LEVITATION SANITY CHECKS ---")
    
    xml_path = os.path.join(PROJECT_ROOT, "assets", "mujoco_levitator.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    # 1. Extract Transducers
    num_transducers = 79
    transducer_positions = np.zeros((num_transducers, 3))
    transducer_zaxis = np.zeros((num_transducers, 3))

    for i in range(num_transducers):
        site_id = model.site(f"site_sensor_{i+1}").id
        transducer_positions[i] = data.site_xpos[site_id]
        mat = data.site_xmat[site_id].reshape(3, 3)
        transducer_zaxis[i] = mat[:, 2]

    # --- TEST 1: Coordinate Alignment ---
    print("\n[TEST 1] Transducer Coordinate Alignment")
    mean_pos = np.mean(transducer_positions, axis=0)
    print(f"Mean Position of all transducers: X={mean_pos[0]:.4f}, Y={mean_pos[1]:.4f}, Z={mean_pos[2]:.4f}")
    if abs(mean_pos[0]) < 0.01 and abs(mean_pos[1]) < 0.01:
        print("  -> PASS: Array is properly centered on the X-Y plane.")
    else:
        print("  -> WARNING: Array is off-center. Check your MuJoCo XML positions.")

    # 2. Setup Levitate Array & Weights (30% power, Vortex trap)
    array = levitate.arrays.NormalTransducerArray(
        positions=transducer_positions.T, 
        normals=transducer_zaxis.T,
        transducer_size=const.TRANSDUCER_SIZE  
    )
    array.freq = const.FREQ
    ball_material = levitate.materials.Material(rho=const.RHO_1, c=const.C_1)
    
    force_evaluator = levitate.fields.RadiationForce(
        array, radius=const.PARTICLE_RADIUS, material=ball_material
    )

    target_focal_point = np.array([0.0, 0.0, 0.04])
    focus_phases = array.focus_phases(target_focal_point)
    vortex_signature = array.signature(position=target_focal_point, stype='vortex')
    array.phases = focus_phases + vortex_signature
    complex_weights = 0.3 * np.exp(1j * array.phases)

    # Calculate actual gravity force of the particle (m * g)
    volume = (4/3) * np.pi * (const.PARTICLE_RADIUS ** 3)
    mass = volume * const.RHO_1
    gravity_force = mass * 9.81
    print(f"\nParticle Weight (Gravity): {gravity_force:.2e} N")

    # --- TEST 2: The Dead-Center Test ---
    print("\n[TEST 2] Dead-Center Balance (Pos: 0, 0, 0.04)")
    forces_center = force_evaluator(complex_weights, target_focal_point)
    print(f"Fx: {forces_center[0]:.2e} N")
    print(f"Fy: {forces_center[1]:.2e} N")
    print(f"Fz: {forces_center[2]:.2e} N")
    
    if abs(forces_center[0]) < 1e-6 and abs(forces_center[1]) < 1e-6:
        print("  -> PASS: Lateral forces are perfectly balanced (0.0) at the center.")
    else:
        print("  -> WARNING: Lateral forces exist at the center. Trap is lopsided.")

    # --- TEST 3: Lateral Spring Test (X-Axis) ---
    off_center_x = np.array([0.001, 0.0, 0.04]) # 1mm to the right
    print("\n[TEST 3] Lateral Restoring Force (Pos: 0.001, 0, 0.04)")
    forces_x = force_evaluator(complex_weights, off_center_x)
    print(f"Fx: {forces_x[0]:.2e} N (Expected: Negative value)")
    
    if forces_x[0] < 0:
        print("  -> PASS: Trap correctly pushes the particle back to the left (X=0).")
    else:
        print("  -> FAIL: Trap pushes the particle further away! (Unstable)")

    # --- TEST 4: Vertical Spring Test (Z-Axis) ---
    too_high_z = np.array([0.0, 0.0, 0.041]) # 1mm too high
    print("\n[TEST 4] Vertical Restoring Force (Pos: 0, 0, 0.041)")
    forces_z = force_evaluator(complex_weights, too_high_z)
    print(f"Fz: {forces_z[2]:.2e} N (Upward acoustic force)")
    
    if forces_z[2] < gravity_force:
        print("  -> PASS: Upward force is less than gravity. Particle will fall back down to 0.04.")
    else:
        print("  -> FAIL: Upward force is still stronger than gravity. Particle will be ejected upwards.")

    print("\n--- SANITY CHECKS COMPLETE ---")

if __name__ == "__main__":
    main()
