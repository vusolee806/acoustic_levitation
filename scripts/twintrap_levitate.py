"""
Twin-trap MuJoCo simulation, rewritten to use `levitate` instead of the
custom PyTorch/autograd Gor'kov implementation.

THE BUG THAT WAS BREAKING YOUR SIMULATION
-------------------------------------------------------------------------
In `twin_trap_pytorch.py` the transducer world positions were read as:

    transducer_positions[i] = model.site_pos[site_id]

`model.site_pos` is the site's position **relative to its parent body's
frame**, not its world position. In `mujoco_levitator.xml`, every
`<site>` sits at the local origin of its body (`pos="0 0 0"`), while the
actual world offset is on the *body* (e.g. `<body name="sensor_1"
pos="0.01968 0.01136 -0.08294">`). So every one of your 79 transducers
was silently treated as sitting at (0, 0, 0) -- the whole array
collapsed onto a single point, which is why the resulting acoustic
field/forces didn't behave like a real bowl-shaped array.

The world-frame position and orientation only exist after forward
kinematics, in `data.site_xpos` / `data.site_xmat`:

    mujoco.mj_forward(model, data)
    world_pos  = data.site_xpos[site_id]            # (3,)
    world_rmat = data.site_xmat[site_id].reshape(3, 3)
    world_zaxis = world_rmat[:, 2]

(Orientation happened to come out right in your version only because
none of the sensor bodies have a body-level rotation, so the local and
world orientations coincide -- position did not have that luck.)

This script fixes that, and swaps the physics for the calibrated
`levitate` backend in `acoustic_physics_levitate.py`.
"""

import sys
import os
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from levitate_lib_physics import AcousticField, PARTICLE_VOL, RHO_1


def load_transducer_world_poses(model, data, num_transducers=79):
    """Read WORLD-frame transducer positions/orientations. Requires that
    `mujoco.mj_forward(model, data)` has already been called so
    `data.site_xpos` / `data.site_xmat` are populated."""
    positions = np.zeros((num_transducers, 3))
    zaxis = np.zeros((num_transducers, 3))
    for i in range(num_transducers):
        site_id = model.site(f"site_sensor_{i + 1}").id
        positions[i] = data.site_xpos[site_id]                  # <-- fix (was model.site_pos)
        rmat = data.site_xmat[site_id].reshape(3, 3)             # <-- fix (was mju_quat2Mat on the local site_quat)
        zaxis[i] = rmat[:, 2]
    return positions, zaxis


def main():
    xml_path = os.path.join(PROJECT_ROOT, "assets", "mujoco_levitator.xml")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except ValueError as e:
        print(f"Error loading XML: {e}")
        return

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # populate data.site_xpos / data.site_xmat before we read them

    num_transducers = 79
    transducer_positions, transducer_zaxis = load_transducer_world_poses(
        model, data, num_transducers)

    field = AcousticField(transducer_positions, transducer_zaxis)

    ball_id = model.body("test_ball").id

    time_log, fx_log, fy_log, fz_log, z_pos_log = [], [], [], [], []

    print("Running simulation... close the MuJoCo window to see the plots.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            ball_pos = data.xpos[ball_id]

            # --- 1. Target focal point (would be the RL action in your setup) ---
            target_focal_point = np.array([0.0, 0.0, 0.04])

            # --- 2. Phases for the twin trap steered to the target ---
            phases = field.generate_twin_trap_phases(target_focal_point)
            complex_amplitudes = field.complex_amplitudes(phases)

            # --- 3. Physics at the ball's current position ---
            U, arf_force = field.evaluate(ball_pos.copy(), complex_amplitudes)

            # No more np.clip(force, -1, 1) hack needed: levitate's transducer
            # model is calibrated to the real Murata MA40S4S output (p0 = 6 Pa
            # @ 1 m @ full drive), so forces come out at the correct physical
            # scale (micronewtons for a ~1.3 mg foam ball) instead of the
            # arbitrary units produced by the uncalibrated P0_A constant.

            dof_start = model.body_dofadr[ball_id]
            data.qfrc_applied[dof_start: dof_start + 3] = arf_force

            time_log.append(data.time)
            fx_log.append(arf_force[0])
            fy_log.append(arf_force[1])
            fz_log.append(arf_force[2])
            z_pos_log.append(ball_pos[2])

            mujoco.mj_step(model, data)
            viewer.sync()

    print("Simulation stopped. Building plots...")

    weight = PARTICLE_VOL * RHO_1 * 9.81

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time_log, fz_log, label="F_z (Vertical Force)", color="blue")
    ax1.plot(time_log, fx_log, label="F_x (Lateral Force)", color="red", alpha=0.6)
    ax1.plot(time_log, fy_log, label="F_y (Lateral Force)", color="green", alpha=0.6)
    ax1.axhline(weight, color="black", linestyle="--", label=f"Particle weight ({weight:.2e} N)")
    ax1.set_ylabel("Force (Newtons)")
    ax1.set_title("Acoustic Radiation Force over Time (levitate backend)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_log, z_pos_log, label="Z Position (Height)", color="purple")
    ax2.set_xlabel("Simulation Time (Seconds)")
    ax2.set_ylabel("Height (Meters)")
    ax2.set_title("Particle Trajectory (Settling into Trap)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
