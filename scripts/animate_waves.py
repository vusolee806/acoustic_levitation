import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import levitate
import HemisAcoustic.assets.constants as const

def main():
    print("Loading array geometry for wave animation...")
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

    # 1. Setup Array
    array = levitate.arrays.NormalTransducerArray(
        positions=transducer_positions.T, 
        normals=transducer_zaxis.T,
        transducer_size=const.TRANSDUCER_SIZE  
    )
    array.freq = const.FREQ
    target_focal_point = np.array([0.0, 0.0, 0.04])

    # 2. Setup the Trap (Using Twin trap as an example)
    array.phases = array.focus_phases(target_focal_point) + array.signature(position=target_focal_point, stype='twin')
    complex_weights = 1.0 * np.exp(1j * array.phases)

    # 3. Create a 2D Grid (XZ Plane slice cutting through the middle)
    x = np.linspace(-0.04, 0.04, 150)
    z = np.linspace(0.00, 0.08, 150)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X) # Flat on the Y=0 plane
    
    # Levitate expects points in shape (3, N)
    grid_points = np.stack((X, Y, Z))
    
    # 4. Evaluate the COMPLEX Pressure field (not radiation force!)
    pressure_evaluator = levitate.fields.Pressure(array)
    complex_pressure = pressure_evaluator(complex_weights, grid_points)

    # 5. Animate the Waves
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Dynamic Acoustic Wave Propagation (40 kHz)", fontsize=14)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")
    
    # Plot the focal point marker
    ax.plot(0, 0.04, 'rx', markersize=10, label="Focal Center")
    ax.legend()

    # Create the initial heatmap frame
    # We use vmin/vmax to lock the color scale so it doesn't flicker
    max_pressure = np.max(np.abs(complex_pressure))
    im = ax.imshow(np.zeros_like(X), extent=[-0.04, 0.04, 0.00, 0.08], 
                   origin='lower', cmap='RdBu', vmin=-max_pressure, vmax=max_pressure)

    # The animation update function
    omega = 2 * np.pi * const.FREQ
    
    def update(frame):
        # Time steps (simulating a fraction of a second to see the waves move)
        t = frame * (1.0 / const.FREQ / 20.0) # 20 frames per wave cycle
        
        # The Magic Formula: p(t) = Real( P * e^(-i * w * t) )
        dynamic_pressure = np.real(complex_pressure * np.exp(-1j * omega * t))
        
        im.set_array(dynamic_pressure)
        return [im]

    print("Generating wave animation...")
    # Run the animation at 30 fps
    ani = animation.FuncAnimation(fig, update, frames=60, interval=33, blit=True)
    
    plt.show()

if __name__ == "__main__":
    main()
