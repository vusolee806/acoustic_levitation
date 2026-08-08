import numpy as np
import matplotlib.pyplot as plt

# Import the physics functions and constants from your src folder
from src.acoustic_physics import calculate_complex_pressure, TARGET_WAVELENGTH

def visualize_sound_wave():
    # 1. Setup Transducer Array (2 opposed transducers for a standing wave)
    # Distance between them is set to 2 wavelengths to clearly see the nodes
    z_dist = TARGET_WAVELENGTH * 2 
    
    trans_pos = np.array([
        [0.0, 0.0, -z_dist/2],  # Bottom transducer
        [0.0, 0.0,  z_dist/2]   # Top transducer
    ])
    
    trans_zaxis = np.array([
        [0.0, 0.0, 1.0],   # Bottom points UP (+Z)
        [0.0, 0.0, -1.0]   # Top points DOWN (-Z)
    ])
    
    phases = np.array([0.0, 0.0]) # In-phase

    # 2. Create a 2D Spatial Grid (X-Z plane)
    # We will look at a slice right down the middle (Y=0)
    x_range = np.linspace(-0.1, 0.1, 100)
    z_range = np.linspace(-0.15, 0.15, 200)
    X, Z = np.meshgrid(x_range, z_range)
    Y = np.zeros_like(X)
    
    # Flatten the grid to iterate through points
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # 3. Calculate Pressure Field
    print("Calculating pressure field... this might take a few seconds.")
    pressure_complex = np.zeros(points.shape[0], dtype=complex)
    
    # Because your calculate_complex_pressure function expects a single 1D point array 
    # to broadcast against the transducer list, we iterate through the grid.
    for i, pt in enumerate(points):
        pressure_complex[i] = calculate_complex_pressure(pt, trans_pos, trans_zaxis, phases)
        
    # Reshape back to the 2D grid
    pressure_complex = pressure_complex.reshape(X.shape)
    
    # 4. Extract Visual Data
    # Absolute magnitude |P| shows the standing wave nodes (dark spots) and antinodes (bright spots)
    pressure_magnitude = np.abs(pressure_complex)
    
    # Real part shows the instantaneous wave peaks and troughs at t=0
    pressure_real = np.real(pressure_complex)

  # 5. Plotting with Matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- ADD THIS: Set a manual limit for the colormap ---
    # Since P0_A = 5.0 and distance to center is ~0.1, expected max amplitude is ~100
    display_max = 150 
    
    # Plot 1: Pressure Magnitude
    c1 = ax1.pcolormesh(X, Z, pressure_magnitude, cmap='inferno', shading='auto', 
                        vmax=display_max) # Clamped maximum
    ax1.scatter(trans_pos[:, 0], trans_pos[:, 2], color='cyan', marker='s', s=100, label='Transducers')
    ax1.set_title('Pressure Magnitude |P|\n(Dark regions are pressure nodes)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.legend()
    fig.colorbar(c1, ax=ax1, label='Pressure Amplitude')

    # Plot 2: Instantaneous Pressure (Real Part)
    c2 = ax2.pcolormesh(X, Z, pressure_real, cmap='RdBu', shading='auto', 
                        vmin=-display_max, vmax=display_max) # Clamped min/max
    ax2.scatter(trans_pos[:, 0], trans_pos[:, 2], color='lime', marker='s', s=100)
    ax2.set_title('Instantaneous Pressure Wave (Real Part)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    fig.colorbar(c2, ax=ax2, label='Instantaneous Pressure')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_sound_wave()
