import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Setup 4 dummy transducers
    trans_pos = torch.tensor([
        [0.05, 0.05, 0.0],
        [-0.05, 0.05, 0.0],
        [-0.05, -0.05, 0.0],
        [0.05, -0.05, 0.0]
    ], dtype=torch.float32)
    
    trans_zaxis = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=torch.float32)
    phases = torch.zeros(4, dtype=torch.float32)
    
    # 2. Generate 500 test points along the Z-axis (from 5cm to 15cm height)
    z_min, z_max = 0.05, 0.15
    num_points = 500
    
    test_points = np.zeros((num_points, 3), dtype=np.float32)
    z_vals = np.linspace(z_min, z_max, num_points)
    test_points[:, 2] = z_vals  # Set Z coordinates, X and Y remain 0.0
    
    # 3. Calculate Physics 
    # (Assuming get_physics_outputs is defined as in the previous step)
    U_array, forces_array = get_physics_outputs(test_points, trans_pos, trans_zaxis, phases)
    
    # Extract just the Z-component of the force
    F_z = forces_array[:, 2]
    
    # Calculate Particle Gravity for reference (F_g = - m * g)
    gravity_force = - (PARTICLE_VOL * RHO_1) * 9.81
    
    # 4. Plotting
    # Convert Z to mm for easier reading
    z_vals_mm = z_vals * 1000 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- Top Plot: Gor'kov Potential ---
    ax1.plot(z_vals_mm, U_array, color='blue', linewidth=2)
    ax1.set_ylabel("Gor'kov Potential U (Joules)", color='blue', fontweight='bold')
    ax1.set_title("Acoustic Levitation Field Along Z-Axis (X=0, Y=0)", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Bottom Plot: Acoustic Radiation Force ---
    ax2.plot(z_vals_mm, F_z, color='red', linewidth=2, label="Acoustic Z-Force")
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='-', label="Zero Force")
    ax2.axhline(gravity_force, color='green', linewidth=1.5, linestyle='--', label=f"Gravity ({gravity_force:.2e} N)")
    
    ax2.set_xlabel("Z-Axis Height (mm)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Z-Force (Newtons)", color='red', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    # Highlight stable trapping nodes (where Force crosses zero downwards)
    for i in range(1, num_points):
        # Check if force goes from positive to negative
        if F_z[i-1] > 0 and F_z[i] < 0:
            ax1.axvline(z_vals_mm[i], color='purple', linestyle=':', alpha=0.5)
            ax2.axvline(z_vals_mm[i], color='purple', linestyle=':', alpha=0.5)
            ax2.plot(z_vals_mm[i], 0, 'mo', markersize=8) # Purple dot at the node
            
    plt.tight_layout()
    plt.show()
