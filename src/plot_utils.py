import matplotlib.pyplot as plt

def plot_simulation_results(time_log, fx, fy, fz, x_pos, y_pos, z_pos):
    """
    Renders a 2x2 dashboard showing acoustic forces, vertical trajectory, 
    and a time-mapped top-down view of the particle's X-Y orbital path.
    """
    print("Đang tạo đồ thị...")
    
    fig = plt.figure(figsize=(14, 8))
    
    # Top Left: Forces
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time_log, fz, label='F_z (Vertical)', color='blue')
    ax1.plot(time_log, fx, label='F_x (Lateral)', color='red', alpha=0.6)
    ax1.plot(time_log, fy, label='F_y (Lateral)', color='green', alpha=0.6)
    ax1.set_ylabel("Force (N)")
    ax1.set_title("Acoustic Radiation Force")
    ax1.legend()
    ax1.grid(True)

    # Bottom Left: Z-Height
    ax2 = plt.subplot(2, 2, 3)
    ax2.plot(time_log, z_pos, label='Z Position', color='purple')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Height (m)")
    ax2.set_title("Vertical Trajectory (Z-Axis)")
    ax2.legend()
    ax2.grid(True)

    # Right Side: X-Y Plane with Time Colormap
    ax3 = plt.subplot(2, 2, (2, 4))
    
    # Map time_log to the 'c' (color) argument using the 'viridis' colormap
    path_scatter = ax3.scatter(x_pos, y_pos, c=time_log, cmap='viridis', s=10, zorder=2)
    ax3.plot(0, 0, 'rx', markersize=12, label='Focal Center', zorder=3) 
    
    # Add a colorbar to act as the time legend
    cbar = plt.colorbar(path_scatter, ax=ax3)
    cbar.set_label('Simulation Time (s)')
    
    ax3.set_xlabel("X Position (m)")
    ax3.set_ylabel("Y Position (m)")
    ax3.set_title("Top-Down Trajectory (Time-Mapped)")
    ax3.axis('equal')  
    ax3.legend(loc="upper left")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
