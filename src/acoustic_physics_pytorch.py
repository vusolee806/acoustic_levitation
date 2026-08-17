import torch
import numpy as np

# ==========================================
# 1. ACOUSTIC & MATERIAL CONSTANTS
# ==========================================
# Fluid medium (Air)
C_0 = 343.0            # Speed of sound in air (m/s)
RHO_0 = 1.225          # Density of air (kg/m^3)

# Particle (e.g., Expanded Polystyrene / EPS Foam)
C_1 = 1200.0           # Speed of sound in EPS (m/s)
RHO_1 = 40.0           # Density of EPS (kg/m^3)

# # Particle (in the AIR)
# C_1 = 343.0          # Speed of sound in air (m/s)
# RHO_1 = 1.225        # Density of air (kg/m^3)


# --- UPDATED PARTICLE & FREQUENCY SETTINGS ---
# Diameter is 0.10, so radius is 0.05
PARTICLE_RADIUS = 0.05 
PARTICLE_VOL = (4/3) * np.pi * (PARTICLE_RADIUS**3)

#choose wavelength such that can get the frequency of 40KHZ
TARGET_WAVELENGTH = 0.008575

# Calculate Frequency to satisfy the wavelength constraint: f = c / lambda
FREQ = C_0 / TARGET_WAVELENGTH  # Evaluates to 40 kHz
OMEGA = 2 * np.pi * FREQ
WAVENUMBER = OMEGA / C_0        # k = 2*pi / lambda
# ---------------------------------------------

TRANSDUCER_RADIUS = 0.005 # 'a' in the directivity formula (5mm)
P0_A = 5.0             # Combined constant for (P_0 * A) Output efficiency & amplitude

# Pre-calculate Gor'kov Constants (Equations 6 and 7)
K1 = 0.25 * PARTICLE_VOL * ((1 / (C_0**2 * RHO_0)) - (1 / (C_1**2 * RHO_1)))
K2 = 0.75 * PARTICLE_VOL * ((RHO_0 - RHO_1) / (OMEGA**2 * RHO_0 * (RHO_0 + 2 * RHO_1)))

# coeffiction check
print(f"K1 Coefficient: {K1:.6e}")
print(f"K2 Coefficient: {K2:.6e}")


# ==========================================
# 2. PHYSICS FUNCTIONS (Separated for Clarity)
# ==========================================

def calculate_complex_pressure(point, trans_pos, trans_zaxis, phases):
    """
    Calculates the total complex pressure P at a given 3D point.
    Implements Equations 1 and 2 from the reference.
    """
    vec_to_point = point.unsqueeze(1) - trans_pos.unsqueeze(0)
    d = torch.linalg.norm(vec_to_point, dim=2)
    d = torch.where(d < 1e-5, torch.tensor(1e-5, device=point.device), d)
    
    dir_to_point = vec_to_point / d.unsqueeze(2)
    cos_theta = torch.sum(trans_zaxis.unsqueeze(0) * dir_to_point, dim=2)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    
    ka_sin = WAVENUMBER * TRANSDUCER_RADIUS * torch.sin(theta)
    j1_val = torch.special.bessel_j1(ka_sin)
    safe_ka_sin = torch.where(ka_sin < 1e-5, torch.tensor(1e-5, device=point.device), ka_sin)
    df = torch.where(ka_sin < 1e-5, torch.tensor(1.0, device=point.device), 2 * j1_val / safe_ka_sin)
    
    amplitude = P0_A * (df / d)
    phase = phases.unsqueeze(0) + WAVENUMBER * d
    phase_term = torch.complex(torch.cos(phase), torch.sin(phase))
    p_j = amplitude.to(torch.complex64) * phase_term
    
    return torch.sum(p_j, dim=1)


def calculate_gorkov_potential(point, trans_pos, trans_zaxis, phases):
    """
    Calculates the Gor'kov potential U.
    Returns the PyTorch TENSOR (not a NumPy array) so the graph is preserved for the force calculation.
    """
    # 1. Calculate pressure at the center point
    p_center = calculate_complex_pressure(point, trans_pos, trans_zaxis, phases)
    
    p_real = p_center.real
    p_imag = p_center.imag
    mag_p_sq = p_real**2 + p_imag**2
    
    # 2. Calculate spatial derivatives (p_x, p_y, p_z) using Autodiff
    grad_outputs = torch.ones_like(p_real)
    p_x_y_z_real = torch.autograd.grad(p_real, point, grad_outputs=grad_outputs, create_graph=True)[0]
    p_x_y_z_imag = torch.autograd.grad(p_imag, point, grad_outputs=grad_outputs, create_graph=True)[0]
    
    mag_nabla_p_sq = torch.sum(p_x_y_z_real**2, dim=1) + torch.sum(p_x_y_z_imag**2, dim=1)
    
    # Equation 5: Gor'kov Potential U
    U = 2 * K1 * mag_p_sq - 2 * K2 * mag_nabla_p_sq
    
    # RETURN THE TENSOR, NOT A NUMPY ARRAY. The graph must stay attached!
    return U


def calculate_arf(point, U_tensor):
    """
    Calculates the Acoustic Radiation Force (F_rad).
    Takes the already-computed U_tensor to avoid recalculating pressure.
    """
    # Equation 3: F = - Gradient(U)
    F = -torch.autograd.grad(U_tensor, point, grad_outputs=torch.ones_like(U_tensor))[0]
    
    return F


# ==========================================
# 3. HOW TO CALL THEM TOGETHER
# ==========================================
def get_physics_outputs(point_numpy, trans_pos, trans_zaxis, phases):
    """Wrapper to handle the PyTorch to NumPy conversion safely."""
    # Convert numpy input to PyTorch tensor and track gradients
    point = torch.tensor(point_numpy, dtype=torch.float32, requires_grad=True)
    
    # Step 1: Calculate U (keeps the graph attached)
    U_tensor = calculate_gorkov_potential(point, trans_pos, trans_zaxis, phases)
    
    # Step 2: Calculate F using the U_tensor 
    F_tensor = calculate_arf(point, U_tensor)
    
    # Step 3: Safely detach both back into pure NumPy arrays for MuJoCo
    return U_tensor.detach().numpy(), F_tensor.detach().numpy()

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
