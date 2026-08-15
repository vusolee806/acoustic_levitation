import mujoco
import mujoco.viewer
import numpy as np
from scipy.special import j1

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
# C_1 = 343.0           # Speed of sound in air (m/s)
# RHO_1 = 1.225           # Density of air (kg/m^3)


# --- UPDATED PARTICLE & FREQUENCY SETTINGS ---
# Diameter is 0.10, so radius is 0.05
PARTICLE_RADIUS = 0.05 
#with object is the ball
PARTICLE_VOL = (4/3) * np.pi * (PARTICLE_RADIUS**3)

#choose wavelength such that can get the frequency of 40KHZ
TARGET_WAVELENGTH = 0.008575

# Calculate Frequency to satisfy the wavelength constraint: f = c / lambda
FREQ = C_0 / TARGET_WAVELENGTH  # Evaluates to 40 kHz
OMEGA = 2 * np.pi * FREQ
WAVENUMBER = OMEGA / C_0        # k = 2*pi / lambda
# ---------------------------------------------

TRANSDUCER_RADIUS = 0.005 # 'a' in the directivity formula (5mm)
P0_A = 10.0             # Combined constant for (P_0 * A) Output efficiency & amplitude

# Pre-calculate Gor'kov Constants (Equations 6 and 7)
K1 = 0.25 * PARTICLE_VOL * ((1 / (C_0**2 * RHO_0)) - (1 / (C_1**2 * RHO_1)))
K2 = 0.75 * PARTICLE_VOL * ((RHO_0 - RHO_1) / (OMEGA**2 * RHO_0 * (RHO_0 + 2 * RHO_1)))

# coeffiction check
print(f"K1 Coefficient: {K1:.6e}")
print(f"K2 Coefficient: {K2:.6e}")

# ==========================================
# 2. PHYSICS FUNCTIONS
# ==========================================
def calculate_complex_pressure(point, trans_pos, trans_zaxis, phases):
    """
    Calculates the total complex pressure P at a given 3D point.
    Implements Equations 1 and 2 from the reference.
    """
    # Vector from each transducer to the point
    vec_to_point = point - trans_pos
    
    # Distance 'd' to the point
    d = np.linalg.norm(vec_to_point, axis=1)
    
    # Avoid division by zero if point overlaps transducer exactly
    d = np.where(d < 1e-5, 1e-5, d)
#     d = np.clip(d, TRANSDUCER_RADIUS, None)
    # Direction vector to point
    dir_to_point = vec_to_point / d[:, np.newaxis]
    
    # Calculate angle theta using dot product with transducer Z-axis
    cos_theta = np.sum(trans_zaxis * dir_to_point, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Equation 2: Directivity Function D_f(theta)
    ka_sin = WAVENUMBER * TRANSDUCER_RADIUS * np.sin(theta)
    # Handle the limit where theta -> 0 (ka_sin -> 0, D_f -> 1)
    df = np.where(ka_sin < 1e-5, 1.0, 2 * j1(ka_sin) / ka_sin)
    
    # Equation 1: Sum of complex pressures P_j(r)
    # P_j = P_0 * A * (D_f / d) * e^(i(phase + k*d))
    amplitude = P0_A * (df / d)
    phase_term = np.exp(1j * (phases + WAVENUMBER * d))
    
    p_j = amplitude * phase_term
    
    # Return total complex pressure
    return np.sum(p_j)

def calculate_gorkov_potential(point, trans_pos, trans_zaxis, phases, eps=1e-4):
    """
    Calculates the Gor'kov potential U at a given 3D point.
    Implements Equations 4 and 5 using finite differences for pressure gradients.
    """
    # 1. Calculate pressure at the center point
    p_center = calculate_complex_pressure(point, trans_pos, trans_zaxis, phases)
    mag_p_sq = np.abs(p_center)**2
    
    # 2. Calculate spatial derivatives (p_x, p_y, p_z) using central finite difference
    dx = np.array([eps, 0, 0])
    dy = np.array([0, eps, 0])
    dz = np.array([0, 0, eps])
    
    p_x = (calculate_complex_pressure(point + dx, trans_pos, trans_zaxis, phases) - 
           calculate_complex_pressure(point - dx, trans_pos, trans_zaxis, phases)) / (2 * eps)
           
    p_y = (calculate_complex_pressure(point + dy, trans_pos, trans_zaxis, phases) - 
           calculate_complex_pressure(point - dy, trans_pos, trans_zaxis, phases)) / (2 * eps)
           
    p_z = (calculate_complex_pressure(point + dz, trans_pos, trans_zaxis, phases) - 
           calculate_complex_pressure(point - dz, trans_pos, trans_zaxis, phases)) / (2 * eps)
           
    mag_nabla_p_sq = np.abs(p_x)**2 + np.abs(p_y)**2 + np.abs(p_z)**2
    
    # Equation 5: Gor'kov Potential U
    U = 2 * K1 * mag_p_sq - 2 * K2 * mag_nabla_p_sq
    return U

def calculate_arf(point, trans_pos, trans_zaxis, phases, eps=1e-4):
    """
    Calculates the Acoustic Radiation Force vector (F_rad).
    Implements Equation 3 using finite difference on the Gor'kov potential.
    """
    dx = np.array([eps, 0, 0])
    dy = np.array([0, eps, 0])
    dz = np.array([0, 0, eps])
    
    # Calculate U around the point
    U_xp = calculate_gorkov_potential(point + dx, trans_pos, trans_zaxis, phases)
    U_xm = calculate_gorkov_potential(point - dx, trans_pos, trans_zaxis, phases)
    
    U_yp = calculate_gorkov_potential(point + dy, trans_pos, trans_zaxis, phases)
    U_ym = calculate_gorkov_potential(point - dy, trans_pos, trans_zaxis, phases)
    
    U_zp = calculate_gorkov_potential(point + dz, trans_pos, trans_zaxis, phases)
    U_zm = calculate_gorkov_potential(point - dz, trans_pos, trans_zaxis, phases)
    
    # Equation 3: F = - Gradient(U), this applies the central difference formula to calculate
    F_x = -(U_xp - U_xm) / (2 * eps) 
    F_y = -(U_yp - U_ym) / (2 * eps)
    F_z = -(U_zp - U_zm) / (2 * eps)
    
    return np.array([F_x, F_y, F_z])

