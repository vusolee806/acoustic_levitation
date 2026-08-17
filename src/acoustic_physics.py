import numpy as np
from scipy.special import j1

# ==========================================
# 1. ACOUSTIC & MATERIAL CONSTANTS
# ==========================================
C_0 = 343.0            # Speed of sound in air (m/s)
RHO_0 = 1.225          # Density of air (kg/m^3)

C_1 = 1200.0           # Speed of sound in EPS (m/s)
RHO_1 = 40.0           # Density of EPS (kg/m^3)

PARTICLE_RADIUS = 0.002 # 2mm radius (scaled down appropriately)
PARTICLE_VOL = (4/3) * np.pi * (PARTICLE_RADIUS**3)

TARGET_WAVELENGTH = 0.008575
FREQ = C_0 / TARGET_WAVELENGTH  # 40 kHz
OMEGA = 2 * np.pi * FREQ
WAVENUMBER = OMEGA / C_0        

TRANSDUCER_RADIUS = 0.01 
P0_A = 5.0             

# Pre-calculate Gor'kov Constants 
K1 = 0.25 * PARTICLE_VOL * ((1 / (C_0**2 * RHO_0)) - (1 / (C_1**2 * RHO_1)))
K2 = 0.75 * PARTICLE_VOL * ((RHO_0 - RHO_1) / (OMEGA**2 * RHO_0 * (RHO_0 + 2 * RHO_1)))

# ==========================================
# 2. PHASE GENERATION (TWIN TRAP STEERING)
# ==========================================
def generate_twin_trap_phases(focal_point, trans_pos):
    """
    Calculates the phase delays to steer the twin trap to a specific focal point.
    Implements the focal phase calculation and pi-phase shift[cite: 5, 6].
    """
    # 1. Calculate distance from each transducer to focal point
    vec_to_focus = focal_point - trans_pos
    d = np.linalg.norm(vec_to_focus, axis=1)
    
    # 2. Number of cycles (N)[cite: 5]
    N = d / TARGET_WAVELENGTH
    
    # 3. Phase delay for focusing: 2 * pi * (1 - fractional_part(N))[cite: 5]
    focus_phases = 2 * np.pi * (1 - (N % 1))
    
    # 4. Apply Twin Trap Signature (pi-phase shift to one half)[cite: 5, 6]
    twin_phases = np.copy(focus_phases)
    # Split the array along the X-axis to form the two halves
    twin_phases[trans_pos[:, 0] > 0] += np.pi
    
    return twin_phases

# ==========================================
# 3. PHYSICS FUNCTIONS
# ==========================================
def calculate_complex_pressure(point, trans_pos, trans_zaxis, phases):
    vec_to_point = point - trans_pos
    d = np.linalg.norm(vec_to_point, axis=1)
    
    d = np.where(d < 1e-5, 1e-5, d)
    dir_to_point = vec_to_point / d[:, np.newaxis]
    
    cos_theta = np.sum(trans_zaxis * dir_to_point, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    ka_sin = WAVENUMBER * TRANSDUCER_RADIUS * np.sin(theta)
    
    # FIX: Safe division to avoid RuntimeWarning when ka_sin -> 0
    safe_ka_sin = np.where(ka_sin < 1e-5, 1e-5, ka_sin)
    df = np.where(ka_sin < 1e-5, 1.0, 2 * j1(safe_ka_sin) / safe_ka_sin)
    
    amplitude = P0_A * (df / d)
    phase_term = np.exp(1j * (phases + WAVENUMBER * d))
    
    p_j = amplitude * phase_term
    
    return np.sum(p_j)

def calculate_gorkov_potential(point, trans_pos, trans_zaxis, phases, eps=1e-4):
    p_center = calculate_complex_pressure(point, trans_pos, trans_zaxis, phases)
    mag_p_sq = np.abs(p_center)**2
    
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
    
    U = 2 * K1 * mag_p_sq - 2 * K2 * mag_nabla_p_sq
    return U

def calculate_arf(point, trans_pos, trans_zaxis, phases, eps=1e-4):
    dx = np.array([eps, 0, 0])
    dy = np.array([0, eps, 0])
    dz = np.array([0, 0, eps])
    
    U_xp = calculate_gorkov_potential(point + dx, trans_pos, trans_zaxis, phases)
    U_xm = calculate_gorkov_potential(point - dx, trans_pos, trans_zaxis, phases)
    
    U_yp = calculate_gorkov_potential(point + dy, trans_pos, trans_zaxis, phases)
    U_ym = calculate_gorkov_potential(point - dy, trans_pos, trans_zaxis, phases)
    
    U_zp = calculate_gorkov_potential(point + dz, trans_pos, trans_zaxis, phases)
    U_zm = calculate_gorkov_potential(point - dz, trans_pos, trans_zaxis, phases)
    
    F_x = -(U_xp - U_xm) / (2 * eps) 
    F_y = -(U_yp - U_ym) / (2 * eps)
    F_z = -(U_zp - U_zm) / (2 * eps)
    
    return np.array([F_x, F_y, F_z])
