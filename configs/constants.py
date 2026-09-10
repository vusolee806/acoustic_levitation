# configs/constants.py
import os
import numpy as np
import xml.etree.ElementTree as ET

# ==========================================
# 0. DYNAMIC XML PARSING
# ==========================================
# Dynamically locate the XML file in the assets folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(PROJECT_ROOT, "assets", "mujoco_levitator.xml")

try:
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    # Extract transducer radius from sensor_1's geom (size="radius half-height")
    sensor_body = root.find(".//body[@name='sensor_1']")
    sensor_geom = sensor_body.find("geom")
    _sensor_size_str = sensor_geom.get("size")
    extracted_transducer_radius = float(_sensor_size_str.split()[0])

    # Extract particle radius from test_ball's geom
    ball_body = root.find(".//body[@name='test_ball']")
    ball_geom = ball_body.find("geom")
    _ball_size_str = ball_geom.get("size")
    extracted_particle_radius = float(_ball_size_str.split()[0])

except Exception as e:
    print(f"Warning: Could not parse XML for constants. Using default fallbacks. Error: {e}")
    extracted_transducer_radius = 0.010
    extracted_particle_radius = 0.002

# ==========================================
# 1. ACOUSTIC & AIR CONSTANTS
# ==========================================
C_0 = 343.0            # Speed of sound in air (m/s)
RHO_0 = 1.225          # Density of air (kg/m^3)[cite: 32]

# Frequency & Wavelength calculations
TARGET_WAVELENGTH = 0.008575  #[cite: 32]
WAVELENGTH = TARGET_WAVELENGTH

# Calculate Frequency to satisfy the wavelength constraint: f = c / lambda
FREQ = C_0 / TARGET_WAVELENGTH  # Evaluates to 40 kHz[cite: 32]
OMEGA = 2 * np.pi * FREQ  #[cite: 32]
WAVENUMBER = OMEGA / C_0  # k = 2*pi / lambda[cite: 32]

# ==========================================
# 2. PARTICLE CONSTANTS (EPS FOAM)
# ==========================================
C_1 = 1200.0           # Speed of sound in EPS foam (m/s)[cite: 32]
RHO_1 = 40.0           # Density of EPS foam (kg/m^3)[cite: 32]

# Dynamically loaded from XML
PARTICLE_RADIUS = extracted_particle_radius 
PARTICLE_VOL = (4/3) * np.pi * (PARTICLE_RADIUS**3)  #[cite: 32]

# ==========================================
# 3. TRANSDUCER ARRAY CONSTANTS
# ==========================================
# Dynamically loaded from XML
TRANSDUCER_RADIUS = extracted_transducer_radius 
TRANSDUCER_SIZE = TRANSDUCER_RADIUS * 2.0  # e.g., 0.02 (20mm diameter)

COMPLEX_WEIGHT_MULTIPLIER = 0.3 # Global amplitude scaling
P0_A = 40.0                     # Combined constant for (P_0 * A) Output efficiency & amplitude[cite: 32]

# ==========================================
# 4. GOR'KOV POTENTIAL CONSTANTS
# ==========================================
# Pre-calculate Gor'kov Constants (Equations 6 and 7)[cite: 32]
K1 = 0.25 * PARTICLE_VOL * ((1 / (C_0**2 * RHO_0)) - (1 / (C_1**2 * RHO_1)))  #[cite: 32]
K2 = 0.75 * PARTICLE_VOL * ((RHO_0 - RHO_1) / (OMEGA**2 * RHO_0 * (RHO_0 + 2 * RHO_1)))  #[cite: 32]
