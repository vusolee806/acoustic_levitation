"""
Acoustic physics backend built on `levitate`
(https://github.com/AppliedAcousticsChalmers/levitate), replacing the
hand-rolled PyTorch/autograd Gor'kov implementation in
`acoustic_physics_pytorch.py`.

Why switch
----------
`levitate` already implements, and has been validated against real
transducer-array hardware for years:
  - the exact same physics as your code (piston/Bessel-J1 directivity,
    Gor'kov potential, radiation force = -grad(U)),
  - phase focusing (`array.focus_phases`), so you no longer hand-roll the
    "number of cycles -> phase" arithmetic,
  - a *calibrated* transducer amplitude (p0 = 6 Pa @ 1 m @ full drive,
    taken from measurements of the Murata MA40S4S -- the exact transducer
    used in the AcoustoReinforce/AcoMan papers you're following). Your
    original P0_A = 40 constant was an arbitrary number not tied to any
    real transducer output, which is why you needed `np.clip(force, -1, 1)`
    to keep the sim stable -- the forces were off by orders of magnitude.

Install
-------
    pip install levitate "scipy<1.15" --break-system-packages

`levitate` 3.0.0 still imports `scipy.special.sph_harm`, which SciPy
deprecated in 1.15 and removed in 1.17. Until levitate publishes a fix,
pin scipy<1.15 (a venv is the cleanest way to keep this isolated from
other projects that need a newer scipy).
"""

import numpy as np
import levitate

# ==========================================
# 1. ACOUSTIC & MATERIAL CONSTANTS
#    (kept numerically identical to your original script)
# ==========================================
C_0 = 343.0            # Speed of sound in air (m/s)
RHO_0 = 1.225          # Density of air (kg/m^3)

C_1 = 1200.0           # Speed of sound in EPS foam (m/s)
RHO_1 = 40.0           # Density of EPS foam (kg/m^3)

PARTICLE_RADIUS = 0.002
PARTICLE_VOL = (4 / 3) * np.pi * (PARTICLE_RADIUS ** 3)

TARGET_WAVELENGTH = 0.008575          # -> 40 kHz in air
FREQ = C_0 / TARGET_WAVELENGTH
OMEGA = 2 * np.pi * FREQ

TRANSDUCER_RADIUS = 0.01              # matches the XML cylinder geom radius (10 mm)

AIR = levitate.materials.Material(rho=RHO_0, c=C_0)
PARTICLE_MATERIAL = levitate.materials.Material(rho=RHO_1, c=C_1)


# ==========================================
# 2. FIELD WRAPPER
# ==========================================
class AcousticField:
    """
    Wraps a `levitate.arrays.TransducerArray` plus the Gor'kov potential
    and radiation-force fields, and exposes the same shape of API your
    MuJoCo driver script already uses (`generate_twin_trap_phases`,
    `get_physics_outputs`), so the rest of your simulation loop barely
    has to change.

    Parameters
    ----------
    trans_pos : (N, 3) array
        Transducer positions, WORLD frame (see note in twin_trap_levitate.py
        about pulling these from `data.site_xpos`, not `model.site_pos`).
    trans_zaxis : (N, 3) array
        Transducer look directions (unit normals), WORLD frame.
    """

    def __init__(self, trans_pos, trans_zaxis, freq=FREQ,
                 transducer_radius=TRANSDUCER_RADIUS,
                 particle_radius=PARTICLE_RADIUS,
                 particle_material=PARTICLE_MATERIAL):
        trans_pos = np.asarray(trans_pos, dtype=float)
        trans_zaxis = np.asarray(trans_zaxis, dtype=float)
        if trans_pos.shape[1] != 3 or trans_zaxis.shape != trans_pos.shape:
            raise ValueError("trans_pos/trans_zaxis must both be (N, 3)")

        self.array = levitate.arrays.TransducerArray(
            positions=trans_pos.T,      # levitate wants (3, N)
            normals=trans_zaxis.T,
            transducer=levitate.transducers.CircularPiston,
            effective_radius=transducer_radius,
            freq=freq,
            medium=AIR,
        )
        self.gorkov = levitate.fields.GorkovPotential(
            self.array, radius=particle_radius, material=particle_material)
        self.radiation_force = levitate.fields.RadiationForce(
            self.array, radius=particle_radius, material=particle_material)

        self._trans_x = trans_pos[:, 0]
        self.num_transducers = trans_pos.shape[0]

    # ---- phase generation -------------------------------------------------
    def generate_twin_trap_phases(self, focal_point):
        """
        Phase delays that steer a twin trap to `focal_point`.

        Equivalent to your original `generate_twin_trap_phases`, but the
        focusing math (distance -> phase) is delegated to
        `array.focus_phases`, which is the same "number of wavelengths,
        keep the fractional part" calculation from Eq. 8-9 of the AcoMan
        paper -- just implemented once, tested, and vectorized in C/NumPy
        rather than re-derived by hand.
        """
        focal_point = np.asarray(focal_point, dtype=float).reshape(3)
        phases = self.array.focus_phases(focal_point).copy()
        # Twin-trap signature: pi-phase shift on one geometric half of the
        # array, exactly as in your code / Section 2.3 of the AcoMan paper.
        phases[self._trans_x > 0] += np.pi
        return phases

    # ---- field evaluation ---------------------------------------------
    def complex_amplitudes(self, phases, amplitudes=None):
        """Combine per-transducer amplitude (0-1, default full drive) and
        phase into the complex driving signal `levitate` expects."""
        if amplitudes is None:
            amplitudes = np.ones(self.num_transducers)
        return amplitudes * np.exp(1j * np.asarray(phases))

    def evaluate(self, points, complex_amplitudes):
        """
        points : (3,) or (3, M) array of world-frame position(s)
        Returns (U, F) with
            U : scalar or (M,)      Gor'kov potential [J]
            F : (3,) or (M, 3)      Acoustic radiation force [N]
        """
        points = np.asarray(points, dtype=float)
        single = points.ndim == 1
        pts = points.reshape(3, 1) if single else points

        U = self.gorkov(complex_amplitudes, pts)
        F = self.radiation_force(complex_amplitudes, pts)  # shape (3, M)

        if single:
            return float(U), np.asarray(F).reshape(3)
        return np.asarray(U), np.asarray(F).T  # -> (M,), (M, 3)

    def find_stable_point(self, start, complex_amplitudes, tolerance=1e-5):
        """
        Locate the *actual* zero-force trap near `start`.

        The geometric focus (where `focus_phases` points) is where the
        pressure/directivity is maximal -- it is NOT generally where the
        particle will sit once gravity is included. `levitate.analysis
        .find_trap` follows the force field from `start` to the nearby
        zero, which is what you want to compare against in a physical
        (gravity-loaded) simulation.
        """
        return levitate.analysis.find_trap(
            self.array, complex_amplitudes, np.asarray(start, dtype=float),
            tolerance=tolerance)


# ==========================================
# 3. SELF-TEST / SANITY PLOT
#    (same z-axis scan as your original __main__, now against a real,
#    calibrated 4-transducer toy array)
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    trans_pos = np.array([
        [0.05, 0.05, 0.0],
        [-0.05, 0.05, 0.0],
        [-0.05, -0.05, 0.0],
        [0.05, -0.05, 0.0],
    ])
    trans_zaxis = np.array([[0.0, 0.0, 1.0]] * 4)

    field = AcousticField(trans_pos, trans_zaxis)

    phases = field.generate_twin_trap_phases([0.0, 0.0, 0.10])
    ca = field.complex_amplitudes(phases)

    z_vals = np.linspace(0.05, 0.15, 500)
    pts = np.zeros((3, z_vals.size))
    pts[2] = z_vals

    U_array, forces_array = field.evaluate(pts, ca)
    F_z = forces_array[:, 2]

    gravity_force = -(PARTICLE_VOL * RHO_1) * 9.81

    z_vals_mm = z_vals * 1000
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(z_vals_mm, U_array, color="blue", linewidth=2)
    ax1.set_ylabel("Gor'kov Potential U (Joules)", color="blue", fontweight="bold")
    ax1.set_title("Acoustic Levitation Field Along Z-Axis (X=0, Y=0) -- levitate backend", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(z_vals_mm, F_z, color="red", linewidth=2, label="Acoustic Z-Force")
    ax2.axhline(0, color="black", linewidth=1.5, label="Zero Force")
    ax2.axhline(gravity_force, color="green", linewidth=1.5, linestyle="--",
                label=f"Gravity ({gravity_force:.2e} N)")
    ax2.set_xlabel("Z-Axis Height (mm)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Z-Force (Newtons)", color="red", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    for i in range(1, len(z_vals)):
        if F_z[i - 1] > 0 and F_z[i] < 0:
            ax1.axvline(z_vals_mm[i], color="purple", linestyle=":", alpha=0.5)
            ax2.axvline(z_vals_mm[i], color="purple", linestyle=":", alpha=0.5)
            ax2.plot(z_vals_mm[i], 0, "mo", markersize=8)

    plt.tight_layout()
    plt.savefig("levitate_zscan_sanity_check.png", dpi=150)
    print("Saved levitate_zscan_sanity_check.png")
