# HemisAcoustic/physics/acoustic_force_hemi.py

import math
import torch
import numpy as np
import mujoco


class HemisphereAcousticSolver:
    """
    Computes 79-transducer phases, acoustic pressure |p|, Gor'kov Potential (U),
    and 3D Radiation Force (F = -grad U) using the true transducer site frames
    from mujoco_levitator.xml.
    """

    def __init__(
        self,
        mj_model,
        freq=40000.0,
        c0=343.0,
        rho0=1.225,
        rho_p=40.0,
        c_p=1150.0,
        r_p=0.0018,
        p0=3.0,
    ):
        self.device = torch.device("cpu")
        self.freq = freq
        self.c0 = c0
        self.rho0 = rho0
        self.rho_p = rho_p
        self.r_p = r_p

        self.l = c0 / freq
        self.k = 2.0 * math.pi / self.l
        self.w = 2.0 * math.pi * freq
        self.a = 0.005  # Transducer radius (5 mm)
        self.p0 = p0    # Amplitude constant (Pa * m at 1 m on-axis)
        self.delta = self.l / 64.0  # Spatial finite-difference step (~0.134 mm)

        # Particle mass and gravity for physical verification
        self.vol = (4.0 / 3.0) * math.pi * (r_p ** 3)
        self.mass = rho_p * self.vol
        self.mg = self.mass * 9.81

        # Exact Gor'kov constants k1 (> 0) and k2 (< 0) for polystyrene in air
        self.k1 = (self.vol / 4.0) * (1.0 / (rho0 * c0**2) - 1.0 / (rho_p * c_p**2))
        self.k2 = (3.0 * self.vol / 4.0) * (rho0 - rho_p) / (
            (self.w**2) * rho0 * (2.0 * rho_p + rho0)
        )

        # FIX #1: Read world positions and normals from site_sensor_i (NOT body_quat!)
        tmp_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, tmp_data)

        trans_pos = []
        trans_norm = []
        for i in range(1, 80):
            site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f"site_sensor_{i}")
            if site_id == -1:
                raise ValueError(f"Could not find site_sensor_{i} in MuJoCo XML.")
            pos = tmp_data.site_xpos[site_id].copy()
            xmat = tmp_data.site_xmat[site_id].reshape(3, 3)
            zaxis = xmat[:, 2].copy()
            zaxis = zaxis / (np.linalg.norm(zaxis) + 1e-12)

            trans_pos.append(pos)
            trans_norm.append(zaxis)

        self.trans_pos = torch.tensor(np.array(trans_pos), dtype=torch.float32, device=self.device)
        self.trans_norm = torch.tensor(np.array(trans_norm), dtype=torch.float32, device=self.device)

    def piston_model_hemi(self, points: torch.Tensor) -> torch.Tensor:
        """
        Directed Baffled Piston Model for the 79-transducer hemispherical array.
        points: (M, 3) -> returns complex transmission matrix H of shape (M, 79).
        """
        diff = points.unsqueeze(1) - self.trans_pos.unsqueeze(0)  # (M, 79, 3)
        dist = torch.norm(diff, dim=-1).clamp(min=1e-9)           # (M, 79)

        cos_theta = torch.sum(diff * self.trans_norm.unsqueeze(0), dim=-1) / dist
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=0.0))

        # Exact piston directivity D(theta) = 2 * J1(k*a*sin(theta)) / (k*a*sin(theta))
        bessel_arg = self.k * self.a * sin_theta
        safe_arg = torch.where(bessel_arg > 1e-6, bessel_arg, torch.ones_like(bessel_arg))
        directivity = torch.where(
            bessel_arg > 1e-6,
            2.0 * torch.special.bessel_j1(safe_arg) / safe_arg,
            torch.ones_like(bessel_arg),
        )

        # FIX #2: Baffled piston only radiates into forward hemisphere (cos_theta > 0)
        directivity = torch.where(cos_theta > 0.0, directivity, torch.zeros_like(directivity))

        phase = torch.exp(1j * self.k * dist)
        H = self.p0 * (phase / dist) * directivity
        return H.to(torch.complex64)

    def solve_phases_naive(self, trap_pos: np.ndarray, twin_trap: bool = True) -> torch.Tensor:
        """
        Basic Naive Phase Retrieval (phase conjugation / time reversal).
        - twin_trap=False: pure focal spot (pressure maximum -> repels dense beads)
        - twin_trap=True : adds +pi Twin-Trap signature to x < 0 half of the bowl
                           (creates pressure node flanked by 2 bright tweezers -> 3D trap)
        """
        pt = torch.tensor(trap_pos, dtype=torch.float32, device=self.device).reshape(1, 3)
        H = self.piston_model_hemi(pt)  # (1, 79)

        # Phase conjugation: x_n = conj(H_n) / |H_n| = exp(-j * k * r_n)
        x = torch.conj(H).T
        x = x / (torch.abs(x) + 1e-9)

        if twin_trap:
            signature = torch.where(self.trans_pos[:, 0:1] >= 0.0, 0.0, math.pi)
            x = x * torch.exp(1j * signature).to(torch.complex64)

        return x.to(torch.complex64)

    def solve_phases_twin_trap(self, trap_pos: np.ndarray) -> torch.Tensor:
        """Backwards-compatible alias for Naive + Twin-Trap phase retrieval."""
        return self.solve_phases_naive(trap_pos, twin_trap=True)

    def compute_pressure(self, query_pts: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Returns acoustic pressure magnitude |p| (Pa) at query_pts (M, 3)."""
        H = self.piston_model_hemi(query_pts)
        return torch.abs(torch.matmul(H, x)).squeeze(-1)

    def compute_gorkov_potential(self, query_pts: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Calculates Gor'kov potential U (Joules) at query_pts (M, 3)."""
        d = self.delta
        offsets = torch.tensor(
            [
                [0, 0, 0],
                [-d, 0, 0], [d, 0, 0],
                [0, -d, 0], [0, d, 0],
                [0, 0, -d], [0, 0, d],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        stencil = (query_pts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)
        H_all = self.piston_model_hemi(stencil)
        p_all = torch.matmul(H_all, x).reshape(-1, 7)

        p = p_all[:, 0]
        px = (p_all[:, 2] - p_all[:, 1]) / (2.0 * d)
        py = (p_all[:, 4] - p_all[:, 3]) / (2.0 * d)
        pz = (p_all[:, 6] - p_all[:, 5]) / (2.0 * d)

        U = self.k1 * torch.abs(p) ** 2 + self.k2 * (
            torch.abs(px) ** 2 + torch.abs(py) ** 2 + torch.abs(pz) ** 2
        )
        return U

    def compute_radiation_force_from_phases(
        self, particle_pos: np.ndarray, x: torch.Tensor
    ) -> np.ndarray:
        """Computes F_rad = -grad(U) at particle_pos using precomputed phases x (79, 1)."""
        d = self.delta
        p0 = torch.tensor(particle_pos, dtype=torch.float32, device=self.device)
        grad_pts = torch.stack(
            [
                p0 + torch.tensor([-d, 0, 0]), p0 + torch.tensor([d, 0, 0]),
                p0 + torch.tensor([0, -d, 0]), p0 + torch.tensor([0, d, 0]),
                p0 + torch.tensor([0, 0, -d]), p0 + torch.tensor([0, 0, d]),
            ]
        )

        U = self.compute_gorkov_potential(grad_pts, x)
        Fx = -(U[1] - U[0]) / (2.0 * d)
        Fy = -(U[3] - U[2]) / (2.0 * d)
        Fz = -(U[5] - U[4]) / (2.0 * d)
        return np.array([Fx.item(), Fy.item(), Fz.item()], dtype=np.float64)

    def compute_radiation_force(self, particle_pos: np.ndarray, trap_pos: np.ndarray) -> np.ndarray:
        """Computes F_rad = -grad(U) at particle_pos for a Twin-Trap focused at trap_pos."""
        x = self.solve_phases_twin_trap(trap_pos)
        return self.compute_radiation_force_from_phases(particle_pos, x)

    def estimate_max_upward_force(self, trap_pos: np.ndarray, z_span=0.004, n_pts=25):
        """
        Probes F_z slightly below trap_pos to find peak upward lifting force (N).
        A ball can only levitate if max_Fz > mg (and ideally > 1.5*mg for motion).
        """
        x = self.solve_phases_twin_trap(trap_pos)
        z_offsets = np.linspace(-z_span, 0.001, n_pts)
        fz_vals = []
        for dz in z_offsets:
            probe = np.array([trap_pos[0], trap_pos[1], trap_pos[2] + dz], dtype=np.float64)
            fz_vals.append(self.compute_radiation_force_from_phases(probe, x)[2])
        return float(np.max(fz_vals))
