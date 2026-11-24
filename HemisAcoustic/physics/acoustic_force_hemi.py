# HemisAcoustic/physics/acoustic_force_hemi.py

import math
import torch
import numpy as np
import mujoco

class HemisphereAcousticSolver:
    """
    Computes 79-transducer phases, Gor'kov Potential (U), and 3D Radiation Force (F = -grad U)
    directly from the transducer positions and normal vectors in mujoco_levitator.xml.
    """
    def __init__(self, mj_model, freq=40000.0, c0=343.0, rho0=1.225, rho_p=40.0, c_p=1150.0, r_p=0.0018, p0=12.0):
        self.device = torch.device("cpu")
        self.l = c0 / freq
        self.k = 2.0 * math.pi / self.l
        self.w = 2.0 * math.pi * freq
        self.a = 0.005  # Transducer radius (5mm)
        self.p0 = p0    # Transducer amplitude factor (tune for trap stiffness)
        self.delta = self.l / 64.0  # Finite difference step for spatial gradient

        # Gor'kov constants k1 and k2 (from Gorkov_new.py)
        vol = (4.0 / 3.0) * math.pi * (r_p ** 3)
        self.k1 = (vol / 4.0) * (1.0 / (rho0 * c0**2) - 1.0 / (rho_p * c_p**2))
        self.k2 = (3.0 * vol / 4.0) * (rho0 - rho_p) / ((self.w**2) * rho0 * (2.0 * rho_p + rho0))

        # Extract 79 transducer positions and normal vectors (zaxis) from MuJoCo XML
        trans_pos = []
        trans_norm = []
        for i in range(1, 80):
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"sensor_{i}")
            pos = mj_model.body_pos[body_id]
            quat = mj_model.body_quat[body_id]
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat)
            zaxis = mat.reshape(3, 3)[:, 2]  # Local Z-axis is the emission direction
            
            trans_pos.append(pos)
            trans_norm.append(zaxis)

        self.trans_pos = torch.tensor(np.array(trans_pos), dtype=torch.float32, device=self.device)   # (79, 3)
        self.trans_norm = torch.tensor(np.array(trans_norm), dtype=torch.float32, device=self.device) # (79, 3)

    def piston_model_hemi(self, points: torch.Tensor) -> torch.Tensor:
        """
        Directed Piston Model for 3D Hemisphere Array.
        points: (M, 3) tensor -> returns Transmission Matrix H of shape (M, 79) complex64
        """
        # diff vector from transducers (79) to points (M): shape (M, 79, 3)
        diff = points.unsqueeze(1) - self.trans_pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1) + 1e-9  # (M, 79)

        # Dot product between diff and transducer normal vector (zaxis)
        cos_theta = torch.sum(diff * self.trans_norm.unsqueeze(0), dim=-1) / dist
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=1e-12))

        # Taylor expansion of Bessel directivity (from Gorkov_new.py)
        bessel_arg = self.k * self.a * sin_theta
        directivity = (0.5 - bessel_arg**2 / 16.0 + bessel_arg**4 / 384.0 
                       - bessel_arg**6 / 18432.0 + bessel_arg**8 / 1474560.0)

        phase = torch.exp(1j * self.k * dist)
        H = 2.0 * self.p0 * (phase / dist) * directivity
        return H.to(torch.complex64)

    def solve_phases_twin_trap(self, trap_pos: np.ndarray) -> torch.Tensor:
        """
        Computes 79 transducer complex activations x (79, 1) to focus at trap_pos
        with a Twin-Trap signature (split along X-axis to create a 3D potential well).
        """
        pt = torch.tensor(trap_pos, dtype=torch.float32, device=self.device).reshape(1, 3)
        H = self.piston_model_hemi(pt)  # (1, 79)
        
        # Conjugate phase retrieval (Naive / 1-step WGS)
        x = torch.conj(H).T
        x = x / (torch.abs(x) + 1e-9)

        # Apply Twin-Trap phase signature (+pi to half the bowl) so the particle is trapped, not pushed out
        signature = torch.where(self.trans_pos[:, 0:1] >= 0, 0.0, math.pi)
        x = x * torch.exp(1j * signature).to(torch.complex64)
        return x

    def compute_gorkov_potential(self, query_pts: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates Gor'kov potential U at query_pts (M, 3) given transducer activations x (79, 1).
        """
        d = self.delta
        offsets = torch.tensor([
            [0, 0, 0],
            [-d, 0, 0], [d, 0, 0],
            [0, -d, 0], [0, d, 0],
            [0, 0, -d], [0, 0, d]
        ], dtype=torch.float32, device=self.device)  # (7, 3)

        # Create stencil of 7 points for each query point: (M*7, 3)
        stencil = (query_pts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)
        H_all = self.piston_model_hemi(stencil)  # (M*7, 79)
        p_all = torch.matmul(H_all, x).reshape(-1, 7)  # (M, 7)

        p = p_all[:, 0]
        px = (p_all[:, 2] - p_all[:, 1]) / (2.0 * d)
        py = (p_all[:, 4] - p_all[:, 3]) / (2.0 * d)
        pz = (p_all[:, 6] - p_all[:, 5]) / (2.0 * d)

        U = self.k1 * torch.abs(p)**2 + self.k2 * (torch.abs(px)**2 + torch.abs(py)**2 + torch.abs(pz)**2)
        return U

    def compute_radiation_force(self, particle_pos: np.ndarray, trap_pos: np.ndarray) -> np.ndarray:
        """
        Computes the 3D Acoustic Radiation Force F_rad = -grad(U) at particle_pos
        when the acoustic trap is focused at trap_pos.
        """
        x = self.solve_phases_twin_trap(trap_pos)
        
        d = self.delta
        p0 = torch.tensor(particle_pos, dtype=torch.float32, device=self.device)
        grad_pts = torch.stack([
            p0 + torch.tensor([-d, 0, 0]), p0 + torch.tensor([d, 0, 0]),
            p0 + torch.tensor([0, -d, 0]), p0 + torch.tensor([0, d, 0]),
            p0 + torch.tensor([0, 0, -d]), p0 + torch.tensor([0, 0, d]),
        ])  # (6, 3)

        U = self.compute_gorkov_potential(grad_pts, x)
        
        # Central difference: F = -dU/dr
        Fx = -(U[1] - U[0]) / (2.0 * d)
        Fy = -(U[3] - U[2]) / (2.0 * d)
        Fz = -(U[5] - U[4]) / (2.0 * d)

        return np.array([Fx.item(), Fy.item(), Fz.item()], dtype=np.float64)
