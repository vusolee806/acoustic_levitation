"""
Gymnasium environment for the 79-transducer acoustic levitator.

Action: continuous 3-D target focal point (x, y, z) in metres.
Observation: 9-D vector [ball_pos (3), ball_vel (3), target_focal (3)].

The env converts the focal-point action into 79 transducer phases via the
existing `generate_twin_trap_phases` function, then computes the acoustic
radiation force on the ball with `get_physics_outputs` (PyTorch autograd
backend), clips it for stability, and applies it as a force on the ball's
translational DOFs in MuJoCo.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .utils import get_asset_path
from .acoustic_physics_pytorch import (
    generate_twin_trap_phases,
    get_physics_outputs,
)


# Workspace / safe-zone bounds (metres)
WORKSPACE_XY = 0.025
WORKSPACE_Z_LO = 0.02
WORKSPACE_Z_HI = 0.08

SAFE_XY = 0.04
SAFE_Z_LO = 0.005
SAFE_Z_HI = 0.10

MAX_VEL = 1.0
MAX_FORCE = 0.5
DEFAULT_MAX_STEPS = 500
DEFAULT_REWARD_SCALE = 50.0
DEFAULT_DRAG = 0.0


class AcousticLevitatorEnv(gym.Env):
    """Gymnasium environment wrapping the 79-transducer MuJoCo levitator.

    The agent picks a continuous 3-D target focal point; the env steers the
    twin-trap phases toward that target and computes the resulting ARF on the
    ball. Reward is the negative distance between the ball and the target,
    with a small alive bonus and an optional quadratic velocity penalty.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        xml_filename: str = "mujoco_levitator.xml",
        max_episode_steps: int = DEFAULT_MAX_STEPS,
        reward_scale: float = DEFAULT_REWARD_SCALE,
        velocity_penalty: float = 0.001,
        alive_bonus: float = 0.01,
        max_force: float = MAX_FORCE,
        drag: float = DEFAULT_DRAG,
        physics_substeps: int = 1,
        device: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # ---------------- MuJoCo model / data ----------------
        xml_path = get_asset_path(xml_filename)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ---- Transducer geometry (extract once) ----
        num_transducers = 79
        transducer_positions = np.zeros((num_transducers, 3), dtype=np.float32)
        transducer_zaxis = np.zeros((num_transducers, 3), dtype=np.float32)
        for i in range(num_transducers):
            site_id = self.model.site(f"site_sensor_{i+1}").id
            transducer_positions[i] = self.model.site_pos[site_id]
            quat = self.model.site_quat[site_id]
            mat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(mat, quat)
            transducer_zaxis[i] = mat.reshape(3, 3)[:, 2].astype(np.float32)

        # Move to torch once. Reuse these tensors on every step.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.trans_pos_torch = torch.tensor(
            transducer_positions, dtype=torch.float32, device=self.device
        )
        self.trans_zaxis_torch = torch.tensor(
            transducer_zaxis, dtype=torch.float32, device=self.device
        )

        # Ball body / dof address (force applies to first 3 translational dofs)
        self.ball_id = self.model.body("test_ball").id
        self.ball_dofadr = int(self.model.body_dofadr[self.ball_id])

        # ---------------- Spaces ----------------
        # Action: target focal point (continuous, Box)
        action_low = np.array(
            [-WORKSPACE_XY, -WORKSPACE_XY, WORKSPACE_Z_LO], dtype=np.float32
        )
        action_high = np.array(
            [WORKSPACE_XY, WORKSPACE_XY, WORKSPACE_Z_HI], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )

        # Observation: ball_pos (3), ball_vel (3), target_focal (3)
        obs_low = np.array(
            [
                -SAFE_XY, -SAFE_XY, SAFE_Z_LO,
                -MAX_VEL, -MAX_VEL, -MAX_VEL,
                action_low[0], action_low[1], action_low[2],
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                SAFE_XY, SAFE_XY, SAFE_Z_HI,
                MAX_VEL, MAX_VEL, MAX_VEL,
                action_high[0], action_high[1], action_high[2],
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ---------------- Hyperparameters ----------------
        self.max_episode_steps = int(max_episode_steps)
        self.reward_scale = float(reward_scale)
        self.velocity_penalty = float(velocity_penalty)
        self.alive_bonus = float(alive_bonus)
        self.max_force = float(max_force)
        self.drag = float(drag)
        self.physics_substeps = max(1, int(physics_substeps))
        self.render_mode = render_mode
        self._viewer = None
        self._step_count = 0
        self._target_focal = np.zeros(3, dtype=np.float32)
        self._np_random = np.random.default_rng()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            torch.manual_seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        # Ball pose: freejoint occupies the first 3 translational + 3 rotational qpos.
        self.data.qpos[0:3] = np.array([0.0, 0.0, 0.05], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity quat (w,x,y,z)
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        # Target focal point: random in workspace, or supplied via options
        if options is not None and "target" in options:
            self._target_focal = np.asarray(
                options["target"], dtype=np.float32
            )
            self._target_focal = np.clip(
                self._target_focal, self.action_space.low, self.action_space.high
            )
        else:
            self._target_focal = self._sample_target()

        self._step_count = 0
        obs = self._get_obs()
        info = {"target_focal": self._target_focal.copy()}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1) sanitize action
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._target_focal = action.copy()

        terminated = False
        truncated = False
        nan_seen = False

        # 2) Physics substeps
        for _ in range(self.physics_substeps):
            arf = self._compute_arf(self.data.xpos[self.ball_id])
            if not np.all(np.isfinite(arf)):
                arf = np.zeros(3, dtype=np.float32)
                nan_seen = True
                terminated = True

            # Apply force + drag on the ball's translational dofs
            self.data.qfrc_applied[self.ball_dofadr:self.ball_dofadr + 3] = arf
            if self.drag > 0.0:
                self.data.qfrc_applied[
                    self.ball_dofadr:self.ball_dofadr + 3
                ] -= self.drag * self.data.qvel[self.ball_dofadr:self.ball_dofadr + 3]

            mujoco.mj_step(self.model, self.data)

            # Workspace escape
            pos = self.data.xpos[self.ball_id]
            if (
                abs(pos[0]) > SAFE_XY
                or abs(pos[1]) > SAFE_XY
                or pos[2] < SAFE_Z_LO
                or pos[2] > SAFE_Z_HI
            ):
                terminated = True
                break

            # qpos / qvel sanity
            if not (
                np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))
            ):
                terminated = True
                nan_seen = True
                break

            if self.render_mode == "human":
                self._sync_viewer()

        # 3) Reward
        pos = self.data.xpos[self.ball_id]
        vel = self.data.qvel[self.ball_dofadr:self.ball_dofadr + 3]
        dist = float(np.linalg.norm(pos - self._target_focal))
        reward = (
            -self.reward_scale * dist
            - self.velocity_penalty * float(np.dot(vel, vel))
            + self.alive_bonus
        )

        # 4) Step counter / truncation
        self._step_count += 1
        if self._step_count >= self.max_episode_steps:
            truncated = True
        if terminated and nan_seen:
            reward = -10.0

        obs = self._get_obs()
        info = {
            "target_focal": self._target_focal.copy(),
            "distance": dist,
            "nan_seen": nan_seen,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            return self._ensure_viewer().sync()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        pos = self.data.xpos[self.ball_id].astype(np.float32)
        vel = self.data.qvel[self.ball_dofadr:self.ball_dofadr + 3].astype(
            np.float32
        )
        return np.concatenate([pos, vel, self._target_focal]).astype(np.float32)

    def _sample_target(self) -> np.ndarray:
        return self._np_random.uniform(
            low=self.action_space.low, high=self.action_space.high
        ).astype(np.float32)

    def _compute_arf(self, ball_pos_world: np.ndarray) -> np.ndarray:
        target_tensor = torch.tensor(
            self._target_focal.reshape(1, 3),
            dtype=torch.float32,
            device=self.device,
        )
        phases = generate_twin_trap_phases(target_tensor, self.trans_pos_torch)

        ball_pos_np = ball_pos_world.reshape(1, 3).astype(np.float32)
        _, forces = get_physics_outputs(
            ball_pos_np, self.trans_pos_torch, self.trans_zaxis_torch, phases
        )
        arf = forces[0].astype(np.float32)
        arf = np.clip(arf, -self.max_force, self.max_force)
        return arf

    # ---- optional viewer ----
    def _ensure_viewer(self):
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def _sync_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()
