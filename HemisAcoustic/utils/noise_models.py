# HemisAcoustic/utils/noise_models.py

import math
from collections import deque
import gymnasium as gym
import numpy as np
import torch


class KinematicNoiseWrapper(gym.Wrapper):
    """
    Domain Randomization wrapper for TrainEnvSingle (operates in physical mm / mm/s units).
    Place directly around TrainEnvSingle, BEFORE SACAcousticWrapper or VecNormalize.
    """
    def __init__(self, env, noise_cfg=None):
        super().__init__(env)
        cfg = noise_cfg or {}

        self.enabled = bool(cfg.get("enabled", False))
        self.ideal_prob = float(cfg.get("ideal_prob", 0.2))          # Fraction of pure-ideal episodes
        self.max_obs_delay = int(cfg.get("max_obs_delay", 2))        # Steps (2 steps = 40ms @ dt=0.02s)
        self.pos_noise_mm = float(cfg.get("pos_noise_mm", 0.3))      # Camera position std (mm)
        self.vel_noise_mms = float(cfg.get("vel_noise_mms", 2.0))    # Estimated velocity std (mm/s)
        self.max_action_lag = float(cfg.get("max_action_lag", 0.35)) # 1st-order inertial lag beta in [0, 1)
        self.action_noise_std = float(cfg.get("action_noise_std", 0.05))  # Proxy for trap jitter

        # Episode-level randomized parameters
        self.ep_delay = 0
        self.ep_pos_std = 0.0
        self.ep_vel_std = 0.0
        self.ep_lag = 0.0
        self.ep_act_std = 0.0

        self.obs_queue = deque(maxlen=max(1, self.max_obs_delay + 1))
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _sample_episode_noise(self):
        """Sample randomized noise intensities for the new episode (zero-centered DR)."""
        if (not self.enabled) or (self.np_random.random() < self.ideal_prob):
            self.ep_delay = 0
            self.ep_pos_std = 0.0
            self.ep_vel_std = 0.0
            self.ep_lag = 0.0
            self.ep_act_std = 0.0
        else:
            self.ep_delay = int(self.np_random.integers(0, self.max_obs_delay + 1))
            self.ep_pos_std = float(self.np_random.uniform(0.0, self.pos_noise_mm))
            self.ep_vel_std = float(self.np_random.uniform(0.0, self.vel_noise_mms))
            self.ep_lag = float(self.np_random.uniform(0.0, self.max_action_lag))
            self.ep_act_std = float(self.np_random.uniform(0.0, self.action_noise_std))

    def _corrupt_obs(self, obs):
        """
        Apply physically consistent sensor noise to [x, y, z, vx, vy, vz, dx, dy, dz] (mm).
        Because dx = target_x - x, a +err measurement error on x causes a -err error on dx.
        """
        noisy = obs.copy()
        if self.ep_pos_std > 0.0:
            pos_err = self.np_random.normal(0.0, self.ep_pos_std, size=3).astype(np.float32)
            noisy[0:3] += pos_err
            noisy[6:9] -= pos_err
        if self.ep_vel_std > 0.0:
            vel_err = self.np_random.normal(0.0, self.ep_vel_std, size=3).astype(np.float32)
            noisy[3:6] += vel_err
        return noisy

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sample_episode_noise()
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        noisy_obs = self._corrupt_obs(obs)
        self.obs_queue.clear()
        for _ in range(self.obs_queue.maxlen):
            self.obs_queue.append(noisy_obs)

        return self.obs_queue[-1 - self.ep_delay].copy(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # 1. Inertial lag + execution jitter (models trap lag & phase/amp disturbance)
        eff_action = (1.0 - self.ep_lag) * action + self.ep_lag * self.prev_action
        if self.ep_act_std > 0.0:
            eff_action = eff_action + self.np_random.normal(0.0, self.ep_act_std, size=action.shape).astype(np.float32)
        eff_action = np.clip(eff_action, self.action_space.low, self.action_space.high)
        self.prev_action = eff_action

        # 2. Advance underlying ideal kinematics
        obs, reward, terminated, truncated, info = self.env.step(eff_action)

        # 3. Sensor noise + observation latency queue
        noisy_obs = self._corrupt_obs(obs)
        self.obs_queue.append(noisy_obs)
        delayed_obs = self.obs_queue[-1 - self.ep_delay].copy()

        return delayed_obs, reward, terminated, truncated, info


class TransducerNoiseModel:
    """
    Hardware perturbation model for the 79-transducer ultrasonic phased array.
    Injects static calibration tolerance, dynamic phase/amplitude fluctuation,
    discrete MCU/FPGA phase quantization, and optional dead transducers.
    """
    def __init__(
        self,
        n_transducers=79,
        static_phase_std_rad=0.08,    # ~4.5 deg fixed per-transducer manufacturing offset
        dynamic_phase_std_rad=0.04,   # ~2.3 deg step-to-step clock/thermal jitter
        static_amp_std=0.08,          # +/- 8% fixed per-transducer gain mismatch
        dynamic_amp_std=0.03,         # +/- 3% voltage ripple per control update
        phase_bits=None,              # e.g., 5 for 5-bit (11.25 deg) quantization, None = continuous
        dropout_prob=0.0,             # Probability a transducer fails/is dead (e.g., 0.02)
        device="cpu",
    ):
        self.n = n_transducers
        self.static_phase_std_rad = static_phase_std_rad
        self.dynamic_phase_std_rad = dynamic_phase_std_rad
        self.static_amp_std = static_amp_std
        self.dynamic_amp_std = dynamic_amp_std
        self.phase_bits = phase_bits
        self.dropout_prob = dropout_prob
        self.device = torch.device(device)
        self.reset()

    def reset(self, seed=None):
        """Re-sample fixed per-transducer hardware tolerances for a new simulation trial."""
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)

        self.static_phase_bias = torch.randn((self.n, 1), generator=gen, device=self.device) * self.static_phase_std_rad
        self.static_amp_gain = 1.0 + torch.randn((self.n, 1), generator=gen, device=self.device) * self.static_amp_std

        if self.dropout_prob > 0.0:
            self.alive_mask = (torch.rand((self.n, 1), generator=gen, device=self.device) >= self.dropout_prob).float()
        else:
            self.alive_mask = torch.ones((self.n, 1), device=self.device)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Perturb complex transducer activations x of shape (79, 1)."""
        phase = torch.angle(x) + self.static_phase_bias
        if self.dynamic_phase_std_rad > 0.0:
            phase = phase + torch.randn_like(phase) * self.dynamic_phase_std_rad

        if self.phase_bits is not None:
            step = 2.0 * math.pi / (2 ** self.phase_bits)
            phase = torch.round(phase / step) * step

        amp_gain = self.static_amp_gain
        if self.dynamic_amp_std > 0.0:
            amp_gain = amp_gain + torch.randn_like(amp_gain) * self.dynamic_amp_std

        amp = torch.abs(x) * torch.clamp(amp_gain, min=0.0) * self.alive_mask
        return (amp * torch.exp(1j * phase)).to(torch.complex64)
