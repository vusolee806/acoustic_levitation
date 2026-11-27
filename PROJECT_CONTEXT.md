# HemisAcoustic: Project Context (for AI agents)

Read this file first. It describes the code as it exists now. Every fact here
was checked against the source on 2026-11-26 unless marked "(unverified)".
If code and this file disagree, the code wins; then fix this file.

Workspace root: `D:/WORKSPACE/ALL_MY_PRJ/ACOUSTIC LEVIATION/acoustic_leviation/acoustic_levitation_project`
Platform: Windows 11, PowerShell. Python package name: `HemisAcoustic` (editable install via `setup.py`).

## 1. Goal

Train a reinforcement-learning agent that plans a path for ONE particle inside
a hemispherical (bowl) acoustic levitator with 79 ultrasonic transducers
(40 kHz). The agent moves the particle from the workspace center to a random
target. The planned path is then replayed in MuJoCo, where an acoustic trap
(Gor'kov potential minimum) is moved along the path and drags a real foam ball
with radiation force.

Two separate layers, keep them apart:

| Layer | What it does | Uses physics? |
|---|---|---|
| RL planning env (`TrainEnvSingle`) | Pure kinematics: action = velocity, position += v*dt | No |
| MuJoCo replay (`simulate_mujoco_*.py`) | Moves trap along RL path, applies Gor'kov force to ball | Yes |

The RL agent never sees acoustic forces. Whether the ball can follow the path is
only checked in the replay scripts.

## 2. Current status

- Algorithms present: PPO (older, baseline), SAC (current main), DDPG (new, untracked in git).
- Single particle only. Multi-particle is out of scope for now.
- Uncommitted work (git status): modified `train_env_single.py`,
  `train_sac_single.yaml`, `train_single.yaml`, `evaluate_sac.py`,
  `simulate_mujoco_sac.py`, `train_sac.py`. Untracked: `noise_models.py`,
  `train_ddpg_single.yaml`, `train_ddpg.py`, `evaluate_ddpg.py`, `simulate_mujoco_ddpg.py`.
- `HemisAcoustic/utils/noise_models.py` is not imported by any script in
  `scripts/` (not checked inside `HemisAcoustic/`). It is written but not wired in.
- `tests/` is empty. There is no `requirements.txt`.

## 3. Repository map (live files only)

```
HemisAcoustic/
  assets/
    mujoco_levitator.xml     MuJoCo model: bowl, bodies sensor_1..sensor_79, body test_ball
    constants.py             Physical constants; parses the XML for radii
    bowl.stl / bowl.obj, create_bowl.py, convert_to_stl.py   mesh tooling
  envs/
    train_env_single.py      TrainEnvSingle (gymnasium env, kinematic)
    sac_wrapper.py           SACAcousticWrapper (obs scaling + reward reshaping)
  particles/
    particle_slim.py         Particle (state holder)
    target_slim.py           Target (x, y, z holder)
    __init__.py              exports Particle as `particle_slim`, Target as `target_slim`
  physics/
    acoustic_force_hemi.py   HemisphereAcousticSolver (phases, Gor'kov U, force)
  utils/
    create_points.py         create_points_hemisphere (start/target sampling)
    noise_models.py          KinematicNoiseWrapper, TransducerNoiseModel (unused)
    optimal_pairing.py       (not read; likely for multi-particle)
configs/
  train_single.yaml          PPO
  train_sac_single.yaml      SAC
  train_ddpg_single.yaml     DDPG
  acoustic.yaml, config.yaml (not read)
scripts/
  train_single.py, evaluate_single.py, simulate_mujoco_ppo.py     PPO
  train_sac.py,    evaluate_sac.py,    simulate_mujoco_sac.py     SAC
  train_ddpg.py,   evaluate_ddpg.py,   simulate_mujoco_ddpg.py    DDPG
  check_particles.py, visualize_heatmap_moving.py                 utilities
  OldCode/                   Legacy scripts. Ignore unless asked.
logs/                        Git-ignored. checkpoints/, tensorboard/, plots
```

Older versions of `README.md` and `PROJECT_CONTEXT.md` described a `src/` + `assets/`
layout and a `levitate`-library workflow that no longer matches the repo. Both files
were rewritten on 2026-11-26. Ignore any copy of the old text found elsewhere (for
example in git history or the `HemisAcoustic.egg-info/` metadata).

## 4. RL environment: `TrainEnvSingle` (`HemisAcoustic/envs/train_env_single.py`)

Workspace is an ellipsoid centered at `(0, 0, z_center)`:
`(x/radius)^2 + (y/radius)^2 + ((z - z_center)/z_radius)^2 <= 1`.
Defaults in all configs: radius 0.03 m, z_radius 0.015 m, z_center 0.0.

- Action: `Box(-1, 1, shape=(3,))`. Scaled to velocity: `v = action * max_velocity`.
- Observation (9 floats, units mm and mm/s, NOT meters):
  `[x, y, z, vx, vy, vz, dx, dy, dz]` where `d = target - particle`.
- Dynamics: `pos += v * delta_time` (no inertia, no acoustic force).
- `reset()`: `create_points_hemisphere(N=2, fixed_center_start=True)`.
  Particle always starts at `(0, 0, z_center)`. Target is random inside the ellipsoid.
- Termination: out of ellipsoid, or distance to target <= `target_radius`.
  Truncation: `current_step >= max_steps`.
- `info`: `particle_pos` (m), `is_success` (bool, distance <= target_radius).
- Raw reward (used by PPO and by evaluation with `train_reward_shaping=False`):
  `150 * (prev_dist - dist) - 0.01 - 0.005*|action|`, then `-50` if out of bounds,
  `+100` if target reached.

Gotcha: `step()` stores the new distance in `particle.last_timestep_dist`.
`SACAcousticWrapper.step()` reads that field after the inner step, so
`prev_dist` is read before and `curr_dist` after the call. Keep this order.

## 5. Wrapper: `SACAcousticWrapper` (`HemisAcoustic/envs/sac_wrapper.py`)

Used by SAC and DDPG (both off-policy, so no `VecNormalize`).

- Observation scaling by fixed divisors, then clip to [-2, 2]:
  `[r, r, zr, vmax, vmax, vmax, 2r, 2r, 2zr]` in mm (and mm/s). Derived from the env config.
- `train_reward_shaping=True` (training env) replaces the reward with:
  `progress_mm - 0.05*dist_mm - 0.02*|action|`, then `+100` on success or `-30` on
  out-of-bounds termination.
- `train_reward_shaping=False` (eval env, simulate, evaluate scripts) passes the
  raw env reward through. Eval rewards and train rewards are therefore on
  different scales. Compare success rate, not reward, across the two.

## 6. Configs (all in `configs/`)

| Key | PPO `train_single.yaml` | SAC `train_sac_single.yaml` | DDPG `train_ddpg_single.yaml` |
|---|---|---|---|
| `max_velocity` (m/s) | 0.05 | 0.05 | 0.02 |
| `delta_time` (s) | 0.02 | 0.02 | 0.02 |
| `max_steps` | 200 | 200 | 200 |
| `target_radius` (m) | 0.001 | 0.001 | 0.001 |
| `total_timesteps` | 500000 | 150000 | 200000 |
| `save_path` | `logs/checkpoints/ppo_single_particle` | `.../sac_single_particle` | `.../ddpg_single_particle` |
| `model_version` | (auto-increment) | `_v2` | `_v2` |

SAC extras: lr 3e-4, buffer 100000, learning_starts 2000, batch 256, tau 0.005,
gamma 0.98, `ent_coef: "auto_0.1"`, net_arch [256, 256], eval every 10000 steps
with 30 episodes. DDPG uses the same except `exploration_noise: 0.1` (Gaussian
action noise sigma) instead of `ent_coef`.

Max step length is `max_velocity * delta_time`: 1 mm (PPO, SAC) and 0.4 mm (DDPG).
With `max_steps=200` the maximum path length is 200 mm (SAC, PPO) or 80 mm (DDPG).
The farthest target from the start is 30 mm (xy edge of the ellipsoid), so both fit,
but DDPG has less slack. The comment "max step = 1 mm/step" in `train_ddpg_single.yaml`
is wrong for its 0.02 m/s setting (it is 0.4 mm/step).

## 7. Start/target sampling (`HemisAcoustic/utils/create_points.py`)

`create_points_hemisphere(N, radius, z_radius, z_center, max_attempts=2000, fixed_center_start=True)`:
first point is `(0, 0, z_center)`, the rest are rejection-sampled inside the ellipsoid.
A new point is accepted only if it lies outside an ellipsoid of semi-axes
`(0.015, 0.015, 0.03)` m around every earlier point. For the target this means an
xy distance of about 13 to 15 mm or more from the start (13 mm when the target is at
the maximum height offset, 15 mm at the same height), so the shortest trips are about 13 mm.
Raises `RuntimeError` if it cannot place all points.

`create_points_multistage_hemisphere` is for future multi-particle work. It calls
`create_points_hemisphere(N, radius, z_center, max_attempts)` with positional
arguments that no longer match the signature (`z_center` lands in `z_radius`).
Treat it as broken and unused.

## 8. Acoustic solver (`HemisAcoustic/physics/acoustic_force_hemi.py`)

`HemisphereAcousticSolver(mj_model, freq=40000, c0=343, rho0=1.225, rho_p=40, c_p=1150, r_p=0.0018, p0=12)`.
Scripts construct it with `p0=40.0`.

- Reads the 79 transducer positions and emission directions from `body_pos` and
  `body_quat` of bodies `sensor_1..sensor_79` in the XML (local z-axis = emission).
- `piston_model_hemi(points)`: directed piston transfer matrix `(M, 79)`.
- `solve_phases_twin_trap(trap_pos)`: conjugate-phase focus at `trap_pos`, plus a
  pi phase shift on transducers with `x < 0` (twin-trap signature).
- `compute_gorkov_potential(pts, x)`: Gor'kov U via 7-point finite-difference stencil.
- `compute_radiation_force(particle_pos, trap_pos)`: `F = -grad U` by central differences.

Parameter mismatch to be aware of: the solver defaults use `c_p=1150`, `r_p=0.0018`,
while `assets/constants.py` uses `C_1=1200` and reads the particle radius from the XML
(`test_ball`, about 2 mm per the old notes, unverified). Change one set only with intent.

## 9. MuJoCo model and replay (`scripts/simulate_mujoco_{ppo,sac,ddpg}.py`)

Details below come from `simulate_mujoco_sac.py`; the PPO and DDPG versions were not read.

- XML: `HemisAcoustic/assets/mujoco_levitator.xml`, `timestep=0.0001`,
  79 transducer bodies, one free-joint `test_ball`.
- Replay flow: roll out the trained policy to get waypoints (20 ms apart) ->
  cubic-interpolate to 5 ms (`smooth_and_interpolate_path`) -> for each trap
  position run 10 MuJoCo substeps, each applying `F_rad + F_drag` to the ball via
  `xfrc_applied`, with `F_drag = -0.00015 * v`.
- Hold 100 steps at the start and 150 at the end of the path.
- `CUSTOM_TARGET` (meters) is hard-coded inside `main()`. Set it to `None` for a
  random target. The script warns if the target is outside the ellipsoid.
- Viewer colors: red = target, green = trap, blue dots = planned path, yellow = actual ball path.
- Opens a MuJoCo viewer and a matplotlib window, so it needs a display.

## 10. Noise models (`HemisAcoustic/utils/noise_models.py`, not yet used)

- `KinematicNoiseWrapper(env, noise_cfg)`: gym wrapper for domain randomization.
  Per episode it samples observation delay, position/velocity noise, action lag
  and action jitter. `ideal_prob` (default 0.2) of episodes stay noise-free.
  Intended order: `TrainEnvSingle` -> `KinematicNoiseWrapper` -> `SACAcousticWrapper`
  (it works in raw mm units, so it must sit before the scaling wrapper).
- `TransducerNoiseModel(n_transducers=79, ...)`: torch model of phase/amplitude
  tolerance, phase quantization and dead transducers. `apply(x)` takes complex
  activations of shape `(79, 1)`. It is meant for the MuJoCo replay, not for RL training.
- Neither has a config block yet. To use them, add a `noise:` section to a YAML
  and wire the wrapper into `make_env` in the train script.

## 11. How to run

Run from the workspace root, in PowerShell. Scripts use paths relative to their own
file, so the working directory only matters for `logs/`.

```powershell
pip install -e .                       # once; installs HemisAcoustic

# SAC (main)
python scripts/train_sac.py            # reads configs/train_sac_single.yaml
python scripts/evaluate_sac.py         # 20 episodes, prints success rate, 3D plot
python scripts/simulate_mujoco_sac.py  # MuJoCo viewer + comparison plot

# DDPG
python scripts/train_ddpg.py
python scripts/evaluate_ddpg.py
python scripts/simulate_mujoco_ddpg.py

# PPO (baseline)
python scripts/train_single.py
python scripts/evaluate_single.py
python scripts/simulate_mujoco_ppo.py
```

Model files are written to `logs/checkpoints/` (git-ignored):

- SAC and DDPG: `<save_path><model_version>.zip` is the BEST checkpoint by eval
  success rate (ties broken by mean reward). `<...>_final.zip` is the last step.
  Evaluate and simulate scripts load the best one.
- PPO: `train_single.py` auto-increments `_v1, _v2, ...` and saves a
  `_vecnormalize.pkl` next to each model. Evaluate and simulate need both files.
- Changing `model_version` in a YAML overwrites the existing file with that name.

Dependencies (no `requirements.txt` exists; inferred from imports): `mujoco`,
`torch`, `stable-baselines3`, `gymnasium`, `numpy`, `scipy`, `pyyaml`, `matplotlib`.

## 12. Conventions to follow when editing

- Units: env and config positions in meters; observations in mm and mm/s;
  `info["particle_pos"]` in meters.
- Keep SAC, DDPG and PPO configs structurally parallel (`env:` and `train:` blocks).
  When you change an `env:` value for one algorithm, say so, because the
  algorithms are compared against each other.
- Training uses `SACAcousticWrapper(shaping=True)`; evaluation, evaluate scripts and
  simulate scripts use `shaping=False`. Do not mix them up.
- New algorithms: copy the `train_ddpg.py` / `evaluate_ddpg.py` /
  `simulate_mujoco_ddpg.py` trio and a new YAML. Existing code does this by copy,
  not by shared module.
- Comments in the codebase are a mix of Vietnamese and English. Match the file you edit.
- Do not edit `scripts/OldCode/`.

## 13. Known issues and traps

1. `scripts/train_single.py` (PPO) builds a `PPO` model with tuned hyperparameters,
   wraps the env in `VecNormalize`, then builds a SECOND `PPO` with default
   hyperparameters on that env and trains only the second one. The tuned model is
   discarded. This is probably a bug (not fixed; the user has not asked).
2. `create_points_multistage_hemisphere` has a stale call signature (section 7).
3. `README.md` is a short overview. Keep it in sync with this file when scripts or configs change.
4. Solver and constants use different particle parameters (section 8).
5. RL path planning ignores acoustic force limits. A path that RL calls successful
   can still lose the ball in MuJoCo if the trap moves too fast. The replay plots
   "Trap-to-Ball Lag Error" for this reason.
6. `logs/` and checkpoints are git-ignored, so models are not versioned.
7. Python-version note from the old file (3.14, wheel availability) was not
   re-checked and may be outdated.

## 14. Unverified items

I did not read these, so do not rely on them: `configs/acoustic.yaml`,
`configs/config.yaml`, `utils/optimal_pairing.py`, `scripts/check_particles.py`,
`scripts/visualize_heatmap_moving.py`, `scripts/evaluate_ddpg.py`,
`scripts/simulate_mujoco_ddpg.py`, `scripts/simulate_mujoco_ppo.py` (the replay
flow in section 9 is read from the SAC script and assumed to be the same in the
other two), the XML internals beyond `timestep` and body names, and
`HemisAcoustic.egg-info/`.

## 15. User requirements (fill in or tell the agent)

Put current goals here so the agent can fit its work to them. Example format:

- [ ] Goal: ...
- [ ] Constraints (algorithm, config values to keep fixed, files not to touch): ...
- [ ] Definition of done (for example success rate >= X% over N eval episodes): ...

