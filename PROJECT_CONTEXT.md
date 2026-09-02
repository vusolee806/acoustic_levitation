# Acoustic Levitation Project — Context

> Living reference for the `acoustic_levitation_project` workspace
> (parent dir: `D:/WORKSPACE/ALL_MY_PRJ/ACOUSTIC LEVIATION/acoustic_leviation/`).
> Update this when files, conventions, or the user's role change.

## What this project is

A 79-transducer bowl-shaped acoustic levitator simulated in MuJoCo. The
goal is to learn a policy (RL) that picks a 3-D target focal point so a
single EPS-foam ball (`test_ball`, density 40 kg/m³, radius 2 mm)
hovers and steers toward the requested location.

Two physics backends exist:

- **NumPy / finite differences** — `src/acoustic_physics.py`.
- **PyTorch + autograd** — `src/acoustic_physics_pytorch.py`
  (preferred; gradient is exact, used by the Gymnasium env).

A separate Isaac Sim script (`scripts/issac_acoustic.py`) mirrors the
MuJoCo setup for higher-fidelity visualization.

## Repository layout

```
acoustic_levitation_project/
├── README.md                 # bare-bones; lists the expected tree
├── requirements.txt          # gymnasium 1.2.3, torch 2.12.0, mujoco 3.9.0, ...
├── setup.py                  # installs `src` as a package
├── MUJOCO_LOG.TXT            # historical warnings: Nan/Inf in QACC
├── assets/
│   ├── mujoco_levitator.xml  # MAIN xml — bowl + 79 sites + test_ball freejoint
│   ├── mujoco_array.xml      # older xml (not used by the env)
│   ├── bowl_base.stl         # mesh present but NOT referenced by the XML
│   └── thick_bowl_base.stl   # mesh present but NOT referenced by the XML
├── configs/default_config.yaml
├── scripts/
│   ├── twin_trap.py          # NumPy reference control loop
│   ├── twin_trap_pytorch.py  # PyTorch reference control loop (pattern source)
│   ├── issac_acoustic.py     # Isaac Sim variant
│   ├── test_env_rollout.py   # NEW — random-policy smoke test for AcousticLevitator-v0
│   └── train_td3_levitator.py # NEW — TD3 / TD3_per / DDPG / DDPG_per / MADDPG trainer
├── src/
│   ├── __init__.py           # registers `AcousticLevitator-v0`
│   ├── utils.py              # `get_asset_path(filename)` — resolves XML paths
│   ├── acoustic_physics.py             # NumPy pressure / Gor'kov / ARF
│   ├── acoustic_physics_pytorch.py     # PyTorch version + `get_physics_outputs`
│   └── levitator_env.py       # NEW — `AcousticLevitatorEnv(gym.Env)`
├── examples/
│   ├── heatmap.py
│   ├── test_physics_pytorch.py
│   └── test_sound_wave.py
├── logs/                     # plots, smoke-test outputs
└── check_points/             # saved RL checkpoints
```

The sibling library `../AcoustoRL/` provides the RL algorithms. See
`AcoustoRL/README.md` for the upstream driver pattern.

## MuJoCo model summary (`assets/mujoco_levitator.xml`)

- `timestep=0.0001`, density=1.225, viscosity=1.85e-5 (air-like).
- 79 transducer bodies `sensor_1..sensor_79`, each with `site_sensor_N`
  positioned at the array surface and a cylinder geom. Sites carry
  per-transducer orientation via `site_quat`.
- `test_ball` body: freejoint (6 DOF), sphere size=0.002 m, density=40,
  initial pos `(0, 0, 0.05)`.
- 79 rangefinder sensors along each transducer's z-axis.
- No mesh references in the XML — the STL files on disk are unused.

## Key constants (from `src/acoustic_physics_pytorch.py`)

| Symbol | Value | Meaning |
|---|---|---|
| `C_0` | 343.0 m/s | Speed of sound in air |
| `RHO_0` | 1.225 kg/m³ | Air density |
| `C_1` | 1200.0 m/s | Speed of sound in EPS |
| `RHO_1` | 40.0 kg/m³ | EPS density |
| `PARTICLE_RADIUS` | 0.002 m | Ball radius |
| `FREQ` | 40 kHz | Ultrasonic carrier (λ ≈ 8.575 mm) |
| `TRANSDUCER_RADIUS` | 0.01 m | Piston radius |
| `P0_A` | 40.0 | Amplitude × directivity scale |
| `K1`, `K2` | derived | Gor'kov potential coefficients |

The twin-trap phase generator (`generate_twin_trap_phases`) computes
focusing phases from the distance-to-focal ratio modulo 1, then adds π
to every transducer whose `x > 0` (the twin-trap signature).

## AcousticLevitator-v0 (the Gymnasium env)

Source: `src/levitator_env.py`, registered in `src/__init__.py`.

| Aspect | Value |
|---|---|
| **Action space** | `Box(3,)` — target focal `(x, y, z)` |
|   bounds | `x, y ∈ [-0.025, 0.025]`, `z ∈ [0.02, 0.08]` |
| **Observation** | `Box(9,)` — `[ball_pos, ball_vel, target_focal]` |
| **Reward** | `−50 · ‖ball − target‖ − 0.001 · ‖v‖² + 0.01` |
| **Episode length** | 500 steps |
| **Termination** | ball leaves safe box, or NaN in `qpos`/`qvel`/`ARF` |
| **Truncation** | `max_episode_steps` reached |
| **Force clip** | `±0.5 N` (addresses QACC NaN warnings) |

### Per-step pipeline
1. Clip action → set `_target_focal`.
2. `phases = generate_twin_trap_phases(focal, trans_pos)`.
3. `_, F = get_physics_outputs(ball_pos[None], trans_pos, trans_zaxis, phases)`.
4. Clip F to ±MAX_FORCE, NaN-guard → 0 with `terminated=True`.
5. Add optional linear drag, write to `data.qfrc_applied[ball_dofadr:ball_dofadr+3]`.
6. `mj_step`; check workspace + finiteness.
7. Sample fresh `target` on `reset()` (or pass via `options["target"]`).

### Transducer geometry extraction (once at `__init__`)
```python
for i in range(79):
    sid = model.site(f"site_sensor_{i+1}").id
    transducer_positions[i] = model.site_pos[sid]
    mujoco.mju_quat2Mat(mat, model.site_quat[sid])
    transducer_zaxis[i] = mat.reshape(3, 3)[:, 2]
```
Same pattern as `scripts/twin_trap_pytorch.py:33-44`.

## AcoustoRL integration

The training driver mirrors `AcoustoRL/examples/train_agent_DDPG_TD3.py`
but points the env at `gym.make("AcousticLevitator-v0")`. Supported
algorithms (imported from `acoustorl`):

- `TD3`, `TD3_per` (default — proven path)
- `DDPG`, `DDPG_per`
- `MADDPG` (also installed but not selected by default)

Replay buffer follows the upstream `_per` suffix convention. Target
folder: `check_points/{env}_{algo}/`.

## How to run

From `acoustic_levitation_project/`:

```bash
# 1. Smoke test (random policy, ~30 s)
python scripts/test_env_rollout.py
#   → logs/smoke_rollout.png shows ball path

# 2. Train (TD3, ~20k steps for a sanity run)
python scripts/train_td3_levitator.py --total_timesteps 20000
#   → check_points/AcousticLevitator-v0_TD3/{actor0.pth,critic0.pth,...}

# 3. Switch algorithms
python scripts/train_td3_levitator.py --algorithm TD3_per --total_timesteps 50000
python scripts/train_td3_levitator.py --algorithm DDPG --total_timesteps 50000
```

## Environment caveats observed

- **Python 3.14** is the only interpreter installed on the box.
  `gymnasium 1.2.3`, `torch 2.12.0`, `mujoco 3.9.0` may not have
  3.14 wheels on PyPI — if `pip install` fails, install Python 3.11
  and re-create the venv.
- **GPU**: code paths `torch.cuda.is_available()` autodetect.
- **MuJoCo warnings**: `MUJOCO_LOG.TXT` records `Nan, Inf or huge
  value in QACC` when forces are unclipped or when the ball escapes
  the workspace. Env clips to ±0.5 N and terminates on workspace exit
  / NaN, which should suppress these.

## Known gotchas

- The twin-trap signature in `generate_twin_trap_phases` uses
  `trans_pos[:, 0] > 0` — not the transducer's outward normal.
  Adequate for the bowl layout; review if you change geometry.
- `MUJOCO_LOG.TXT` is the smoking gun for stability issues; check it
  after a long training run.
- `bowl_base.stl` and `thick_bowl_base.stl` are on disk but not
  referenced by the XML — visuals only, no physics impact.

## Out of scope (for now)

- Multi-particle / multi-agent setups.
- Curriculum learning, domain randomization.
- Wiring the bowl-base mesh into the XML.
- Switching to Isaac Sim (already covered by `scripts/issac_acoustic.py`).
- SAC integration (the rlkit-style `SAC.py` in `AcoustoRL` is not
  wired into `algorithm_instantiation`; wire separately if needed).
