# Acoustic Levitation Project — Context

> Living reference for the `acoustic_levitation_project` workspace
> (parent dir: `D:/WORKSPACE/ALL_MY_PRJ/ACOUSTIC LEVIATION/acoustic_leviation/`).
> Update this when files, conventions, or the user's role change.

## What this project is

A 79-transducer bowl-shaped acoustic levitator simulated in MuJoCo. The
project demonstrates acoustic radiation force computation and particle
levitation using a single EPS-foam ball (`test_ball`, density 40 kg/m³, 
radius 2 mm) that hovers and can be steered toward target focal points.

Three physics backends exist:

- **NumPy / finite differences** — `src/acoustic_physics.py`.
- **PyTorch + autograd** — `src/acoustic_physics_pytorch.py`
  (gradient is exact, used for custom physics).
- **Levitate library** — `scripts/twintrap_gemini_levitate.py`
  (current preferred method; uses external `levitate` package for
  radiation force computation with twin-trap and vortex signatures).

A separate Isaac Sim script (`scripts/issac_acoustic.py`) mirrors the
MuJoCo setup for higher-fidelity visualization.

## Repository layout

```
acoustic_levitation_project/
├── README.md                 # bare-bones; lists the expected tree
├── requirements.txt          # torch 2.12.0, mujoco 3.9.0, levitate, ...
├── setup.py                  # installs `src` as a package
├── MUJOCO_LOG.TXT            # historical warnings: Nan/Inf in QACC
├── assets/
│   ├── mujoco_levitator.xml  # MAIN xml — bowl + 79 sites + test_ball freejoint
│   ├── mujoco_array.xml      # older xml (not used)
│   ├── bowl_base.stl         # mesh present but NOT referenced by the XML
│   └── thick_bowl_base.stl   # mesh present but NOT referenced by the XML
├── configs/default_config.yaml
├── scripts/
│   ├── twin_trap.py          # NumPy reference control loop
│   ├── twin_trap_pytorch.py  # PyTorch reference control loop
│   ├── twintrap_gemini_levitate.py  # CURRENT — levitate library twin-trap demo
│   └── issac_acoustic.py     # Isaac Sim variant
├── src/
│   ├── __init__.py           # package initialization
│   ├── utils.py              # `get_asset_path(filename)` — resolves XML paths
│   ├── acoustic_physics.py             # NumPy pressure / Gor'kov / ARF
│   └── acoustic_physics_pytorch.py     # PyTorch version + `get_physics_outputs`
├── examples/
│   ├── heatmap.py
│   ├── test_physics_pytorch.py
│   └── test_sound_wave.py
└── logs/                     # plots, simulation outputs
```

## MuJoCo model summary (`assets/mujoco_levitator.xml`)

- `timestep=0.0001`, density=1.225, viscosity=1.85e-5 (air-like).
- 79 transducer bodies `sensor_1..sensor_79`, each with `site_sensor_N`
  positioned at the array surface and a cylinder geom. Sites carry
  per-transducer orientation via `site_quat`. Arranged in a bowl shape.
- Bowl geometry: a curved reflective surface surrounds the transducer array
  to enhance acoustic field confinement.
- `test_ball` body: freejoint (6 DOF), sphere size=0.002 m, density=40,
  initial pos `(0, 0, 0.05)`. Material: EPS foam (sound speed 1200 m/s).
- 79 rangefinder sensors along each transducer's z-axis.
- No mesh references in the XML — the STL files on disk are unused.

## Key constants (shared across backends)

| Symbol | Value | Meaning |
|---|---|---|
| `C_0` | 343.0 m/s | Speed of sound in air |
| `RHO_0` | 1.225 kg/m³ | Air density |
| `C_1` | 1200.0 m/s | Speed of sound in EPS |
| `RHO_1` | 40.0 kg/m³ | EPS density |
| `PARTICLE_RADIUS` | 0.002 m | Ball radius |
| `FREQ` | 40 kHz | Ultrasonic carrier (λ ≈ 8.575 mm) |
| `TRANSDUCER_RADIUS` | 0.01 m | Piston radius (legacy) |
| `TRANSDUCER_SIZE` | 0.02 m | Transducer diameter (levitate) |

## Levitate Library Integration

The current preferred method uses the external `levitate` library
(`scripts/twintrap_gemini_levitate.py`). Key workflow:

### 1. Setup array and materials
```python
import levitate

levitate.frequency = 40000.0  # 40 kHz

# EPS foam particle material
ball_material = levitate.materials.Material(rho=40.0, c=1200.0)

# Extract transducer geometry from MuJoCo
mujoco.mj_kinematics(model, data)
for i in range(79):
    site_id = model.site(f"site_sensor_{i+1}").id
    transducer_positions[i] = data.site_xpos[site_id]
    mat = data.site_xmat[site_id].reshape(3, 3)
    transducer_zaxis[i] = mat[:, 2]  # Z-axis vector

# Build levitate array
array = levitate.arrays.NormalTransducerArray(
    positions=transducer_positions.T,  # (3, 79)
    normals=transducer_zaxis.T,
    transducer_size=0.02  # 20mm diameter
)

force_evaluator = levitate.fields.RadiationForce(
    array, radius=0.002, material=ball_material
)
```

### 2. Generate twin-trap signature
```python
target_focal_point = np.array([0.0, 0.0, 0.04])

# Focus phases bring field to a point
focus_phases = array.focus_phases(target_focal_point)

# Twin signature splits the trap (stype='twin' or 'vortex')
twin_signature = array.signature(position=target_focal_point, stype='vortex')

# Combine and convert to complex weights
array.phases = focus_phases + twin_signature
complex_weights = 0.3 * np.exp(1j * array.phases)
```

### 3. Visualization (optional)
```python
viz = array.visualize
viz.append(levitate.visualizers.PressureSlice(array))
fig = viz(complex_weights)
fig.show()  # Interactive Plotly 3D pressure field
```

### 4. Runtime force application
```python
while viewer.is_running():
    ball_pos = data.xpos[ball_id]
    forces_array = force_evaluator(complex_weights, ball_pos)
    arf_force = np.clip(forces_array, -1.0, 1.0)
    
    dof_start = model.body_dofadr[ball_id]
    data.qfrc_applied[dof_start : dof_start + 3] = arf_force
    mujoco.mj_step(model, data)
```

The script logs force components (Fx, Fy, Fz) and ball z-position over
time, then plots them with matplotlib after the viewer closes.

## Legacy physics backends

### PyTorch backend (`src/acoustic_physics_pytorch.py`)

Differentiable pressure + Gor'kov potential calculation. The function
`get_physics_outputs()` takes:
- 79×3 tensor of sensor positions
- 79 complex-valued transducer amplitudes  
- ball_pos (3,)
- optional radius / material properties

Returns a dict:
```python
{
  "pressure": <complex tensor>,
  "arf_force": <3-D force>,
  "gorkov_potential": scalar,
  ...
}
```

Previously used for RL training. Still present for custom physics
experiments but superseded by levitate library for primary simulations.

The twin-trap phase generator (`generate_twin_trap_phases`) in this
backend computes focusing phases from distance-to-focal ratio modulo 1,
then adds π to every transducer whose `x > 0` (the twin-trap signature).

### NumPy backend (`src/acoustic_physics.py`)

Finite-difference pressure computation, then Gor'kov potential + gradient.
Reference implementation, rarely invoked now.

## How to run

From `acoustic_levitation_project/`:

```bash
# Primary: Twin-trap simulation with levitate library
python scripts/twintrap_gemini_levitate.py
#   → Interactive Plotly 3D visualization of pressure field
#   → Matplotlib plots: force components and ball trajectory over time

# Legacy: PyTorch backend with twin-trap control
python scripts/twin_trap_pytorch.py

# Legacy: NumPy backend with twin-trap control
python scripts/twin_trap.py

# Isaac Sim variant
python scripts/issac_acoustic.py
```

## Dependencies

Core packages:
- `mujoco==3.9.0` — Physics simulation
- `levitate` — Acoustic field computation and visualization
- `torch==2.12.0` — Legacy PyTorch backend
- `numpy` — Numerical operations
- `matplotlib` — Plotting
- `plotly` — Interactive 3D visualization
- `gymnasium==1.2.3` — No longer used for training; imports may remain

## Environment caveats observed

- **Python 3.14** is the only interpreter installed on the box.
  `torch 2.12.0`, `mujoco 3.9.0` may not have 3.14 wheels on
  PyPI — if `pip install` fails, install Python 3.11 and re-create
  the venv.
- **GPU**: code paths `torch.cuda.is_available()` autodetect (legacy
  PyTorch backend only).
- **MuJoCo warnings**: `MUJOCO_LOG.TXT` records `Nan, Inf or huge
  value in QACC` when forces are unclipped or when the ball escapes
  the workspace. Forces are clipped to ±1 N in the runtime loop, which
  suppresses these warnings.

## Known gotchas

- The levitate-based twin-trap signature (`stype='vortex'`) is
  geometry-aware; the legacy PyTorch generator uses
  `trans_pos[:, 0] > 0` instead of outward normals. Adequate for the
  current bowl layout; review if you change geometry.
- `MUJOCO_LOG.TXT` is the smoking gun for stability issues; check it
  after a long simulation run.
- `bowl_base.stl` and `thick_bowl_base.stl` are on disk but not
  referenced by the XML — visuals only, no physics impact.
- Force clipping at ±1 N caps the maximum lift, but keeps MuJoCo
  stable. Increase only if you can guarantee the ball never escapes
  the workspace.

## Out of scope (for now)

- Multi-particle / multi-agent setups.
- Wiring the bowl-base mesh into the XML.
- Switching to Isaac Sim (already covered by `scripts/issac_acoustic.py`).
- RL training infrastructure (removed from project; legacy
  experiments left in `examples/` for reference).
