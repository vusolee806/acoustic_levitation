# Acoustic Levitation Project

MuJoCo simulation of a 79-transducer bowl-shaped acoustic levitator with
direct physics integration via the external `levitate` library.

## Overview

A single EPS-foam ball (density 40 kg/m³, radius 2 mm) is levitated and
steered using a 40 kHz ultrasonic transducer array arranged in a bowl
geometry. Acoustic radiation force (ARF) is computed using the
`levitate` library and applied to the ball in MuJoCo via
`qfrc_applied`.

## Repository layout

```
acoustic_levitation_project/
├── README.md
├── PROJECT_CONTEXT.md     # Living reference for architecture & decisions
├── requirements.txt
├── setup.py
├── assets/
│   ├── mujoco_levitator.xml    # MAIN xml — bowl + 79 sites + test_ball
│   └── mujoco_array.xml        # older xml (not used)
├── configs/default_config.yaml
├── scripts/
│   ├── twintrap_gemini_levitate.py  # PRIMARY — twin-trap with levitate library
│   ├── twin_trap_pytorch.py         # Legacy PyTorch backend control loop
│   ├── twin_trap.py                 # Legacy NumPy backend control loop
│   └── issac_acoustic.py            # Isaac Sim variant
├── src/
│   ├── acoustic_physics.py             # NumPy pressure / Gor'kov / ARF
│   ├── acoustic_physics_pytorch.py     # PyTorch version + get_physics_outputs
│   └── utils.py                        # get_asset_path()
├── examples/
│   ├── heatmap.py
│   ├── test_physics_pytorch.py
│   └── test_sound_wave.py
└── logs/                                # plots, simulation outputs
```

## Quick start

```bash
# Primary simulation: twin-trap with levitate library
python scripts/twintrap_gemini_levitate.py
# → Interactive Plotly 3D pressure field visualization
# → Matplotlib plots of force components and ball trajectory

# Legacy backends
python scripts/twin_trap_pytorch.py
python scripts/twin_trap.py

# Isaac Sim variant
python scripts/issac_acoustic.py
```

## Physics summary

- **Transducer array**: 79 pistons at 40 kHz, 20 mm diameter
- **Particle material**: EPS foam (ρ = 40 kg/m³, c = 1200 m/s)
- **Twin-trap**: focus phases + vortex signature from levitate library
- **Force application**: ARF clipped to ±1 N, written to `qfrc_applied`

## Dependencies

- `mujoco==3.9.0` — Physics simulation
- `levitate` — Acoustic field computation and visualization
- `torch==2.12.0` — Legacy PyTorch backend
- `numpy`, `matplotlib`, `plotly`

See `PROJECT_CONTEXT.md` for full architecture documentation,
gotchas, and out-of-scope items.