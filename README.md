# HemisAcoustic

Reinforcement-learning path planning for a particle in a hemispherical acoustic
levitator (79 ultrasonic transducers, 40 kHz), with a MuJoCo replay that moves an
acoustic trap along the planned path.

## How it works

1. An RL agent (SAC, DDPG or PPO) controls one particle in a kinematic environment
   (`TrainEnvSingle`). The action is a 3D velocity. The particle starts at the
   workspace center and must reach a random target inside an ellipsoid
   (30 mm radius in x and y, 15 mm in z).
2. The trained policy produces a path of waypoints.
3. The `simulate_mujoco_*` scripts interpolate the path and move a Gor'kov-potential
   trap along it in MuJoCo. A foam ball is pulled by the radiation force, and the
   script plots how closely the ball follows the plan.

The RL environment contains no acoustic forces. Only the MuJoCo replay checks
whether the ball can follow the path.

## Setup

```powershell
pip install -e .
```

There is no `requirements.txt`. The code imports `mujoco`, `torch`,
`stable-baselines3`, `gymnasium`, `numpy`, `scipy`, `pyyaml` and `matplotlib`.
Install these yourself.

## Usage

Run from the project root. Each algorithm has a train, evaluate and simulate script.

```powershell
# SAC (main)
python scripts/train_sac.py
python scripts/evaluate_sac.py
python scripts/simulate_mujoco_sac.py

# DDPG
python scripts/train_ddpg.py
python scripts/evaluate_ddpg.py
python scripts/simulate_mujoco_ddpg.py

# PPO (baseline)
python scripts/train_single.py
python scripts/evaluate_single.py
python scripts/simulate_mujoco_ppo.py
```

Hyperparameters and workspace size are set in `configs/train_sac_single.yaml`,
`configs/train_ddpg_single.yaml` and `configs/train_single.yaml`. Models are saved
to `logs/checkpoints/`, which is git-ignored.

`simulate_mujoco_*.py` opens a MuJoCo viewer and a matplotlib window. To pick the
target, edit `CUSTOM_TARGET` (meters) inside `main()`, or set it to `None` for a
random one.

## Layout

```
HemisAcoustic/
  assets/      MuJoCo XML, bowl meshes, physical constants
  envs/        TrainEnvSingle, SACAcousticWrapper
  particles/   Particle and Target state classes
  physics/     HemisphereAcousticSolver (phases, Gor'kov potential, force)
  utils/       point sampling, noise models (not yet used by any script)
configs/       one YAML per algorithm
scripts/       train / evaluate / simulate scripts; OldCode/ is legacy
logs/          checkpoints, tensorboard, plots (git-ignored)
```

## More detail

`PROJECT_CONTEXT.md` has the full description: observation and reward definitions,
config comparison, conventions for editing, and known issues.
