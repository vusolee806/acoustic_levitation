acoustic_levitation_project/
├── .gitignore
├── README.md
├── requirements.txt      # Or environment.yml for Conda
├── setup.py              # Makes your local scripts importable anywhere
├── assets/               # MuJoCo XML, MJCF, STL mesh files
│   ├── mujoco_array.xml        #old xml file
│   └── mujoco_levitator.xml    #main xml file for transducer position and test_ball position
├── configs/              # Hyperparameters (YAML or JSON)
│   └── default_config.yaml
├── logs/                 # Saved tensorboard runs, metrics, and plots
├── check_points/         # Saved model weights (if doing RL)
├── scripts/              # High-level executable scripts
│   ├── twin_trap_pytorch.py    
│   ├── issac_acoustic.py.py
│   └── twin_trap.py
└── src/                  # Core library code (Your custom logic)
    ├── __init__.py
    ├── acoustic_physics_pytorch.py     # Physic function calculated by pytorch
    ├── acoustic_physics.py             # Physic function calculated by numpy
    └── utils.py                        # Math utilities, coordinate transforms
