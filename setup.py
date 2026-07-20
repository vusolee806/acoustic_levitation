from setuptools import setup, find_packages

setup(
    name="my_mujoco_project",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mujoco",
        "gymnasium[mujoco]",
        "numpy",
        "matplotlib",
        "pyyaml"
    ],
)
