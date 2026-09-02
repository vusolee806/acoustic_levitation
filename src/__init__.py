"""Acoustic levitation project package.

Registers the `AcousticLevitator-v0` Gymnasium environment on import so
`gym.make("AcousticLevitator-v0")` works as soon as `src` is on
`sys.path`.
"""

from gymnasium.envs.registration import register

from .levitator_env import AcousticLevitatorEnv

register(
    id="AcousticLevitator-v0",
    entry_point="src.levitator_env:AcousticLevitatorEnv",
    max_episode_steps=500,
)

__all__ = ["AcousticLevitatorEnv"]