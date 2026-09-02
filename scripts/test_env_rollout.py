"""Smoke test for AcousticLevitator-v0.

Runs 5 episodes of the random-policy Gymnasium env, prints reward / ARF
stats per episode, and plots ball trajectory vs. target focal point.

Usage (from project root):
    python scripts/test_env_rollout.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import src  # registers AcousticLevitator-v0
from src.levitator_env import AcousticLevitatorEnv


def physics_sanity_checks(env: AcousticLevitatorEnv) -> None:
    """Confirm ARF behaves the way the reference scripts show:
    near-zero force when target == ball, attractive force toward an
    offset target.
    """
    env.reset(seed=0)
    env._target_focal = np.array([0.0, 0.0, 0.05], dtype=np.float32)
    f0 = env._compute_arf(np.array([0.0, 0.0, 0.05], dtype=np.float32))
    print(f"[sanity] ARF @ target=(0,0,0.05), ball=(0,0,0.05): {f0}")

    # Offset the ball 5mm along +x; ARF should pull it back toward the trap.
    f1 = env._compute_arf(np.array([0.005, 0.0, 0.05], dtype=np.float32))
    print(f"[sanity] ARF @ target=(0,0,0.05), ball=(0.005,0,0.05): {f1}")
    assert f1[0] < 0, f"Expected restoring force along -x, got {f1}"


def random_episode(env: AcousticLevitatorEnv, seed: int) -> dict:
    core = env.unwrapped  # env is TimeLimit-wrapped; .data/.ball_id/._compute_arf live on the base env
    obs, info = env.reset(seed=seed)
    target = info["target_focal"]
    positions = [core.data.xpos[core.ball_id].copy()]
    rewards = []
    forces = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        positions.append(core.data.xpos[core.ball_id].copy())
        forces.append(core._compute_arf(core.data.xpos[core.ball_id]))
    return {
        "seed": seed,
        "target": target,
        "positions": np.array(positions),
        "rewards": np.array(rewards),
        "forces": np.array(forces),
        "return": float(np.sum(rewards)),
        "length": int(len(rewards)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main() -> None:
    env = gym.make("AcousticLevitator-v0")

    # Verify registered spaces
    assert env.observation_space.shape == (9,), env.observation_space.shape
    assert env.action_space.shape == (3,), env.action_space.shape
    print(f"[space] obs={env.observation_space}, action={env.action_space}")

    physics_sanity_checks(env.unwrapped)

    episodes = [random_episode(env, seed=s) for s in range(5)]
    print("\n=== Random-policy rollout (5 episodes) ===")
    for ep in episodes:
        last_pos = ep["positions"][-1]
        dist = float(np.linalg.norm(last_pos - ep["target"]))
        fmean = float(np.mean(np.linalg.norm(ep["forces"], axis=1))) if len(ep["forces"]) else 0.0
        print(
            f"seed={ep['seed']:>2} return={ep['return']:+.3f} "
            f"steps={ep['length']:>3} last_dist={dist*1000:6.1f}mm "
            f"mean_|F|={fmean:.3f}N "
            f"terminated={ep['terminated']} truncated={ep['truncated']}"
        )

    # Trajectory plot for the last episode
    ep = episodes[-1]
    pos = ep["positions"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(pos[:, 0] * 1000, pos[:, 2] * 1000, label="ball path", linewidth=2)
    ax.scatter([ep["target"][0] * 1000], [ep["target"][2] * 1000],
               marker="*", s=200, label="target focal", color="red")
    ax.scatter([0], [50], marker="o", s=80, label="initial ball", color="green")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(f"Random-policy rollout — seed={ep['seed']}, return={ep['return']:+.2f}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PROJECT_ROOT, "logs", "smoke_rollout.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")

    env.close()


if __name__ == "__main__":
    main()
