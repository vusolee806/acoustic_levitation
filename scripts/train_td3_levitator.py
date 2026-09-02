"""Train a TD3 (or TD3_per / DDPG / DDPG_per / MADDPG) agent on
`AcousticLevitator-v0` using the existing AcoustoRL training utilities.

This is a thin wrapper around `train_off_policy_agent` that wires the env
through `gym.make("AcousticLevitator-v0")`, builds the requested agent
class, and points the replay buffer at it. Reuses 100% of AcoustoRL's
algorithm code.

Usage (from project root):
    python scripts/train_td3_levitator.py --total_timesteps 20000
"""

import os
import sys
import random

import numpy as np
import torch
import gymnasium as gym
from argparse import ArgumentParser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACOUSTORL_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "AcoustoRL"))
sys.path.append(PROJECT_ROOT)
sys.path.append(ACOUSTORL_ROOT)

import src  # registers AcousticLevitator-v0
from acoustorl import DDPG, DDPG_per, TD3, TD3_per, MADDPG
from acoustorl.common import per
from acoustorl.common.general_utils import (
    ReplayBuffer,
    train_off_policy_agent,
    eval_policy,
)


AGENT_MAP = {
    "TD3": TD3,
    "TD3_per": TD3_per,
    "DDPG": DDPG,
    "DDPG_per": DDPG_per,
    "MADDPG": MADDPG,
}


def build_agent(algo: str, kwargs: dict):
    cls = AGENT_MAP[algo]
    if algo == "TD3_per" or algo == "DDPG_per":
        kwargs.setdefault("if_use_huber_loss", False)
    return cls(**kwargs)


def build_replay_buffer(algo: str, state_dim: int, action_dim: int,
                       max_size: int, device: torch.device):
    if algo.endswith("_per"):
        return per.ReplayBuffer(
            state_dim=state_dim, action_dim=action_dim,
            max_size=max_size, device=device,
        )
    return ReplayBuffer(
        state_dim=state_dim, action_dim=action_dim,
        max_size=max_size, device=device,
    )


def main() -> None:
    parser = ArgumentParser(description="Train an RL agent on AcousticLevitator-v0")
    parser.add_argument("--algorithm", default="TD3", choices=list(AGENT_MAP.keys()))
    parser.add_argument("--env", default="AcousticLevitator-v0")
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--exploration_noise", default=0.1, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--update_times", default=1, type=int)
    parser.add_argument("--total_timesteps", default=20000, type=int)
    parser.add_argument("--buffer_size", default=int(1e5), type=int)
    parser.add_argument("--minimal_size", default=int(2e3), type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seeds", default=1, type=int,
                        help="Number of random seeds to run sequentially.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("---------------------------------------")
    print(f"Algorithm: {args.algorithm}, Env: {args.env}, Device: {device}")
    print("---------------------------------------")

    target_folder = os.path.join(PROJECT_ROOT, "check_points", f"{args.env}_{args.algorithm}")
    os.makedirs(target_folder, exist_ok=True)

    for seed in range(args.seeds):
        env = gym.make(args.env)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        min_action = float(env.action_space.low[0])
        max_action = float(env.action_space.high[0])

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "min_action": min_action,
            "max_action": max_action,
            "hidden_dim": args.hidden_dim,
            "exploration_noise": args.exploration_noise,
            "discount": args.discount,
            "tau": args.tau,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "device": device,
        }
        # TD3-family kwargs (no-op for DDPG)
        if args.algorithm.startswith("TD3"):
            kwargs["policy_noise"] = args.policy_noise
            kwargs["noise_clip"] = args.noise_clip
            kwargs["policy_freq"] = args.policy_freq

        agent = build_agent(args.algorithm, kwargs)
        replay_buffer = build_replay_buffer(
            args.algorithm, state_dim, action_dim,
            max_size=args.buffer_size, device=device,
        )

        return_list, std_list = train_off_policy_agent(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            batch_size=args.batch_size,
            minimal_size=args.minimal_size,
            total_timesteps=args.total_timesteps,
            update_times=args.update_times,
            env_name=args.env,
            target_folder=target_folder,
            seed=seed,
        )

        np.save(os.path.join(target_folder, f"return_list{seed}.npy"),
                np.asarray(return_list))
        np.save(os.path.join(target_folder, f"std_list{seed}.npy"),
                np.asarray(std_list))

        env.close()


if __name__ == "__main__":
    main()