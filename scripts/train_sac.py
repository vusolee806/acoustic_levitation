# scripts/train_sac.py

import os
import yaml
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from HemisAcoustic.envs.train_env_single import TrainEnvSingle
from HemisAcoustic.envs.sac_wrapper import SACAcousticWrapper


class SaveBestSACCallback(BaseCallback):
    """Evaluates SAC periodically and saves the best model checkpoint."""
    def __init__(self, eval_env, save_prefix, eval_freq=10_000, n_eval_episodes=30, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_prefix = save_prefix
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_success_rate = -1.0
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        successes = 0
        episode_rewards = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            ep_rew = 0.0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                ep_rew += float(reward[0])
                if done[0]:
                    if info[0].get("is_success", False):
                        successes += 1
                    break
            episode_rewards.append(ep_rew)

        success_rate = successes / self.n_eval_episodes
        mean_reward = float(np.mean(episode_rewards))

        if self.verbose:
            print(
                f"[Eval @ {self.num_timesteps:>7d} steps] "
                f"Mean Reward: {mean_reward:7.2f} | "
                f"Success Rate: {success_rate * 100:5.1f}%"
            )

        if (success_rate > self.best_success_rate) or (
            success_rate == self.best_success_rate and mean_reward > self.best_mean_reward
        ):
            self.best_success_rate = success_rate
            self.best_mean_reward = mean_reward
            self.model.save(self.save_prefix + ".zip")
            if self.verbose:
                print(f"  -> Saved new best SAC model to {self.save_prefix}.zip")

        return True


def make_env(config, shaping=True):
    def _init():
        return Monitor(SACAcousticWrapper(TrainEnvSingle(config), train_reward_shaping=shaping))
    return _init


def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/train_sac_single.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config["train"]
    model_version = train_cfg.get("model_version", "_v2")
    save_prefix = train_cfg["save_path"] + model_version
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    # Plain DummyVecEnv with static SACAcousticWrapper (no VecNormalize)
    train_env = DummyVecEnv([make_env(config, shaping=True)])
    eval_env = DummyVecEnv([make_env(config, shaping=False)])

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=train_cfg["learning_rate"],
        buffer_size=train_cfg["buffer_size"],
        learning_starts=train_cfg["learning_starts"],
        batch_size=train_cfg["batch_size"],
        tau=train_cfg["tau"],
        gamma=train_cfg["gamma"],
        train_freq=train_cfg["train_freq"],
        gradient_steps=train_cfg["gradient_steps"],
        ent_coef=train_cfg["ent_coef"],
        policy_kwargs=dict(net_arch=train_cfg["net_arch"]),
        verbose=1,
        device="auto",
    )

    callback = SaveBestSACCallback(
        eval_env=eval_env,
        save_prefix=save_prefix,
        eval_freq=train_cfg["eval_freq"],
        n_eval_episodes=train_cfg["n_eval_episodes"],
    )

    print(f"=== TRAINING SB3 SAC ({model_version}) FOR {train_cfg['total_timesteps']} TIMESTEPS ===")
    model.learn(total_timesteps=train_cfg["total_timesteps"], callback=callback)

    final_prefix = save_prefix + "_final"
    model.save(final_prefix + ".zip")
    print(f"=== DONE! Best model: {save_prefix}.zip | Final model: {final_prefix}.zip ===")


if __name__ == "__main__":
    main()
