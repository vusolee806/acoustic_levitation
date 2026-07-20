import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import your custom environment from the src folder
from src.envs import CustomCartPoleEnv
from gymnasium.wrappers import TimeLimit
def main():
    # 1. Path to your trained model
    # We try to load the best model saved by the callback; fall back to final if missing
    model_path = "check_points/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "check_points/ppo_cartpole_final.zip"
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found in check_points/. Please run training first.")

    print(f"Loading model from: {model_path}")

    # 2. Initialize environment for quantitative evaluation (no rendering)
    xml_path = "assets/cartpole.xml"
    eval_env = CustomCartPoleEnv(xml_path=xml_path)
    eval_env = TimeLimit(eval_env, max_episode_steps=1000)
    # 3. Quantitative Evaluation
    print("Evaluating agent performance over 10 episodes...")
    mean_reward, std_reward = evaluate_policy(
        PPO.load(model_path), 
        eval_env, 
        n_eval_episodes=10, 
        deterministic=True
    )
    print(f"--> Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    eval_env.close()

    # 4. Qualitative Evaluation (Visualizing the agent)
    print("\nStarting visual playback... Press Ctrl+C in the terminal to stop.")
    
    # Note: For visual rendering in Gymnasium/MuJoCo, we often rely on standard loops.
    # We recreate the environment specifically for a visual test run.
    vis_env = CustomCartPoleEnv(xml_path=xml_path)
    model = PPO.load(model_path, env=vis_env)

    # If you want to use MuJoCo's passive viewer for high-framerate 3D graphics,
    # we can import mujoco.viewer.
    try:
        import mujoco.viewer
        # Use MuJoCo's interactive viewer directly on the underlying model
        with mujoco.viewer.launch_passive(vis_env.model, vis_env.data) as viewer:
            obs, _ = vis_env.reset()
            while viewer.is_running():
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = vis_env.step(action)
                
                # Sync the viewer with the underlying physics data
                viewer.sync()
                
                if terminated or truncated:
                    obs, _ = vis_env.reset()
    except ImportError:
        print("mujoco.viewer package not found. Running headless step-through instead.")
        obs, _ = vis_env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = vis_env.step(action)
            if terminated or truncated:
                obs, _ = vis_env.reset()

    vis_env.close()

if __name__ == "__main__":
    main()
