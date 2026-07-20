import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit # 1. Import the wrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import your custom environment from the src folder
from src.envs import CustomCartPoleEnv

def main():
    # 1. Define directories to match your project structure
    models_dir = "check_points"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Initialize the training and evaluation environments
    # Assuming you run this script from the root 'mujoco_project' folder
    xml_path = "assets/cartpole.xml"
    env = CustomCartPoleEnv(xml_path=xml_path)
    env = TimeLimit(env, max_episode_steps=1000) # Wrap training env, forces the environment to reset after 1000 steps,
    
    eval_env = CustomCartPoleEnv(xml_path=xml_path)
    eval_env = TimeLimit(eval_env, max_episode_steps=1000) # Wrap eval env
    # 3. Setup an Evaluation Callback
    # Tests the agent every 5,000 steps and saves the best model to check_points/best_model.zip
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # 4. Initialize the PPO Agent
    print("Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy",          # Uses a standard Multi-Layer Perceptron neural network
        env,
        verbose=1,            # Print training metrics to the console
        tensorboard_log=log_dir,
        learning_rate=0.0003, # Standard stable learning rate for PPO
        n_steps=2048,         # Number of steps to collect before updating the network
        batch_size=64
    )

    # 5. Train the Agent
    # 300,000 timesteps is usually enough to master a continuous CartPole
    timesteps = 250_000 
    print(f"Starting training for {timesteps} timesteps...")
    
    # tb_log_name organizes your TensorBoard logs neatly
    model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name="PPO_CartPole")

    # 6. Save the final model (just in case the last epoch is the best one)
    final_path = os.path.join(models_dir, "ppo_cartpole_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")

    # Clean up memory
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
