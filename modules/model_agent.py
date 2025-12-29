import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

class TradingAgent:
       def __init__(self, env, model_path="./models/"):
              """
              Args:
              env (gym.Env): The custom StockTradingEnv from Module B.
              model_path (str): Directory to save trained models.
              """
              self.env = env
              self.model_path = model_path
              os.makedirs(self.model_path, exist_ok=True)

              # Initialize PPO Model
              # MlpPolicy is used here, but for (window_size, features), 
              # SB3 handles the flattening automatically. 
              self.model = PPO(
                     policy="MlpPolicy", 
                     env=self.env,
                     learning_rate=3e-4,
                     n_steps=2048,
                     batch_size=64,
                     n_epochs=10,
                     gamma=0.99,         # Discount factor for future rewards
                     gae_lambda=0.95,
                     clip_range=0.2,     # PPO clipping for stability
                     verbose=1,
                     tensorboard_log="./logs/ppo_trading_tensorboard/"
              )

       def train(self, total_timesteps=100000):
              """Trains the agent and saves the best model."""
              print(f"Starting training for {total_timesteps} steps...")
              
              # Save a checkpoint every 10,000 steps
              checkpoint_callback = CheckpointCallback(
                     save_freq=10000, 
                     save_path=self.model_path,
                     name_prefix="ppo_trade_model"
              )

              self.model.learn(
                     total_timesteps=total_timesteps,
                     callback=checkpoint_callback,
                     progress_bar=True
              )
              
              self.model.save(os.path.join(self.model_path, "final_trading_model"))
              print("Training complete. Model saved.")

       def predict_action(self, observation):
              """
              Takes an observation and returns the discrete action.
              0: Hold, 1: Buy, 2: Sell
              """
              action, _states = self.model.predict(observation, deterministic=True)
              
              # Optional: Get action probabilities for analysis
              # (Useful for understanding agent confidence)
              return int(action)

       def load_model(self, filename):
              """Loads a pre-trained model."""
              full_path = os.path.join(self.model_path, filename)
              self.model = PPO.load(full_path, env=self.env)
              print(f"Model loaded from {full_path}")