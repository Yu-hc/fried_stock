# Cell 1: Import modules and load/process data
import pandas as pd
import sys
import os

# Get the absolute path to the directory containing the current notebook
# and then move up one level to the project root
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
       sys.path.append(module_path)

# Now you can import from the modules folder
from modules.data_processor import FeatureFactory
from modules.environment import StockTradingEnv
from modules.model_agent import TradingAgent
from modules.rewards import RewardFunction
# Assuming evaluator.py exists for Module E
from modules.evaluator import Evaluator  # If not present, skip evaluation

# Load raw data
raw_df = pd.read_csv('../data/2330_2015-2025.csv')

# Process data with Module A
factory = FeatureFactory(strategy='donchian')
train_df, test_df = factory.process_and_split(raw_df)

# Create environment with Module B for training
train_env = StockTradingEnv(df=train_df, initial_balance=10000.0, window_size=30)
# Create a separate environment for testing
test_env = StockTradingEnv(df=test_df, initial_balance=10000.0, window_size=30)


# Initialize agent with Module D
agent = TradingAgent(env=train_env, model_path="../models/")

# Train the agent (adjust timesteps as needed)
agent.train(total_timesteps=1000000)

# Basic evaluation: Run a test episode and log results
obs, _ = test_env.reset()
done = False
trade_log = []
while not done:
       action = agent.predict_action(obs)
       obs, reward, done, truncated, info = test_env.step(action)
       trade_log.append(info)
# Mock data for demonstration if you haven't run the model yet:
# trade_log = [{'step': 0, 'balance': 10000}, {'step': 50, 'balance': 11500}]
# price_history = processed_df['Close'].iloc[test_indices]
# # Initialize and Run Evaluation
evaluator = Evaluator(initial_balance=10000)
report = evaluator.evaluate(trade_log, test_df['raw_close'])

print("--- Performance Report ---")
for key, value in report.items():
       print(f"{key}: {value}")