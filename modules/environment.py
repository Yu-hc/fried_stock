import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from rewards import RewardFunction

class StockTradingEnv(gym.Env):
       """
       A custom stock trading environment for Gymnasium.
       Action Space: 0 = Hold, 1 = Buy, 2 = Sell
       """
       metadata = {'render_modes': ['human']}

       def __init__(self, df, initial_balance=10000.0, window_size=30):
              super(StockTradingEnv, self).__init__()

              self.df = df.reset_index(drop=True)
              self.initial_balance = initial_balance
              self.window_size = window_size
              
              # Action Space: [Hold, Buy, Sell]
              self.action_space = spaces.Discrete(3)

              # Observation Space: (window_size, num_features)
              # We use float32 for compatibility with most RL libraries (Stable Baselines3, etc.)
              self.num_features = len(df.columns)
              self.observation_space = spaces.Box(
                     low=-np.inf, 
                     high=np.inf, 
                     shape=(self.window_size, self.num_features), 
                     dtype=np.float32
              )

              # State Variables
              self.current_step = 0
              self.balance = self.initial_balance
              self.shares_held = 0
              self.net_worth = self.initial_balance
              self.max_net_worth = self.initial_balance
              self.cost_basis = 0
              self.reward_handler = RewardFunction(penalty_weight=2.5)

       def reset(self, seed=None, options=None):
              super().reset(seed=seed)
              
              self.current_step = self.window_size
              self.balance = self.initial_balance
              self.shares_held = 0
              self.net_worth = self.initial_balance
              self.max_net_worth = self.initial_balance
              self.cost_basis = 0
              
              return self._get_observation(), {}

       def _get_observation(self):
              # Returns the slice of data for the current window
              obs = self.df.iloc[self.current_step - self.window_size: self.current_step].values
              return obs.astype(np.float32)

       def step(self, action):
              prev_nav = self.net_worth
              
              # 1. Perform Trade (returns number of shares moved for cost calculation)
              shares_traded, price = self._take_action(action)
              
              # 2. Update State
              self.current_step += 1
              self.net_worth = self.balance + (self.shares_held * price)
              self.max_net_worth = max(self.max_net_worth, self.net_worth)

              # 3. Call Module C
              reward = self.reward_handler.calculate(
                     current_nav=self.net_worth,
                     previous_nav=prev_nav,
                     max_nav=self.max_net_worth,
                     action_type=action,
                     current_price=price,
                     shares_traded=shares_traded
              )
              # 5. Check if Done
              done = self.current_step >= len(self.df) - 1
              truncated = self.net_worth <= self.initial_balance * 0.1  # Bankrupt if lost 90%

              # 6. Info Dictionary
              info = {
                     'net_worth': self.net_worth,
                     'shares_held': self.shares_held,
                     'step': self.current_step
              }

              return self._get_observation(), reward, done, truncated, info

       def _take_action(self, action, current_price):
              if action == 1: # Buy
                     # Simplified: Buy as many shares as possible with current balance
                     total_possible = self.balance // current_price
                     if total_possible > 0:
                            self.shares_held += total_possible
                            self.balance -= total_possible * current_price
                            self.cost_basis = current_price

              elif action == 2: # Sell
                     # Simplified: Sell all shares held
                     if self.shares_held > 0:
                            self.balance += self.shares_held * current_price
                            self.shares_held = 0
                            self.cost_basis = 0

       def _calculate_reward(self):
              """
              Module C Integration: Returns - Drawdown Penalty
              """
              # Calculate base return (percentage change in net worth)
              returns = (self.net_worth - self.initial_balance) / self.initial_balance
              
              # Calculate Drawdown from Peak
              drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
              
              reward = returns
              
              # Apply Penalty for Drawdown > 5%
              if drawdown > 0.05:
                     reward -= (drawdown * 2) # Weighted penalty

              return reward

       def render(self, mode='human'):
              print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Balance: {self.balance:.2f}')