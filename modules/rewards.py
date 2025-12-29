import numpy as np

class RewardFunction:
       def __init__(self, penalty_weight=2.0, transaction_cost_pct=0.001):
              """
              Args:
              penalty_weight (float): Multiplier for the drawdown penalty.
              transaction_cost_pct (float): Cost per trade (e.g., 0.1% = 0.001).
              """
              self.penalty_weight = penalty_weight
              self.transaction_cost_pct = transaction_cost_pct

       def calculate(self, current_nav, previous_nav, max_nav, action_type, current_price, shares_traded):
              """
              Calculates the scalar reward based on NAV change and risk metrics.
              
              Logic:
              1. Log Returns: Measures percentage growth.
              2. Transaction Penalty: Deducts costs based on shares traded.
              3. Drawdown Penalty: Applies a penalty if current NAV is >5% below peak.
              """
              
              # 1. Basic Return (Log return is more stable for RL training)
              # We use a small epsilon to avoid log(0)
              log_return = np.log(current_nav / (previous_nav + 1e-6))

              # 2. Transaction Costs (Friction)
              # penalty = cost_percentage * value_of_trade
              costs = (shares_traded * current_price) * self.transaction_cost_pct
              cost_penalty = costs / current_nav # Normalized to NAV

              # 3. Drawdown Penalty (Risk Metric)
              drawdown = (max_nav - current_nav) / (max_nav + 1e-6)
              drawdown_penalty = 0
              if drawdown > 0.05:
              # Quadratic penalty: the deeper the drawdown, the more it hurts
                     drawdown_penalty = (drawdown ** 2) * self.penalty_weight

              # Final Reward Formula
              reward = log_return - cost_penalty - drawdown_penalty
              
              return float(reward)