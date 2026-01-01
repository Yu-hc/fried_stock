import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Evaluator:
       def __init__(self, initial_balance=10000):
              self.initial_balance = initial_balance

       def evaluate(self, trade_log, price_history):
              """
              Evaluates the performance of a trading strategy.
              
              Args:
              trade_log (list): List of dicts containing 'step', 'action', 'price', 'balance'.
              price_history (pd.Series): The closing prices for the test period.
              
              Returns:
              dict: A performance report containing returns, Sharpe ratio, and Max Drawdown.
              """
              # 1. Convert trade log to DataFrame and map to price_history indices
              df_trades = pd.DataFrame(trade_log)
              
              # We need a continuous balance curve. We'll forward-fill the balance 
              # across the entire price_history length.
              results = pd.DataFrame(index=price_history.index)
              results['Price'] = price_history.values
              
              # Map the balance from trade_log to the steps
              balance_map = df_trades.set_index('step')['net_worth']
              results['Strategy_Balance'] = balance_map
              results['Strategy_Balance'] = results['Strategy_Balance'].ffill().fillna(self.initial_balance)
              
              # 2. Calculate Cumulative Returns
              results['Strategy_Returns'] = results['Strategy_Balance'] / self.initial_balance
              
              # Benchmark: Buy and Hold (Investing all initial balance at first price)
              first_price = price_history.iloc[0]
              results['Benchmark_Returns'] = price_history / first_price
              
              # 3. Calculate Metrics
              metrics = self._calculate_metrics(results)
              
              # 4. Generate Visualizations
              self._plot_results(results)
              
              return metrics

       def _calculate_metrics(self, df):
              # Daily Returns for Sharpe Ratio (assuming df represents daily steps)
              daily_returns = df['Strategy_Balance'].pct_change().dropna()
              
              # Total Return
              total_return = (df['Strategy_Balance'].iloc[-1] / self.initial_balance) - 1
              
              # Sharpe Ratio (Annualized, assuming 252 trading days)
              # Risk-free rate assumed at 0% for simplicity
              if daily_returns.std() != 0:
                     sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
              else:
                     sharpe_ratio = 0
              
              # Maximum Drawdown
              rolling_max = df['Strategy_Balance'].cummax()
              drawdown = (df['Strategy_Balance'] - rolling_max) / rolling_max
              max_drawdown = drawdown.min()
              
              return {
              'Total_Return': round(total_return, 4),
              'Sharpe_Ratio': round(sharpe_ratio, 4),
              'Max_Drawdown': round(max_drawdown, 4)
              }

       def _plot_results(self, df):
              plt.figure(figsize=(12, 6))
              plt.plot(df['Strategy_Returns'], label='RL Strategy', color='purple', linewidth=2)
              plt.plot(df['Benchmark_Returns'], label='Buy & Hold Benchmark', color='gray', linestyle='--')
              
              plt.title('Strategy Performance vs. Benchmark')
              plt.xlabel('Steps (Time)')
              plt.ylabel('Cumulative Return (Multiple of Initial)')
              plt.legend()
              plt.grid(True, alpha=0.3)
              plt.show()